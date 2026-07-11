"""
An exponential-minimum text watermarker from Kuditipudi et al.
"""
import torch

from src.llms.gemma import GemmaModel
from src.watermarkers.watermarker import Watermarker


class ExponentialWatermarker(Watermarker):
    """
    Watermarks text with exponential-minimum sampling (EXP).

    Each element of the watermark key is one uniform random number per vocabulary
    token. The decoder samples the token whose exponential race time is smallest.
    Detection then looks for a low-cost alignment between text tokens and those
    same random numbers, as described in Section 2.5.2 of Kuditipudi et al.
    """

    def __init__(self, configs: dict):
        super().__init__(configs)

        # Match TournamentWatermarker's Gemma-specific, KV-cache generation path.
        assert isinstance(self.llm, GemmaModel)

        # EXP does not use tournament rounds, but retaining this required config
        # lets one configs dict be used unchanged for both watermarkers.
        self.num_rounds: int = self.configs["num_rounds"]

    def sample_xi(self) -> torch.Tensor:
        """Sample one Uniform(0, 1) vector over the vocabulary per key position."""
        vocab_size = self.llm.model.config.vocab_size
        return torch.rand(self.key_length, vocab_size, device=self.device)

    def decoder(self, key: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token by the exponential-minimum decoder from equation (4)."""
        probs = torch.softmax(logits / self.llm.temperature, dim=-1)
        # torch.rand samples from [0, 1), but clamping makes the logarithm safe
        # even if a caller supplies a key containing zero.
        tiny = torch.finfo(key.dtype).tiny
        exponential_times = -torch.log(key.clamp_min(tiny)) / probs
        return exponential_times.argmin()

    @torch.no_grad()
    def generate(self, context: torch.Tensor, generation_length: int) -> torch.Tensor:
        """Generate watermarked tokens while threading Gemma's KV cache."""
        model = self.llm.model
        out = model(input_ids=context.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        generated = []

        for t in range(generation_length):
            key = self.xi[t % self.key_length]
            next_token = self.decoder(key, out.logits[0, -1, :])
            generated.append(next_token)

            if next_token.item() == self.llm.tokenizer.eos_token_id:
                break

            out = model(
                input_ids=next_token.view(1, 1),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values

        return torch.cat([context, torch.stack(generated)])

    def distance(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Return EXP's practical alignment cost from equation (6)."""
        selected_keys = keys[torch.arange(len(y), device=y.device), y]
        return torch.log1p(-selected_keys).sum()

    def test_statistic(self, y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Efficiently compute Algorithm 3 for EXP's additive alignment cost."""
        block_size = self.block_size
        text_length = len(y)

        if text_length < block_size:
            raise ValueError("y must be at least block_size tokens long")

        # costs[t, j] is log(1 - xi[j, y[t]]). A diagonal of this matrix
        # corresponds to aligning consecutive text and wrapped key positions.
        costs = torch.log1p(-xi[:, y].transpose(0, 1))
        rows = torch.arange(text_length, device=y.device)
        min_cost = torch.tensor(float("inf"), device=y.device)

        for start in range(self.key_length):
            columns = (start + rows) % self.key_length
            diagonal = costs[rows, columns]
            prefix = torch.cat([
                torch.zeros(1, device=diagonal.device),
                diagonal.cumsum(0),
            ])
            window_costs = prefix[block_size:] - prefix[:-block_size]
            min_cost = torch.minimum(min_cost, window_costs.min())

        return min_cost
