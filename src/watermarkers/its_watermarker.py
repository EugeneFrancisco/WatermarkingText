"""An inverse-transform-sampling text watermarker from Kuditipudi et al."""
import torch

from src.llms.gemma import GemmaModel
from src.watermarkers.watermarker import Watermarker


class ITSWatermarker(Watermarker):
    """
    Watermark text with inverse transform sampling (ITS).

    ITS uses one random permutation of the vocabulary and one Uniform(0, 1)
    value per watermark-key position. The permutation chooses an ordering for
    the model CDF, and the uniform value selects a token from that CDF. This is
    the practical ITS variant in Section 2.4.2 of Kuditipudi et al.
    """

    def __init__(self, configs: dict):
        super().__init__(configs)

        # Match the existing watermarkers' Gemma-specific, KV-cache generation path.
        assert isinstance(self.llm, GemmaModel)

        # ITS does not use tournament rounds, but retaining this required config
        # allows one configs dict to be used unchanged across watermarkers.
        self.num_rounds: int = self.configs["num_rounds"]

        vocab_size = self.llm.model.config.vocab_size
        # Maps each token id to its index in the CDF ordering. The practical
        # ITS variant shares this one permutation across all key positions.
        self.permutation = torch.randperm(vocab_size, device=self.device)
        self.token_order = self.permutation.argsort()

    def sample_xi(self) -> torch.Tensor:
        """Sample one Uniform(0, 1) value for every watermark-key position."""
        return torch.rand(self.key_length, device=self.device)

    def decoder(self, key: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token by inverse-transform sampling in the permuted order."""
        probs = torch.softmax(logits / self.llm.temperature, dim=-1)
        ordered_cdf = probs[self.token_order].cumsum(dim=-1)

        # searchsorted returns the first permuted token whose CDF value is at
        # least the key, exactly implementing equation (1) in the paper.
        cdf_index = torch.searchsorted(ordered_cdf, key)
        cdf_index = cdf_index.clamp_max(len(self.token_order) - 1)
        return self.token_order[cdf_index]

    @torch.no_grad()
    def generate(self, context: torch.Tensor, generation_length: int) -> torch.Tensor:
        """Generate watermarked tokens while threading Gemma's KV cache."""
        model = self.llm.model
        out = model(input_ids=context.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        generated = []
        key_offset = self.sample_key_offset()

        for t in range(generation_length):
            key = self.xi[(key_offset + t) % self.key_length]
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
        """Return the practical ITS alignment cost from equation (3)."""
        normalized_ranks = self.permutation[y] / (len(self.permutation) - 1)
        return torch.abs(keys - normalized_ranks).sum()

    def test_statistic(self, y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Efficiently compute Algorithm 3 for ITS's additive alignment cost."""
        block_size = self.block_size
        text_length = len(y)

        if text_length < block_size:
            raise ValueError("y must be at least block_size tokens long")

        normalized_ranks = self.permutation[y] / (len(self.permutation) - 1)
        # costs[t, j] is |xi[j] - rank(y[t])|. A diagonal aligns consecutive
        # text tokens with consecutive, wrapped watermark-key positions.
        costs = torch.abs(normalized_ranks.unsqueeze(1) - xi.unsqueeze(0))
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
