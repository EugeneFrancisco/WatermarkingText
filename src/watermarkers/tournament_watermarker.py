"""
This file defines an inheritor of the watermarker class that implements a text watermarker
which uses the tournament sampling method from SynthID-text combined with the detection
method from Kuditipudi et al.
"""
import torch
from src.watermarkers.watermarker import Watermarker
from src.llms.gemma import GemmaModel

class TournamentWatermarker(Watermarker):
    # pylint: disable=W1401
    r"""
    The TournamentWatermarker is a form of text watermarker which takes elements from both
    SynthID-text and from Kuditipudi et al. Here is how the TournamentWatermarker samples
    and later detects watermarked text:

    Sampling:
        The TournamentWatermarker uses a variant of Tournament sampling from SynthID-text.
        Given context y[:t - 1], we wish to generate token y[t] and we are equipped with
        an llm distribution p(. | y[:t - 1]). To do so, we use a random seed xi[t] to generate
        m "random function" g_1, ..., g_m. We then sample M = 2^m candidate tokens (with
        replacement) from p. These tokens are randomly paired up into M/2 pairs and, in this
        first tournament layer, the token with higher score under g_1 is selected to move on
        to the next layer. In the next layer, the M/2 remaining tokens are split into M/4
        pairs and the process repeats with g_2.
    
    Watermark Detection:
        The goal here is, given a string of text y that is possibly watermarked, to return
        an estimate for the probability of observing y given it was not watermarked. We use
        a variant of watermark detection from Kuditipudi et al. Let y' be a substring of y
        of length k and xi a substring of seed keys of length k. Define

            Score(y, xi) = 1/(mk) * \sum_{t = 1}^k\sum_{\ell = 1}^m g_\ell(y_t, xi_t).
        
        We then use Algorithm 2 and Algorithm 3 from Kuditipudi et al where we use the negated
        Score in place of d. In particular,

            d(y^i, xi^j) = Score(y^i, xi^j).
    """
    # A large prime used as the modulus for the g-value hash. Keys are sampled in
    # [0, _PRIME) so that every intermediate product below stays within int64.
    _PRIME = 2_147_483_647  # 2^31 - 1

    def __init__(self, configs: dict):
        super().__init__(configs)

        # This assertion is here because the generate method of this class will take advantage
        # of KV caching and this requires us to know what the model we are using is.
        assert isinstance(self.llm, GemmaModel)

        # The number of tokens that compete under a single g in a single layer.
        # If this number is 2, then the original M candidate tokens will be split
        # into groups of 2.
        self.num_participants = 2

        # This is the number of rounds of tournament sampling that are done. In the SynthID-text
        # paper, this is denoted by m.
        self.num_rounds = self.configs["num_rounds"]

        # This is the number of candidate tokens that we initially sample from the llm
        # distribution.
        self.num_samples = self.num_participants ** self.num_rounds

    def sample_xi(self) -> torch.Tensor:
        """
        Samples a random watermarking key sequence that is self.key_length long. Each key
        is an integer seed that primes the tournament functions g_1, ..., g_m at one position.
        """
        return torch.randint(0, self._PRIME, (self.key_length,), device=self.device)

    def _g_values(self, tokens: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Vectorised tournament functions. Given token ids and keys (broadcastable against each
        other), returns a tensor with an extra trailing dimension of size self.num_rounds whose
        [..., l] entry is g_{l + 1}(token, key): a deterministic Bernoulli(0.5) value in {0, 1}.

        The value is a cheap integer hash of (token, key, layer) mapped to a pseudo-uniform
        number in [0, 1) and thresholded at 0.5. Replacing SynthID-text's context-based seed
        with the pre-chosen key follows Kuditipudi et al.
        """
        tokens = tokens.to(torch.int64).unsqueeze(-1)                       # (..., 1)
        keys = keys.to(torch.int64).unsqueeze(-1)                           # (..., 1)
        layers = torch.arange(1, self.num_rounds + 1, device=tokens.device)  # (m,)

        z = (tokens * 15485863 + keys * 32452843 + layers * 49979687) % self._PRIME
        z = (z * z) % self._PRIME
        z = (z * 32452843) % self._PRIME
        z = (z * z) % self._PRIME
        # Threshold the pseudo-uniform value z / _PRIME at 0.5 in exact integer
        # arithmetic (2*z > _PRIME). This avoids float64 (unsupported on MPS) and
        # keeps the g-values deterministic. 2*z < 2^32 stays within int64.
        return (z * 2 > self._PRIME).to(torch.float32)  # (..., m)

    def decoder(self, key: float, logits: torch.Tensor) -> torch.Tensor:
        """
        Given a watermarking key and logits for the next sample from the underlying llm
        distribution, returns the token-id of the next sampled watermarked text. This is
        done using tournament sampling on the probability distribution associated with logits.

        Rather than materialising the 2^m leaves of the tournament, propagate the
        winner distribution through one layer at a time. For a binary tournament
        with Bernoulli g-values, one layer transforms the probability of token x as

            p'(x) = p(x) * (1 + g(x) - E_p[g]).

        This has exactly the same output distribution as explicitly sampling and
        running the tournament, while using memory linear in the vocabulary size.
        """
        # Accumulate the recurrence in float32 even when model logits use a lower
        # precision. Later layers can otherwise underflow for low-probability tokens.
        probs = torch.softmax(
            logits.to(torch.float32) / self.llm.temperature,
            dim=-1,
        )

        token_ids = torch.arange(logits.shape[-1], device=logits.device)
        key_tensor = torch.as_tensor(key, device=logits.device)
        g_values = self._g_values(token_ids, key_tensor)  # (vocab_size, m)

        for layer in range(self.num_rounds):
            g = g_values[:, layer]
            # Mathematically g_mass is in [0, 1], but CUDA's parallel float32
            # reduction can round it just outside that interval. With many
            # rounds that makes some entries slightly negative, which causes
            # torch.multinomial to trigger a device-side assertion.
            g_mass = torch.sum(probs * g).clamp(0, 1)
            probs = probs * (1 + g - g_mass)
            probs.clamp_min_(0)

            # The recurrence is normalized analytically. Renormalizing limits
            # floating-point drift across many tournament layers.
            probs = probs / probs.sum()

        return torch.multinomial(probs, 1).squeeze(0)  # 0-dim token id

    @torch.no_grad()
    def generate(self, context: torch.Tensor, generation_length: int) -> torch.Tensor:
        """
        Generates watermarked text given the context which is generation_length long.
        This function is essentially an implementation of Algorithm 1 from Kuditipudi et al. except
        that it takes advantage of K/V caching to make generation quicker.
        Args:
            context: a sequence_length tensor of token ids.
            generation_length: the length of text we want to generate.
        Returns:
            A sequence_length + generation_length tensor of token ids (or shorter if a stop
            character is encountered).
        """
        model = self.llm.model

        # Prime the KV cache on the full context, then feed one token at a time.
        out = model(input_ids=context.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        generated = []
        key_offset = self.sample_key_offset()

        for t in range(generation_length):
            logits = out.logits[0, -1, :]  # (vocab,)
            key = self.xi[(key_offset + t) % self.key_length]
            next_token = self.decoder(key, logits)  # 0-dim
            generated.append(next_token)

            if next_token.item() == self.llm.tokenizer.eos_token_id:
                break

            out = model(input_ids=next_token.view(1, 1), past_key_values=past, use_cache=True)
            past = out.past_key_values

        return torch.cat([context, torch.stack(generated)])

    def distance(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Given a subsequence of tokens y and a subsequence of keys, finds how "close" the keys
        align with y. In Gumbel sampling, for example, the keys would be used to sample the
        Gumbel variables and the distance could be the negated correlation of y with those
        Gumbel variables. Smaller distance should mean that the keys align more closely to
        y.

        Here the alignment is the negated SynthID-text Score: since tournament sampling biases
        the selected tokens towards high g-values, watermarked text scores highly, so the more
        aligned the keys are the smaller (more negative) this distance is.
        """
        return self.distance_single_token(y, keys).mean()

    def distance_single_token(
        self, y: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """Return the negated mean tournament score for one token and key."""
        return -self._g_values(y, key).mean(dim=-1)

    def test_statistic(
        self, y: torch.Tensor, xi: torch.Tensor, use_levenshtein: bool
    ) -> torch.Tensor:
        """
        Efficient override of the base test statistic (Algorithm 3 of Kuditipudi et al.).

        The base class recomputes the block distance for every (y-substring, xi-substring)
        pair, which repeatedly re-sums the same g-values for overlapping blocks. Instead we

            1. Pre-compute a (len(y), key_length) score matrix S where
                   S[a, b] = -sum_{l=1}^m g_l(y[a], xi[b]),
               i.e. the negated per-position g-value sum for each (token, key) pair.
            2. For the ordinary distance, slide a length-block_size window diagonally
               through S. For the Levenshtein distance, run dynamic programming on each
               relevant block_size-by-block_size submatrix of S.

        In both cases each g-value is computed only once, including when overlapping
        Levenshtein alignments reuse the same token-key substitution costs.
        """
        k = self.block_size
        n = self.key_length
        len_y = len(y)

        if len_y < k:
            raise ValueError("y must be at least block_size tokens long")

        # Convert the per-token mean back to a sum so the normalization below
        # remains identical to distance().
        scores = self.distance_single_token(
            y.view(-1, 1), xi.view(1, -1)
        ) * self.num_rounds  # (len_y, n)

        if use_levenshtein:
            min_dist = torch.tensor(float("inf"), device=y.device)
            block_indices = torch.arange(k, device=y.device)

            for text_start in range(len_y - k + 1):
                text_indices = text_start + block_indices
                for key_start in range(n):
                    key_indices = (key_start + block_indices) % n
                    # Definition 5 uses the per-token mean score as d_0, so undo
                    # the sum-over-rounds representation used by scores.
                    substitution_costs = (
                        scores[text_indices.unsqueeze(1), key_indices.unsqueeze(0)]
                        / self.num_rounds
                    )
                    dist = self._levenshtein_from_costs(substitution_costs)
                    min_dist = torch.minimum(min_dist, dist)

            return min_dist

        rows = torch.arange(len_y, device=y.device)
        min_sum = torch.tensor(float("inf"), device=y.device)
        # Outer loop over the xi offset i. For offset i the diagonal picks column (i + t) % n
        # for row t, wrapping xi around the key sequence. The inner sliding-window minimum over
        # start index j is done with a cumulative sum: each length-k window sum is the running
        # sum with the left-behind element subtracted and the newly included one added.
        for i in range(n):
            cols = (i + rows) % n
            diag = scores[rows, cols]  # (len_y,) diagonal starting at xi offset i
            prefix = torch.cat([torch.zeros(1, device=diag.device), diag.cumsum(0)])
            window_sums = prefix[k:] - prefix[:-k]  # (len_y - k + 1,) length-k block sums
            min_sum = torch.minimum(min_sum, window_sums.min())

        return min_sum / (k * self.num_rounds)
