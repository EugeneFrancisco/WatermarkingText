"""
A file for the text watermarker base class.
"""
from typing import Callable
from abc import ABC, abstractmethod
from src.llms.llm import LLM
import torch
from torch.utils.data import Dataset

class Watermarker(ABC):
    """
    A Watermarker base class that future watermarkers will inherit from.
    """
    def __init__(self, configs: dict):
        self.configs = configs

        # All calculations will be done on this device.
        self.device = self.configs["device"]

        # The base LLM that will be used to generate text.
        self.llm: LLM = self.configs["llm"]

        # The length of the random key sequence xi. In Kuditipudi et al. this is denoted by n.
        self.key_length: int = self.configs["key_length"]

        # The watermarking key that will be used to seed the sample.
        self.xi = self.sample_xi()

        # How many times to resample to get a baseline for the p-value in detect.
        self.resample_size = self.configs["resample_size"]

        # The block size for sliding window detection.
        self.block_size = self.configs["block_size"]

        # The cost of inserting or deleting one token in the simple
        # Levenshtein alignment from Definition 5 of Kuditipudi et al.
        self.levenshtein_penalty = self.configs["levenshtein_penalty"]

        if self.configs.get("evaluating", False):
            # This is the number of tokens
            self.generation_length = self.configs["generation_length"]

            # This is the false positive rate that we wish to use for watermark detection.
            # During evaluation, this is used to estimate statistical power of detection.
            self.fpr = self.configs["fpr"]

    @abstractmethod
    def sample_xi(self) -> torch.Tensor:
        """
        Samples a random watermarking key sequence that is self.key_length long.
        """

    def sample_key_offset(self) -> int:
        """Sample the starting position used for one watermarked generation."""
        return torch.randint(self.key_length, (), device=self.device).item()

    @abstractmethod
    def decoder(self, key: float, logits: torch.Tensor) -> torch.Tensor:
        """
        Given a watermarking key and logits for the next sample from the underlying llm
        distribution, returns the token-id of the next sampled watermarked text.
        """

    @abstractmethod
    def generate(self, context: torch.Tensor, generation_length: int) -> torch.Tensor:
        """
        Generates watermarked text given the context which is generation_length long. This is an
        abstract method so that implementations can be taylored to the particular llm we are
        using and this way we can KV cache. This function is essentially an implementation
        of Algorithm 4 from Kuditipudi et al., which selects a random key offset before
        generating each text.
        Args:
            context: a sequence_length tensor of token ids.
            generation_length: the length of text we want to generate.
        Returns:
            A sequence_length + generation_length tensor of token ids (or shorter if a stop
            character is encountered).
        """

    @abstractmethod
    def distance(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Given a subsequence of tokens y and a subsequence of keys, finds how "close" the keys
        align with y. In Gumbel sampling, for example, the keys would be used to sample the 
        Gumbel variables and the distance could be the negated correlation of y with those
        Gumbel variables. Smaller distance should mean that the keys align more closely to
        y.
        """

    @abstractmethod
    def distance_single_token(self, y: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Given a single token id y and a single key, returns how well this y and this key
        align. This corresponds to d_0(y, key) in Kuditipudi et al.

        Args:
            y: A torch tensor of a single token id.
            key: A torch tensor of a single key.
        Returns:
            A float torch tensor representing the distance.
        """

    def distance_levenshtein(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Return the simple Levenshtein alignment cost from Definition 5.

        Matching a token and key costs distance_single_token, while inserting
        or deleting an element costs self.levenshtein_penalty.
        """
        substitution_costs = self.distance_single_token(
            y.unsqueeze(1), keys.unsqueeze(0)
        )
        return self._levenshtein_from_costs(substitution_costs)

    def _levenshtein_from_costs(
        self, substitution_costs: torch.Tensor
    ) -> torch.Tensor:
        """Compute Definition 5 from a precomputed substitution-cost matrix."""
        num_tokens, num_keys = substitution_costs.shape
        distances = torch.empty(
            (num_tokens + 1, num_keys + 1),
            device=substitution_costs.device,
            dtype=substitution_costs.dtype,
        )
        distances[:, 0] = (
            torch.arange(num_tokens + 1, device=substitution_costs.device)
            * self.levenshtein_penalty
        )
        distances[0, :] = (
            torch.arange(num_keys + 1, device=substitution_costs.device)
            * self.levenshtein_penalty
        )

        for i in range(1, num_tokens + 1):
            for j in range(1, num_keys + 1):
                substitution = (
                    distances[i - 1, j - 1]
                    + substitution_costs[i - 1, j - 1]
                )
                insertion = distances[i, j - 1] + self.levenshtein_penalty
                deletion = distances[i - 1, j] + self.levenshtein_penalty
                distances[i, j] = torch.minimum(
                    substitution, torch.minimum(insertion, deletion)
                )

        return distances[num_tokens, num_keys]

    def _test_statistic_brute_force(
        self,
        y: torch.Tensor,
        xi: torch.Tensor,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute Algorithm 3 using the supplied alignment distance."""
        min_dist = torch.tensor(float("inf"), device=y.device)
        for i in range(len(y) - self.block_size + 1):
            y_i = y[i:i + self.block_size]
            for j in range(self.key_length):
                indices = (
                    j + torch.arange(self.block_size, device=xi.device)
                ) % self.key_length
                xi_j = xi[indices]
                dist = distance_function(y_i, xi_j)
                min_dist = torch.minimum(min_dist, dist)
        return min_dist

    def test_statistic(
        self, y: torch.Tensor, xi: torch.Tensor, use_levenshtein: bool
    ) -> torch.Tensor:
        """
        Computes the best distance (see method) to align y with the stored key.
        Args:
            y: A sequence_length torch tensor of token ids.
            xi: A key sequence (xi) to test on.
            use_levenshtein: Whether to use the edit-robust Levenshtein cost.
        Returns:
            A torch Tensor float representing the minimum distance. See Algorithm 3 of Kuditipudi
            et al.
        """
        distance_function = (
            self.distance_levenshtein if use_levenshtein else self.distance
        )
        return self._test_statistic_brute_force(y, xi, distance_function)

    def detect(self, y: torch.Tensor, use_levenshtein: bool) -> torch.Tensor:
        """
        An implementation of Algorithm 2 from Kuditipudi et al. This function
        returns an estimated p-value for the probability of observing y given
        that the text was not watermarked. It does this by comparing the texts'
        test statistic on randomly sampled watermarking keys with the rest statistic
        of the watermarking key used for text generation.
        Args:
            y: A sequence_length torch tensor that we want to find the p-value of.
            use_levenshtein: Whether detection should use the Levenshtein cost.
        Returns:
            A torch.Tensor float between 0 and 1 p-value.
        """
        stats = torch.empty(self.resample_size, 1, device=self.device)
        for t in range(self.resample_size):
            this_xi = self.sample_xi()
            this_stat = self.test_statistic(y, this_xi, use_levenshtein)
            stats[t] = this_stat
        p_hat = (
            1/(self.resample_size + 1)
            * (
                1
                + torch.sum(
                    stats <= self.test_statistic(y, self.xi, use_levenshtein)
                )
            )
        )
        return p_hat

    def evaluate(self, dataset: Dataset) -> dict:
        """
        Evaluates the watermarker on the passed in dataset. Evaluation is done by performing
        watermarked text generation on the passed in dataset and seeing how well our watermarker
        detects the watermark. Detection is done at a fixed false positive rate of
        self.fpr.

        TODO, later on, we should insert a noiser between the text generation and the detector.

        Args:
            dataset: A torch dataset of data that we wish to evaluate. The dataset should be
            organized so that each element of the dataset should be a "prompt" of token ids
            of fixed size. This means the dataset's dimensions should be an N x prompt_length
            matrix of token ids.
        Returns:
            A dictionary of information of how the watermarker performed. The dictionary has the
            following structure:

            {
                power: the true positive rate observed when we reject the null hypothesis at
                    significance level self.fpr.
                mean_p: the mean p-value observed.
                percentiles: {
                    0.25: the 25th percentile p-value observed.
                    0.50: the 50th percentile p-value observed.
                    0.75: the 75th percentile p-value observed.
                }
                min: the minimum p-value observed.
                max: the maximum p-value observed.
            }
        """

