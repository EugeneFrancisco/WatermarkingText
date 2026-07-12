"""
A file for the text watermarker base class.
"""
from abc import ABC, abstractmethod
from src.llms.llm import LLM
import torch

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

    def test_statistic(self, y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        Computes the best distance (see method) to align y with the stored key.
        Args:
            y: A sequence_length torch tensor of token ids.
            xi: A key sequence (xi) to test on.
        Returns:
            A torch Tensor float representing the minimum distance. See Algorithm 3 of Kuditipudi
            et al.
        """
        min_dist = torch.inf
        for i in range(len(y) - self.block_size + 1):
            y_i = y[i:i + self.block_size]
            for j in range(self.key_length):
                indices = (j + torch.arange(self.block_size)) % self.key_length
                xi_j = xi[indices]
                dist = self.distance(y_i, xi_j)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def detect(self, y: torch.Tensor) -> torch.Tensor:
        """
        An implementation of Algorithm 2 from Kuditipudi et al. This function
        returns an estimated p-value for the probability of observing y given
        that the text was not watermarked. It does this by comparing the texts'
        test statistic on randomly sampled watermarking keys with the rest statistic
        of the watermarking key used for text generation.
        Args:
            y: A sequence_length torch tensor that we want to find the p-value of.
        Returns:
            A torch.Tensor float between 0 and 1 p-value.
        """
        stats = torch.empty(self.resample_size, 1, device=self.device)
        for t in range(self.resample_size):
            this_xi = self.sample_xi()
            this_stat = self.test_statistic(y, this_xi)
            stats[t] = this_stat
        p_hat = (
            1/(self.resample_size + 1)
            * (1 + torch.sum(stats <= self.test_statistic(y, self.xi)))
        )
        return p_hat
