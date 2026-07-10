"""
This file defines a base class for LLMs.
"""
import torch

from abc import ABC, abstractmethod

class LLM(ABC):
    """
    An LLM base class which allows very simple interfacing with an LLM for watermarking
    experiments. All we need in this class is to encode/decode text into/from tokens.
    And to get the pre-softmax logits during text generation to control how text will
    be sampled.
    """
    def __init__(self, configs: dict):
        self.configs = configs
        self.device: str = configs["device"]

    @abstractmethod
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """
        Given a batched context window, returns the batched pre-softamx
        logits for the next tokens.

        Args:
            context: a B x sequence_length tensor of context token ids.

        Returns:
            A B x vocab_size tensor of pre-softmax logits.
        """

    @abstractmethod
    def generate(self, context: torch.Tensor, length: int) -> torch.Tensor:
        """
        Given a context window for a single batch, generates text until
        either length tokens have been generated or a stop character is reached.

        Args:
            context: a sequence_length tensor of context token ids.

        Returns:
            A (sequence_length + length) tensor of token ids. The sequence
            might actually be shorter than sequence_length + length if a stop
            character is generated.
        """

    @abstractmethod
    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """
        Given a tensor of tokens, converts those tokens into readable words.

        Args:
            tokens: a sequence_length tensor of token ids.

        Returns:
            A string of what those tokens decode to in text.
        """

    @abstractmethod
    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Given a string of text, returns the sequence of tokens that the text
        encodes to under the tokenizer that the LLM is using.

        Args:
            text: a string of text we with to encode into tokens.

        Returns:
            A sequence_length tensor of token ids.
        """
