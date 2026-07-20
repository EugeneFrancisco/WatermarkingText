"""Tests for the numerical stability of tournament sampling."""

from unittest.mock import patch

import torch

from src.watermarkers.tournament_watermarker import TournamentWatermarker


class _DummyLLM:
    temperature = 1.0


def test_decoder_clamps_rounding_error_before_multinomial() -> None:
    """A reduction rounded above one must not create negative probabilities."""
    watermarker = TournamentWatermarker.__new__(TournamentWatermarker)
    watermarker.llm = _DummyLLM()
    watermarker.num_rounds = 1

    key = torch.tensor(123)
    token_ids = torch.arange(4_096)
    g_values = watermarker._g_values(token_ids, key).squeeze(-1)
    favored_token = torch.nonzero(g_values == 1)[0].item()
    unfavored_token = torch.nonzero(g_values == 0)[0].item()
    logits = torch.full((len(token_ids),), -100.0)
    logits[favored_token] = 0
    logits[unfavored_token] = -12

    original_sum = torch.sum
    original_multinomial = torch.multinomial

    def rounded_sum(values: torch.Tensor) -> torch.Tensor:
        return original_sum(values) + 1e-5

    def assert_valid_probabilities(
        probabilities: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        assert torch.isfinite(probabilities).all()
        assert torch.all(probabilities >= 0)
        assert probabilities.sum() > 0
        return original_multinomial(probabilities, num_samples)

    with (
        patch("torch.sum", side_effect=rounded_sum),
        patch("torch.multinomial", side_effect=assert_valid_probabilities),
    ):
        token = watermarker.decoder(key, logits)

    assert 0 <= token.item() < len(logits)
