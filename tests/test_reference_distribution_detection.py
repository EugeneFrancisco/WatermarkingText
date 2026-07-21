import tempfile
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.watermarkers.watermarker import Watermarker

transformers = types.ModuleType("transformers")
transformers.AutoModelForCausalLM = object
transformers.AutoTokenizer = object
with patch.dict(sys.modules, {"transformers": transformers}):
    from src.llms.gemma import GemmaModel
    from src.watermarkers.exponential_watermarker import ExponentialWatermarker
    from src.watermarkers.its_watermarker import ITSWatermarker
    from src.watermarkers.tournament_watermarker import TournamentWatermarker


class _DetectWatermarker(Watermarker):
    def __init__(self, reference_distribution: torch.Tensor | None):
        self.device = "cpu"
        self.resample_size = 4
        self.xi = torch.tensor(2.0)
        self.reference_distribution = reference_distribution
        self.sampled_keys = iter((1.0, 3.0, 0.0, 4.0))
        self.sample_calls = 0

    def sample_xi(self) -> torch.Tensor:
        self.sample_calls += 1
        return torch.tensor(next(self.sampled_keys))

    def decoder(self, key: float, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate(
        self, context: torch.Tensor, generation_length: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def distance(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def distance_single_token(
        self, y: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def test_statistic(
        self, y: torch.Tensor, xi: torch.Tensor, use_levenshtein: bool
    ) -> torch.Tensor:
        return xi


class ReferenceDistributionDetectionTest(unittest.TestCase):
    def test_detect_uses_fixed_reference_without_resampling(self) -> None:
        watermarker = _DetectWatermarker(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        p_value = watermarker.detect(torch.tensor([0]), use_levenshtein=False)

        torch.testing.assert_close(p_value, torch.tensor(0.6))
        self.assertEqual(watermarker.sample_calls, 0)

    def test_detect_resamples_when_no_reference_is_available(self) -> None:
        watermarker = _DetectWatermarker(None)

        p_value = watermarker.detect(torch.tensor([0]), use_levenshtein=False)

        torch.testing.assert_close(p_value, torch.tensor(0.6))
        self.assertEqual(watermarker.sample_calls, 4)

    def test_non_levenshtein_reference_is_not_used_for_levenshtein(self) -> None:
        watermarker = _DetectWatermarker(torch.tensor([100.0]))

        watermarker.detect(torch.tensor([0]), use_levenshtein=True)

        self.assertEqual(watermarker.sample_calls, 4)

    def test_exponential_watermarker_loads_configured_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npy"
            np.save(path, np.array([1.25, 2.5], dtype=np.float64))
            config = {
                "device": "cpu",
                "llm": object.__new__(GemmaModel),
                "key_length": 1,
                "resample_size": 1,
                "block_size": 1,
                "levenshtein_penalty": 0,
                "reference_dist_paths": {"non_levenshtein": str(path)},
            }

            with patch.object(
                ExponentialWatermarker,
                "sample_xi",
                return_value=torch.zeros(1),
            ):
                watermarker = ExponentialWatermarker(config)

            torch.testing.assert_close(
                watermarker.reference_distribution,
                torch.tensor([1.25, 2.5]),
            )

    def test_builder_can_start_without_completed_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "non_levenshtein.npy"
            config = {
                "device": "cpu",
                "llm": object.__new__(GemmaModel),
                "key_length": 1,
                "resample_size": 1,
                "block_size": 1,
                "levenshtein_penalty": 0,
                "reference_dist_paths": {
                    "non_levenshtein": str(missing_path)
                },
                "num_rounds": 1,
                "building_reference_distribution": True,
            }

            watermarker = TournamentWatermarker(config)

            self.assertEqual(
                watermarker.reference_distribution_path, missing_path
            )
            self.assertIsNone(watermarker.reference_distribution)

    def test_every_watermarker_loads_configured_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npy"
            np.save(path, np.array([1.25, 2.5], dtype=np.float64))
            llm = object.__new__(GemmaModel)
            llm.model = SimpleNamespace(
                config=SimpleNamespace(vocab_size=8)
            )
            config = {
                "device": "cpu",
                "llm": llm,
                "key_length": 1,
                "resample_size": 1,
                "block_size": 1,
                "levenshtein_penalty": 0,
                "reference_dist_paths": {"non_levenshtein": str(path)},
                "num_rounds": 1,
            }

            for watermarker_type in (
                ExponentialWatermarker,
                ITSWatermarker,
                TournamentWatermarker,
            ):
                with self.subTest(watermarker=watermarker_type.__name__):
                    watermarker = watermarker_type(config)
                    torch.testing.assert_close(
                        watermarker.reference_distribution,
                        torch.tensor([1.25, 2.5]),
                    )


if __name__ == "__main__":
    unittest.main()
