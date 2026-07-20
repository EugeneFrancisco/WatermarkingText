import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.reference_distributions_script import save_reference_distributions
from src.watermarkers.watermarker import Watermarker


class _CheckpointWatermarker(Watermarker):
    """Small deterministic watermarker used to exercise checkpoint behavior."""

    def __init__(self, fail_after: int | None = None):
        self.device = "cpu"
        self.generation_length = 1
        self.block_size = 1
        self.key_length = 1
        self.generate_calls = 0
        self.fail_after = fail_after

    def sample_xi(self) -> torch.Tensor:
        return torch.zeros(1)

    def decoder(self, key: float, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate(
        self, context: torch.Tensor, generation_length: int
    ) -> torch.Tensor:
        if (
            self.fail_after is not None
            and self.generate_calls >= self.fail_after
        ):
            raise RuntimeError("simulated restart")
        self.generate_calls += 1
        return torch.cat((context, context[-1:].clone()))

    def distance(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def distance_single_token(
        self, y: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def test_statistic(
        self, y: torch.Tensor, xi: torch.Tensor, use_levenshtein: bool
    ) -> torch.Tensor:
        return y[0].float()


class _SometimesShortWatermarker(_CheckpointWatermarker):
    """Skip prompts whose token value is odd by generating no continuation."""

    def generate(
        self, context: torch.Tensor, generation_length: int
    ) -> torch.Tensor:
        self.generate_calls += 1
        if context[-1].item() % 2:
            return context
        return torch.cat((context, context[-1:].clone()))


class ReferenceDistributionResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = [torch.tensor([value]) for value in range(205)]

    def test_interrupted_distribution_resumes_at_next_dataset_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory)
            commits = []
            interrupted = _CheckpointWatermarker(fail_after=150)

            with self.assertRaisesRegex(RuntimeError, "simulated restart"):
                interrupted.build_null_distribution(
                    self.dataset,
                    len(self.dataset),
                    use_levenshtein=False,
                    save_dir=save_dir,
                    checkpoint_callback=lambda: commits.append(None),
                )

            checkpoint_path = save_dir / ".non_levenshtein.checkpoint.npz"
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                self.assertEqual(int(checkpoint["next_dataset_index"]), 100)
                np.testing.assert_array_equal(
                    checkpoint["statistics"], np.arange(100, dtype=np.float64)
                )
            self.assertEqual(len(commits), 1)

            resumed = _CheckpointWatermarker()
            values = resumed.build_null_distribution(
                self.dataset,
                len(self.dataset),
                use_levenshtein=False,
                save_dir=save_dir,
                checkpoint_callback=lambda: commits.append(None),
            )

            self.assertEqual(resumed.generate_calls, 105)
            torch.testing.assert_close(values, torch.arange(205).double())
            self.assertTrue((save_dir / "non_levenshtein.npy").exists())
            self.assertFalse(checkpoint_path.exists())

    def test_completed_distribution_is_loaded_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _CheckpointWatermarker()
            expected = first.build_null_distribution(
                self.dataset, len(self.dataset), False, directory
            )

            second = _CheckpointWatermarker(fail_after=0)
            actual = second.build_null_distribution(
                self.dataset, len(self.dataset), False, directory
            )

            self.assertEqual(second.generate_calls, 0)
            torch.testing.assert_close(actual, expected)

    def test_short_generations_do_not_count_toward_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            watermarker = _SometimesShortWatermarker()

            values = watermarker.build_null_distribution(
                self.dataset, 50, False, directory
            )

            self.assertEqual(watermarker.generate_calls, 99)
            torch.testing.assert_close(
                values, torch.arange(0, 100, 2).double()
            )
            saved = np.load(Path(directory) / "non_levenshtein.npy")
            self.assertEqual(len(saved), 50)

    def test_completed_distribution_with_wrong_size_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed_path = Path(directory) / "non_levenshtein.npy"
            np.save(completed_path, np.arange(3, dtype=np.float64))
            watermarker = _CheckpointWatermarker()

            values = watermarker.build_null_distribution(
                self.dataset, 5, False, directory
            )

            self.assertEqual(watermarker.generate_calls, 5)
            torch.testing.assert_close(values, torch.arange(5).double())
            self.assertEqual(len(np.load(completed_path)), 5)

    def test_partial_checkpoint_rejects_a_different_dataset_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            interrupted = _CheckpointWatermarker(fail_after=101)
            with self.assertRaises(RuntimeError):
                interrupted.build_null_distribution(
                    self.dataset, len(self.dataset), False, directory
                )

            resumed = _CheckpointWatermarker()
            with self.assertRaisesRegex(ValueError, "different dataset length"):
                resumed.build_null_distribution(
                    self.dataset[:-1], len(self.dataset) - 1, False, directory
                )

    def test_reference_helper_only_builds_non_levenshtein_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            watermarker = _CheckpointWatermarker()
            commits = []
            paths = save_reference_distributions(
                watermarker,
                "test_method",
                self.dataset,
                len(self.dataset),
                directory,
                lambda: commits.append(None),
            )

            expected_path = Path(directory) / "test_method/non_levenshtein.npy"
            self.assertEqual(paths, [expected_path])
            self.assertTrue(expected_path.exists())
            self.assertFalse(
                (Path(directory) / "test_method/levenshtein.npy").exists()
            )
            self.assertEqual(watermarker.generate_calls, len(self.dataset))
            self.assertEqual(len(commits), 3)


if __name__ == "__main__":
    unittest.main()
