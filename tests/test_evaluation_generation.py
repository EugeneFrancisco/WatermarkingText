import unittest

import torch

from src.watermarkers.watermarker import Watermarker


class _LLM:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, context: torch.Tensor, length: int) -> torch.Tensor:
        self.generate_calls += 1
        return torch.cat((context, torch.ones(length, dtype=torch.long)))


class _EvaluationWatermarker(Watermarker):
    def __init__(self):
        self.device = "cpu"
        self.llm = _LLM()
        self.generation_length = 1
        self.block_size = 1
        self.fpr = 0.05
        self.generate_calls = 0

    def sample_xi(self) -> torch.Tensor:
        raise NotImplementedError

    def decoder(self, key: float, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate(
        self, context: torch.Tensor, generation_length: int
    ) -> torch.Tensor:
        self.generate_calls += 1
        return torch.cat((context, torch.ones(generation_length, dtype=torch.long)))

    def distance(self, y: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def distance_single_token(
        self, y: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def detect(self, y: torch.Tensor, use_levenshtein: bool) -> torch.Tensor:
        return torch.tensor(0.5)


class EvaluationGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = [torch.tensor([0])]

    def test_evaluate_watermarks_by_default(self) -> None:
        watermarker = _EvaluationWatermarker()

        watermarker.evaluate(self.dataset)

        self.assertEqual(watermarker.generate_calls, 1)
        self.assertEqual(watermarker.llm.generate_calls, 0)

    def test_evaluate_can_generate_without_watermarking(self) -> None:
        watermarker = _EvaluationWatermarker()

        watermarker.evaluate(self.dataset, watermark=False)

        self.assertEqual(watermarker.generate_calls, 0)
        self.assertEqual(watermarker.llm.generate_calls, 1)


if __name__ == "__main__":
    unittest.main()
