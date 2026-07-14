"""
A file for the text watermarker base class.
"""
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import BinaryIO, Callable
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.llms.llm import LLM


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
        An implementation of Algorithms 2 and 5 from Kuditipudi et al. This
        function returns an estimated p-value for the probability of observing y
        given that the text was not watermarked. When the watermarker has a
        non-Levenshtein reference distribution, it compares against those saved
        statistics. Otherwise, it computes null statistics using randomly sampled
        watermarking keys.
        Args:
            y: A sequence_length torch tensor that we want to find the p-value of.
            use_levenshtein: Whether detection should use the Levenshtein cost.
        Returns:
            A torch.Tensor float between 0 and 1 p-value.
        """
        observed_stat = self.test_statistic(
            y, self.xi, use_levenshtein=use_levenshtein
        )
        reference_distribution = getattr(
            self, "reference_distribution", None
        )
        if reference_distribution is not None and not use_levenshtein:
            reference_distribution = reference_distribution.to(
                device=observed_stat.device, dtype=observed_stat.dtype
            )
            return (
                1 + torch.sum(reference_distribution <= observed_stat)
            ) / (reference_distribution.numel() + 1)

        stats = torch.empty(self.resample_size, 1, device=self.device)
        for t in range(self.resample_size):
            this_xi = self.sample_xi()
            this_stat = self.test_statistic(y, this_xi, use_levenshtein)
            stats[t] = this_stat
        p_hat = (
            1/(self.resample_size + 1)
            * (
                1
                + torch.sum(stats <= observed_stat)
            )
        )
        return p_hat

    def build_null_distribution(
        self,
        dataset: Dataset,
        use_levenshtein: bool,
        save_dir: str | Path,
        checkpoint_callback: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """
        Using the passed in dataset, generates text using the dataset and then computes
        null test statistics for later bootstrapped use. Progress is saved after every
        100 computed statistics, so an interrupted run resumes from its latest checkpoint
        instead of rebuilding the distribution from scratch. The generated text is always
        self.generation_length long.

        Note that the dataset passed in here should closely reflect the data that we later
        wish to check watermarks; otherwise the distribution of these reference statistics
        will differ from the test statistics calculated in the original detect method.

        Args:
            dataset: A dataset of prompts where each element of the dataset is one prompt.
            use_levenshtein: Whether detection should use the Levenshtein cost.
            save_dir: Directory in which the completed distribution and its in-progress
                checkpoint are stored.
            checkpoint_callback: Optional callback invoked after each atomic checkpoint.
                Modal callers use this to commit the mounted Volume.
        Returns:
            A torch tensor of all the observed test statistics
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        distribution_name = (
            "levenshtein" if use_levenshtein else "non_levenshtein"
        )
        completed_path = save_path / f"{distribution_name}.npy"
        checkpoint_path = save_path / f".{distribution_name}.checkpoint.npz"
        dataset_length = len(dataset)

        if completed_path.exists():
            print(f"Loading completed reference distribution: {completed_path}")
            values = np.load(completed_path, allow_pickle=False)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                if checkpoint_callback is not None:
                    checkpoint_callback()
            return torch.from_numpy(np.array(values, copy=True)).to(self.device)

        statistics: list[float] = []
        next_dataset_index = 0
        checkpoint_interval = 100
        if checkpoint_path.exists():
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                checkpoint_dataset_length = int(checkpoint["dataset_length"])
                if checkpoint_dataset_length != dataset_length:
                    raise ValueError(
                        "Cannot resume reference distribution with a different "
                        f"dataset length: checkpoint has {checkpoint_dataset_length}, "
                        f"current dataset has {dataset_length}"
                    )
                next_dataset_index = int(checkpoint["next_dataset_index"])
                statistics = checkpoint["statistics"].astype(
                    np.float64, copy=False
                ).tolist()
            if not 0 <= next_dataset_index <= dataset_length:
                raise ValueError(
                    f"Invalid next dataset index {next_dataset_index} in "
                    f"checkpoint {checkpoint_path}"
                )
            print(
                f"Resuming {distribution_name} reference distribution at "
                f"dataset index {next_dataset_index}/{dataset_length} "
                f"with {len(statistics)} completed statistics"
            )

        def atomic_numpy_save(
            path: Path, writer: Callable[[BinaryIO], None]
        ) -> None:
            temporary_path = path.with_name(
                f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary_path.open("wb") as output_file:
                    writer(output_file)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)

        progress = tqdm(
            range(next_dataset_index, dataset_length),
            total=dataset_length,
            initial=next_dataset_index,
            desc=f"Building {distribution_name} null distribution",
        )
        statistics_at_last_checkpoint = len(statistics)
        for dataset_index in progress:
            prompt = dataset[dataset_index]
            prompt = torch.as_tensor(prompt, device=self.device)
            output = self.generate(prompt, self.generation_length)
            generated = output[len(prompt):]

            if len(generated) >= self.block_size:
                xi = self.sample_xi()
                statistic = self.test_statistic(
                    generated, xi, use_levenshtein=use_levenshtein
                )
                statistics.append(float(statistic.detach().cpu()))

            if (
                len(statistics) - statistics_at_last_checkpoint
                >= checkpoint_interval
            ):
                atomic_numpy_save(
                    checkpoint_path,
                    lambda output_file, next_index=dataset_index + 1: np.savez(
                        output_file,
                        statistics=np.asarray(statistics, dtype=np.float64),
                        next_dataset_index=np.asarray(next_index, dtype=np.int64),
                        dataset_length=np.asarray(dataset_length, dtype=np.int64),
                    ),
                )
                statistics_at_last_checkpoint = len(statistics)
                if checkpoint_callback is not None:
                    checkpoint_callback()

        if not statistics:
            raise ValueError(
                "dataset must produce at least one generation of block_size tokens"
            )

        values = np.asarray(statistics, dtype=np.float64)
        atomic_numpy_save(
            completed_path,
            lambda output_file: np.save(output_file, values),
        )
        checkpoint_path.unlink(missing_ok=True)
        if checkpoint_callback is not None:
            checkpoint_callback()

        return torch.from_numpy(values.copy()).to(self.device)

    def evaluate(self, dataset: Dataset) -> dict:
        """
        Evaluates the watermarker on the passed in dataset. Evaluation is done by performing
        watermarked text generation on the passed in dataset and seeing how well our watermarker
        detects the watermark. Detection is done at a fixed false positive rate of
        self.fpr. For each prompt, we generate self.generation_length tokens. This generated
        text is what is used to evaluate detection.

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
        p_values: list[torch.Tensor] = []

        for prompt in tqdm(dataset, desc="Evaluating"):
            prompt = torch.as_tensor(prompt, device=self.device)
            output = self.generate(prompt, self.generation_length)
            generated = output[len(prompt):]

            if len(generated) < self.block_size:
                # Skip text that is too small; for example, if we encounter an end of text token.
                continue

            p_values.append(self.detect(generated, use_levenshtein=False))

        if not p_values:
            raise ValueError("dataset must contain at least one prompt")

        values = torch.stack(p_values).flatten().float().cpu()
        percentiles = torch.quantile(
            values, torch.tensor([0.25, 0.50, 0.75])
        )

        return {
            "power": (values <= self.fpr).float().mean().item(),
            "mean_p": values.mean().item(),
            "percentiles": {
                0.25: percentiles[0].item(),
                0.50: percentiles[1].item(),
                0.75: percentiles[2].item(),
            },
            "min": values.min().item(),
            "max": values.max().item(),
        }
