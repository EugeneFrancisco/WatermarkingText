"""Utilities for building and loading fixed-token C4 evaluation datasets."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


C4_REALNEWSLIKE_VALIDATION_URL = (
    "https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/"
    "c4-validation.00000-of-00001.json.gz"
)
DEFAULT_OUTPUT_DIR = Path("data/c4_realnewslike_gemma")
DEFAULT_TOKENIZER = "google/gemma-3-270m"


class FixedTokenDataset(Dataset):
    """A read-only dataset backed by an ``N x prompt_length`` NumPy array."""

    def __init__(self, path: str | Path, mmap: bool = True):
        load_mode = "r" if mmap else None
        self.input_ids = np.load(Path(path) / "input_ids.npy", mmap_mode=load_mode)
        if self.input_ids.ndim != 2:
            raise ValueError(
                f"input_ids.npy must be two-dimensional, got {self.input_ids.shape}"
            )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        # Copy because memory-mapped arrays are read-only and torch warns when a
        # tensor aliases non-writable NumPy memory.
        return torch.from_numpy(np.array(self.input_ids[index], copy=True)).long()


def load_c4_realnewslike_dataset(
    path: str | Path = DEFAULT_OUTPUT_DIR,
) -> FixedTokenDataset:
    """Load the preprocessed prompts expected by ``Watermarker.evaluate``."""
    return FixedTokenDataset(path)


def stream_c4_data(
    prompt_length: int,
    generation_length: int,
    save_path: str | Path,
    num_examples: int = 1_000,
) -> Path:
    """Stream and save a Gemma-tokenized C4 realnewslike evaluation sample.

    The resulting directory contains ``input_ids.npy`` with fixed-length
    prompts, ``continuations.npy`` with the held-out tokens that immediately
    follow each prompt, and ``metadata.json`` describing the preprocessing.

    Args:
        prompt_length: Number of tokens in each saved prompt.
        generation_length: Number of held-out continuation tokens per prompt.
        save_path: Directory in which to save the arrays and metadata.
        num_examples: Number of prompt-continuation pairs to save.

    Returns:
        The path containing the saved dataset.
    """
    return build_c4_realnewslike_dataset(
        output_dir=save_path,
        num_samples=num_examples,
        prompt_length=prompt_length,
        continuation_length=generation_length,
    )


def _iter_c4_texts(url: str) -> Iterator[str]:
    """Yield document text from a gzipped C4 JSON-lines shard."""
    request = urllib.request.Request(url, headers={"User-Agent": "WatermarkingText/1.0"})
    with urllib.request.urlopen(request) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            for raw_line in compressed:
                record = json.loads(raw_line)
                yield record["text"]


def build_c4_realnewslike_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    num_samples: int = 1_000,
    prompt_length: int = 50,
    continuation_length: int = 50,
    seed: int = 0,
    source_url: str = C4_REALNEWSLIKE_VALIDATION_URL,
) -> Path:
    """Build a deterministic tokenizer-specific sample of C4 realnewslike.

    Each retained row consists of the final ``continuation_length`` document
    tokens and the ``prompt_length`` tokens immediately preceding them. A
    reservoir sample makes every sufficiently long validation document equally
    likely to be selected without loading the shard into memory.
    """
    if min(num_samples, prompt_length, continuation_length) <= 0:
        raise ValueError("sample count and token lengths must all be positive")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rng = random.Random(seed)
    reservoir: list[tuple[list[int], list[int]]] = []
    eligible_documents = 0
    required_length = prompt_length + continuation_length

    for text in _iter_c4_texts(source_url):
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < required_length:
            continue

        eligible_documents += 1
        split = len(token_ids) - continuation_length
        item = (
            token_ids[split - prompt_length : split],
            token_ids[split:],
        )
        if len(reservoir) < num_samples:
            reservoir.append(item)
        else:
            replacement = rng.randrange(eligible_documents)
            if replacement < num_samples:
                reservoir[replacement] = item

    if len(reservoir) < num_samples:
        raise RuntimeError(
            f"requested {num_samples} samples but found only {len(reservoir)} "
            "eligible documents"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_ids = np.asarray([item[0] for item in reservoir], dtype=np.int32)
    continuations = np.asarray([item[1] for item in reservoir], dtype=np.int32)
    np.save(output_path / "input_ids.npy", input_ids)
    np.save(output_path / "continuations.npy", continuations)

    metadata = {
        "source": "allenai/c4",
        "source_url": source_url,
        "config": "realnewslike",
        "split": "validation",
        "tokenizer": tokenizer_name,
        "tokenizer_vocab_size": len(tokenizer),
        "add_special_tokens": True,
        "prompt_length": prompt_length,
        "continuation_length": continuation_length,
        "num_samples": num_samples,
        "eligible_documents": eligible_documents,
        "seed": seed,
        "sampling": "reservoir",
        "array_dtype": str(input_ids.dtype),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed-token Gemma dataset from C4 realnewslike."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--prompt-length", type=int, default=50)
    parser.add_argument("--continuation-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = build_c4_realnewslike_dataset(
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        num_samples=args.num_samples,
        prompt_length=args.prompt_length,
        continuation_length=args.continuation_length,
        seed=args.seed,
    )
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()
