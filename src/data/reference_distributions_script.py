"""Build C4 reference distributions for every watermarker."""

from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import Dataset, Subset

from src.watermarkers.watermarker import Watermarker


def save_reference_distributions(
    watermarker: Watermarker,
    watermarker_name: str,
    dataset: Dataset,
    output_dir: str | Path,
    flush: Callable[[], None],
) -> list[Path]:
    """Build and save both reference distributions for one watermarker."""
    method_dir = Path(output_dir) / watermarker_name
    method_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for use_levenshtein, filename in (
        (False, "non_levenshtein.npy"),
        (True, "levenshtein.npy"),
    ):
        path = method_dir / filename
        if path.exists():
            print(f"Skipping existing reference distribution: {path}")
            paths.append(path)
            continue

        values = watermarker.build_null_distribution(dataset, use_levenshtein)
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("wb") as output_file:
            np.save(output_file, values.detach().cpu().numpy())
        temporary_path.replace(path)
        flush()
        paths.append(path)

    return paths


def main() -> None:
    from src.configs import build_configs
    from src.data.utils import load_c4_realnewslike_dataset
    from src.watermarkers.exponential_watermarker import ExponentialWatermarker
    from src.watermarkers.its_watermarker import ITSWatermarker
    from src.watermarkers.tournament_watermarker import TournamentWatermarker

    dataset = load_c4_realnewslike_dataset("data/c4_realnewslike_gemma")
    # First 5_000 examples are used for the reference distribution. Make sure
    # there is no overlap during evaluation.
    dataset = Subset(dataset, range(min(5_000, len(dataset))))

    watermarker_types = (
        (
            "exponential",
            ExponentialWatermarker,
            "configs/exponential_configs.json",
        ),
        ("its", ITSWatermarker, "configs/its_configs.json"),
        ("tournament", TournamentWatermarker, "configs/tournament_configs.json"),
    )
    for name, watermarker_type, method_config in watermarker_types:
        config = build_configs("configs/watermarking_configs.json", method_config)
        watermarker = watermarker_type(config)
        save_reference_distributions(
            watermarker,
            name,
            dataset,
            "data/reference_distributions",
            lambda: None,
        )


if __name__ == "__main__":
    main()
