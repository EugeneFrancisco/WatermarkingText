"""Build C4 reference distributions for every watermarker."""

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from src.watermarkers.watermarker import Watermarker


def save_reference_distributions(
    watermarker: Watermarker,
    watermarker_name: str,
    dataset: Dataset,
    num_statistics: int,
    output_dir: str | Path,
    flush: Callable[[], None],
) -> list[Path]:
    """Build the resumable non-Levenshtein distribution for one watermarker."""
    method_dir = Path(output_dir) / watermarker_name
    path = method_dir / "non_levenshtein.npy"
    watermarker.build_null_distribution(
        dataset,
        num_statistics,
        use_levenshtein=False,
        save_dir=method_dir,
        checkpoint_callback=flush,
    )
    return [path]


def main() -> None:
    from src.configs import build_configs
    from src.data.utils import load_c4_realnewslike_dataset
    from src.watermarkers.exponential_watermarker import ExponentialWatermarker
    from src.watermarkers.its_watermarker import ITSWatermarker
    from src.watermarkers.tournament_watermarker import TournamentWatermarker

    dataset = load_c4_realnewslike_dataset(
        "data/c4_realnewslike_gemma_reference"
    )
    num_statistics = 5_000

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
        config["building_reference_distribution"] = True
        watermarker = watermarker_type(config)
        save_reference_distributions(
            watermarker,
            name,
            dataset,
            num_statistics,
            "data/reference_distributions",
            lambda: None,
        )


if __name__ == "__main__":
    main()
