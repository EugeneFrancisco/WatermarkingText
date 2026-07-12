from torch.utils.data import Subset

from src.configs import build_configs
from src.data.utils import load_c4_realnewslike_dataset
from src.watermarkers.tournament_watermarker import TournamentWatermarker
from src.watermarkers.exponential_watermarker import ExponentialWatermarker


def main() -> None:
    # watermarker_configs = build_configs(
    #     "configs/watermarking_configs.json",
    #     "configs/exponential_configs.json",
    # )
    # watermarker = ExponentialWatermarker(watermarker_configs)

    watermarker_configs = build_configs(
        "configs/watermarking_configs.json",
        "configs/tournament_configs.json",
    )
    watermarker = TournamentWatermarker(watermarker_configs)

    dataset = load_c4_realnewslike_dataset("data/c4_realnewslike_gemma")
    demonstration_data = Subset(dataset, range(min(10, len(dataset))))
    results = watermarker.evaluate(demonstration_data)
    print(results)


if __name__ == "__main__":
    main()
