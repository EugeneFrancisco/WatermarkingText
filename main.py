from pathlib import Path

from src.configs import build_configs
from src.watermarkers.tournament_watermarker import TournamentWatermarker

CONFIG_PATH = Path(__file__).parent / "configs" / "tournament_example.json"


def main():
    watermarker_configs = build_configs(
        "configs/watermarking_configs.json",
        "configs/tournament_configs.json"
    )
    watermarker = TournamentWatermarker(watermarker_configs)
    gemma_llm = watermarker.llm

    # Generate watermarked text for "The capital of france is". The watermarker and llm
    # assume on-device tensors, so it is our job to move the tokenized context there.
    context = gemma_llm.text_to_tokens("The capital of France is").to(watermarker.device)
    output = watermarker.generate(context, 100)
    print(gemma_llm.tokens_to_text(output))

    # Detect: low p-value means the text looks watermarked.
    generated = output[len(context):]
    p_value = watermarker.detect(generated, use_levenshtein=False)
    print("Detection p-value:", p_value.item())

if __name__ == "__main__":
    main()
