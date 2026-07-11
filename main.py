import torch
from src.llms.gemma import GemmaModel
from src.watermarkers.tournament_watermarker import TournamentWatermarker

TOY_GEMMA = "google/gemma-3-270m"
BIG_GEMMA = "google/gemma-2-9b"

def main():
    llm_configs = {
        "model_name": TOY_GEMMA,
        "on_modal": False,
        "hf_token": None,
        "device": "mps",
        "dtype": torch.float32,
        "temperature": 1.0,   # tournament sampling needs entropy, so keep this > 0
    }
    GemmaLLM = GemmaModel(llm_configs)

    watermarker_configs = {
        "device": "mps",
        "llm": GemmaLLM,
        "key_length": 256,
        "num_rounds": 8,
        "block_size": 20,
        "resample_size": 50,
    }
    watermarker = TournamentWatermarker(watermarker_configs)

    # Generate watermarked text for "The capital of france is". The watermarker and llm
    # assume on-device tensors, so it is our job to move the tokenized context there.
    context = GemmaLLM.text_to_tokens("The capital of France is").to(watermarker.device)
    output = watermarker.generate(context, 100)
    print(GemmaLLM.tokens_to_text(output))

    # Detect: low p-value means the text looks watermarked.
    generated = output[len(context):]
    p_value = watermarker.detect(generated)
    print("Detection p-value:", p_value.item())

if __name__ == "__main__":
    main()
