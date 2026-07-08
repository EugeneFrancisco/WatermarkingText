import torch
from src.gemma import GemmaModel

TOY_GEMMA = "google/gemma-3-270m"
BIG_GEMMA = "google/gemma-2-9b"

def main():
    cfg = {
        "model_name": "google/gemma-3-270m",
        "on_modal": False,
        "hf_token": None,
        "device": "cpu",      # or "mps" on Apple Silicon
        "dtype": torch.float32,
        "temperature": 0,
    }
    m = GemmaModel(cfg)
    out = m.generate(m.text_to_tokens("The capital of France is"), 10)
    print(m.tokens_to_text(out))

if __name__ == "__main__":
    main()
