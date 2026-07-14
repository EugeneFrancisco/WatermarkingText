"""Run a configurable text-watermarker evaluation on a Modal GPU.

The local preprocessed dataset is mounted read-only at
``/root/project/data/c4_realnewslike_gemma``. Hugging Face model files are
cached separately in a persistent Modal Volume.

modal run --detach modal_evaluate.py \
  --watermarker tournament \
  --num-examples 10 \
  --model-name google/gemma-2-9b

modal run --detach modal_evaluate.py \
  --build-references \
  --num-examples 1000 \
  --model-name google/gemma-2-9b

"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "watermarker-evaluation"
PROJECT_ROOT = Path("/root/project")
DATASET_PATH = PROJECT_ROOT / "data/c4_realnewslike_gemma"
HF_CACHE_PATH = Path("/cache/huggingface")
RESULTS_PATH = Path("/results")
REFERENCE_DISTRIBUTIONS_PATH = Path("/reference_distributions")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(
    "watermarking-huggingface-cache", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "watermarking-evaluation-results", create_if_missing=True
)
reference_distributions_volume = modal.Volume.from_name(
    "watermarking-reference-distributions", create_if_missing=True
)

# Keep source/configuration and generated data as separate mounts. In particular,
# this avoids uploading the local virtual environment and papers with every run.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2.0,<3",
        "torch>=2.6,<3",
        "tqdm>=4.66,<5",
        "transformers>=4.53,<6",
    )
    .env(
        {
            "HF_HOME": str(HF_CACHE_PATH),
            "HF_HUB_CACHE": str(HF_CACHE_PATH / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src", str(PROJECT_ROOT / "src"))
    .add_local_dir("configs", str(PROJECT_ROOT / "configs"))
    .add_local_dir(
        "data/c4_realnewslike_gemma",
        str(DATASET_PATH),
    )
    .add_local_dir(
        "data/reference_distributions",
        str(PROJECT_ROOT / "data/reference_distributions"),
    )
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(HF_CACHE_PATH): hf_cache},
)
def evaluate_watermarker(
    watermarker_name: str = "tournament",
    num_examples: int = 500,
    model_name: str = "google/gemma-3-270m",
) -> dict:
    """Build the selected watermarker and evaluate the first dataset examples."""
    import os
    import sys

    import torch
    from torch.utils.data import Subset

    os.chdir(PROJECT_ROOT)
    # Modal installs this app module at /root/modal_evaluate.py, so /root—not
    # PROJECT_ROOT—is normally on sys.path. The project source is mounted under
    # /root/project and must be added explicitly for `src.*` imports.
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.utils import load_c4_realnewslike_dataset
    from src.llms.gemma import GemmaModel
    from src.watermarkers.exponential_watermarker import ExponentialWatermarker
    from src.watermarkers.its_watermarker import ITSWatermarker
    from src.watermarkers.tournament_watermarker import TournamentWatermarker

    watermarker_types = {
        "exponential": (
            ExponentialWatermarker,
            PROJECT_ROOT / "configs/exponential_configs.json",
        ),
        "its": (
            ITSWatermarker,
            PROJECT_ROOT / "configs/its_configs.json",
        ),
        "tournament": (
            TournamentWatermarker,
            PROJECT_ROOT / "configs/tournament_configs.json",
        ),
    }
    normalized_name = watermarker_name.strip().lower()
    if normalized_name not in watermarker_types:
        supported = ", ".join(sorted(watermarker_types))
        raise ValueError(
            f"Unsupported watermarker {watermarker_name!r}; expected one of: "
            f"{supported}"
        )
    watermarker_class, method_config_path = watermarker_types[normalized_name]

    with (PROJECT_ROOT / "configs/watermarking_configs.json").open(
        encoding="utf-8"
    ) as config_file:
        common = json.load(config_file)
    with method_config_path.open(encoding="utf-8") as config_file:
        method_config = json.load(config_file)

    # Local configuration uses MPS. Override all hardware/authentication fields
    # before constructing Gemma so the model is created directly on the GPU.
    llm_config = common["llm"] | {
        "model_name": model_name,
        "on_modal": True,
        "hf_token": None,
        "device": "cuda",
        "dtype": torch.bfloat16,
    }
    llm_config.pop("type", None)
    llm = GemmaModel(llm_config)

    # Persist freshly downloaded weights before beginning the long evaluation.
    hf_cache.commit()

    watermarker_config = common["watermarker"] | method_config | {
        "device": "cuda",
        "llm": llm,
    }
    watermarker = watermarker_class(watermarker_config)

    dataset = load_c4_realnewslike_dataset(DATASET_PATH)
    count = min(num_examples, len(dataset))
    if count <= 0:
        raise ValueError(
            "num_examples must be positive and the dataset must be nonempty"
        )

    results = watermarker.evaluate(Subset(dataset, range(count)))
    results["num_examples_requested"] = num_examples
    results["num_examples_available"] = len(dataset)
    results["model_name"] = model_name
    results["watermarker"] = normalized_name
    # These originate in JSON and are therefore safe to return through Modal.
    # Runtime-only objects such as the model and torch dtype are intentionally
    # omitted.
    results["experiment_config"] = common["watermarker"] | method_config
    return results


@app.function(
    image=image,
    gpu="A10",
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(HF_CACHE_PATH): hf_cache,
        str(REFERENCE_DISTRIBUTIONS_PATH): reference_distributions_volume,
    },
)
def build_reference_distributions(
    num_examples: int = 1_000,
    model_name: str = "google/gemma-3-270m",
) -> list[str]:
    """Build Tournament and ITS non-Levenshtein distributions on Modal."""
    import os
    import sys

    import torch
    from torch.utils.data import Subset

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    reference_distributions_volume.reload()

    from src.data.reference_distributions_script import (
        save_reference_distributions,
    )
    from src.data.utils import load_c4_realnewslike_dataset
    from src.llms.gemma import GemmaModel
    from src.watermarkers.its_watermarker import ITSWatermarker
    from src.watermarkers.tournament_watermarker import TournamentWatermarker

    with (PROJECT_ROOT / "configs/watermarking_configs.json").open(
        encoding="utf-8"
    ) as config_file:
        common = json.load(config_file)

    llm_config = common["llm"] | {
        "model_name": model_name,
        "on_modal": True,
        "hf_token": None,
        "device": "cuda",
        "dtype": torch.bfloat16,
    }
    llm_config.pop("type", None)
    llm = GemmaModel(llm_config)
    hf_cache.commit()

    dataset = load_c4_realnewslike_dataset(DATASET_PATH)
    count = min(num_examples, len(dataset))
    if count <= 0:
        raise ValueError(
            "num_examples must be positive and the dataset must be nonempty"
        )
    dataset = Subset(dataset, range(count))

    watermarker_types = (
        (
            "tournament",
            TournamentWatermarker,
            PROJECT_ROOT / "configs/tournament_configs.json",
        ),
        ("its", ITSWatermarker, PROJECT_ROOT / "configs/its_configs.json"),
    )
    saved_paths: list[str] = []
    for name, watermarker_type, method_config_path in watermarker_types:
        method_dir = REFERENCE_DISTRIBUTIONS_PATH / name
        expected_path = method_dir / "non_levenshtein.npy"
        if expected_path.exists():
            print(f"Skipping completed reference distribution for {name}")
            saved_paths.append(str(expected_path))
            continue

        with method_config_path.open(encoding="utf-8") as config_file:
            method_config = json.load(config_file)
        config = common["watermarker"] | method_config | {
            "device": "cuda",
            "llm": llm,
        }
        watermarker = watermarker_type(config)
        paths = save_reference_distributions(
            watermarker,
            name,
            dataset,
            REFERENCE_DISTRIBUTIONS_PATH,
            reference_distributions_volume.commit,
        )
        saved_paths.extend(str(path) for path in paths)

    return saved_paths


@app.function(volumes={str(RESULTS_PATH): results_volume})
def save_results(results: dict) -> str:
    """Serialize one completed evaluation to the persistent results volume."""
    from datetime import datetime, timezone
    import re
    from uuid import uuid4

    def filename_component(value: object) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    watermarker = filename_component(results.get("watermarker", "watermarker"))
    model = filename_component(results.get("model_name", "model").split("/")[-1])
    unique_suffix = uuid4().hex[:8]
    filename = f"{timestamp}_{watermarker}_{model}_{unique_suffix}.txt"
    result_path = RESULTS_PATH / filename

    result_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_volume.commit()
    return str(result_path)


@app.function(timeout=24 * 60 * 60)
def run_and_save_evaluation(
    watermarker: str,
    num_examples: int,
    model_name: str,
) -> str:
    """Run and persist an evaluation entirely within Modal."""
    # Spawning the long GPU call gives it an independently trackable function
    # call while this remote orchestrator waits for its result.
    evaluation_call = evaluate_watermarker.spawn(
        watermarker,
        num_examples,
        model_name,
    )
    results = evaluation_call.get()
    result_path = save_results.remote(results)
    print(
        f"Saved results to Modal volume 'watermarking-evaluation-results' "
        f"at {result_path}"
    )
    return result_path


@app.local_entrypoint()
def main(
    watermarker: str = "tournament",
    num_examples: int = 1_000,
    model_name: str = "google/gemma-3-270m",
    build_references: bool = False,
) -> None:
    """Submit remote orchestration and return without waiting for completion."""
    if build_references:
        function_call = build_reference_distributions.spawn(
            num_examples,
            model_name,
        )
        print(
            "Submitted detached reference-distribution call "
            f"{function_call.object_id}"
        )
        return

    function_call = run_and_save_evaluation.spawn(
        watermarker,
        num_examples,
        model_name,
    )
    print(f"Submitted detached evaluation call {function_call.object_id}")
