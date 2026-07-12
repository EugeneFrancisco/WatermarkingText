"""Utilities for loading experiment configuration from JSON files."""

import json
from pathlib import Path
from typing import Any

import torch

from src.llms.gemma import GemmaModel


_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_LLM_CLASSES = {
    "gemma": GemmaModel,
}


def _build_llm(config: dict[str, Any]):
    """Construct the configured LLM, converting JSON-only values as needed."""
    config = config.copy()
    llm_type = config.pop("type", None)
    if llm_type not in _LLM_CLASSES:
        supported = ", ".join(sorted(_LLM_CLASSES))
        raise ValueError(
            f"Unsupported LLM type {llm_type!r}; expected one of: {supported}"
        )

    dtype_name = config.get("dtype")
    if dtype_name not in _DTYPES:
        supported = ", ".join(sorted(_DTYPES))
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}; expected one of: {supported}"
        )
    config["dtype"] = _DTYPES[dtype_name]
    return _LLM_CLASSES[llm_type](config)


def build_watermarker_configs(json_path: str | Path) -> dict[str, Any]:
    """Build a watermarker constructor dictionary from a JSON config file.

    The file must contain a ``watermarker`` object and an ``llm`` object. All
    entries in ``watermarker`` are passed through, which allows individual
    watermarker implementations to add settings such as ``num_rounds``. The
    ``llm`` object is used to construct the model placed under the ``llm`` key.
    """
    path = Path(json_path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")

    watermarker_config = config.get("watermarker")
    llm_config = config.get("llm")
    if not isinstance(watermarker_config, dict):
        raise ValueError("Config must contain a 'watermarker' object")
    if not isinstance(llm_config, dict):
        raise ValueError("Config must contain an 'llm' object")

    result = watermarker_config.copy()
    result["llm"] = _build_llm(llm_config)
    return result

def _build_one(json_path: str | Path) -> dict[str, Any]:
    """
    Converts the json in jason_path into a dict that is returned. This can be used for
    any config files that contain specific configurations relevant to only one type of watermarker.
    """
    path = Path(json_path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


def build_configs(watermarker_configs_path: str | Path, *other_configs_paths: str| Path) -> dict:
    """
    Given a path to a general watermarker_config and any number of paths to other config files,
    returns a dict which is the union of those files. The watermarker config file is built using
    build_watermarker_configs.
    """
    watermarker_configs = build_watermarker_configs(watermarker_configs_path)
    for other_path in other_configs_paths:
        watermarker_configs |= _build_one(other_path)

    return watermarker_configs
