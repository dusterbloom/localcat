"""
Model path resolution utilities for STT/TTS models.
"""

import os
from pathlib import Path
from loguru import logger


def resolve_parakeet_model_path(model_id_or_path: str) -> str:
    """
    Resolve Parakeet model path for production (Tauri bundle) vs development.

    In production with HF_HUB_OFFLINE=1, HuggingFace can't resolve repo IDs
    to cached models. This function detects production mode and returns the
    absolute path to the bundled model when available.

    Args:
        model_id_or_path: HuggingFace model ID or local path

    Returns:
        Absolute path if a bundled model is detected, otherwise returns input unchanged
    """
    default_model_id = "mlx-community/parakeet-tdt-0.6b-v3"

    if "TAURI_RESOURCE_DIR" in os.environ and model_id_or_path == default_model_id:
        hf_home = Path(os.environ.get("HF_HOME", ""))
        if hf_home.exists():
            bundled_model = hf_home / "hub" / "models--mlx-community--parakeet-tdt-0.6b-v3"
            if bundled_model.exists():
                logger.debug(f"Resolved Parakeet model to bundled path: {bundled_model}")
                return str(bundled_model)

    return model_id_or_path

