"""
Warm (prefetch) models used by dev server to avoid first‑run latency.

Best‑effort: warms Kokoro MLX TTS; warms Parakeet STT if parakeet-mlx present
otherwise warms Whisper‑MLX via Pipecat service.

Usage:
  python server/tools/warm_models.py

Environment overrides:
  TTS_MODEL (default: mlx-community/Kokoro-82M-bf16)
  STT_MODEL (default: mlx-community/parakeet-tdt-0.6b-v3)
"""
from __future__ import annotations

import os
import importlib.util


def _log_ok(msg: str):
    print(f"[warm] ✅ {msg}")


def _log_skip(msg: str, e: Exception | str):
    print(f"[warm] ⚠️  {msg} skipped: {e}")


def warm_kokoro_tts():
    model = os.getenv("TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
    from mlx_audio.tts.utils import load_model
    load_model(model)
    _log_ok(f"Kokoro TTS model: {model}")


def warm_parakeet_stt():
    model = os.getenv("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
    from parakeet_mlx import from_pretrained
    from_pretrained(model)
    _log_ok(f"Parakeet STT model: {model}")


def warm_whisper_mlx_stt():
    from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
    WhisperSTTServiceMLX(model=MLXModel.TINY)
    _log_ok("Whisper-MLX STT model")


def main():
    # Always try to warm Kokoro
    try:
        warm_kokoro_tts()
    except Exception as e:
        _log_skip("Kokoro TTS warm", e)

    # Warm STT: prefer Parakeet if available; else Whisper-MLX
    try:
        if importlib.util.find_spec("parakeet_mlx"):
            try:
                warm_parakeet_stt()
            except Exception as e:
                _log_skip("Parakeet STT warm", e)
        else:
            warm_whisper_mlx_stt()
    except Exception as e:
        _log_skip("STT warm", e)


if __name__ == "__main__":
    main()

