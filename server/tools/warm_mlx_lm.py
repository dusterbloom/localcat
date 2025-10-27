"""
Warm (prefetch) one or more MLX-LM models into the Hugging Face cache.

Why: Avoid first-run download and speed up cold start for Direct MLX backend.

Usage:
  python -m tools.warm_mlx_lm --model mlx-community/gemma3n:e2b
  python -m tools.warm_mlx_lm --model mlx-community/Qwen3-VL-4B-Instruct-4bit
  python -m tools.warm_mlx_lm --file models.txt   # one model id per line

Environment:
  - HF_HOME (optional) set a shared cache dir, e.g.
      export HF_HOME=$HOME/AI-Models/shared/huggingface
  - HF_HUB_OFFLINE=1 to run offline after models are cached.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable


def _log(msg: str):
    print(f"[warm-mlx] {msg}")


def _iter_models(args: argparse.Namespace) -> Iterable[str]:
    if args.model:
        for m in args.model:
            yield m
    if args.file:
        from pathlib import Path
        p = Path(args.file)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line


def warm_model(model_id: str) -> None:
    try:
        import mlx_lm
    except Exception as e:
        _log("mlx-lm not installed. Install with: pip install mlx-lm")
        raise SystemExit(1) from e

    try:
        _log(f"Loading: {model_id}")
        mlx_lm.load(model_id)
        _log(f"OK: {model_id}")
    except Exception as e:
        _log(f"FAIL: {model_id}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warm MLX-LM models into cache")
    parser.add_argument(
        "--model",
        action="append",
        help="Model id (Hugging Face). Repeatable.",
    )
    parser.add_argument(
        "--file",
        help="File with one model id per line",
    )

    args = parser.parse_args(argv)

    models = list(_iter_models(args))
    if not models:
        _log("No models provided. Use --model or --file.")
        return 2

    for mid in models:
        warm_model(mid)

    _log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

