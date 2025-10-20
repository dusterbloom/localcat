#!/usr/bin/env python3
"""
Fetch and cache all required Hugging Face models for a fully offline run,
based on server/.env and sensible defaults.

Usage:
  server/.venv/bin/python server/scripts/fetch_offline_models.py

Notes:
  - This script requires network access only during fetch.
  - It writes snapshots under server/models/hf_cache/hub/ in the exact layout
    that the app expects.
  - After fetch completes, rebuild the app and it will run offline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Set

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "server"
HF_HUB_DIR = SERVER_DIR / "models" / "hf_cache" / "hub"


def read_env(env_path: Path) -> dict[str, str]:
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def collect_required_repos(env: dict[str, str]) -> Set[str]:
    repos: Set[str] = set()

    # STT
    stt_engine = env.get("VOICE_AGENT_STT_ENGINE", env.get("STT_ENGINE", "parakeet_batch")).lower()
    if "whisper" in stt_engine:
        repos.add("mlx-community/whisper-medium-mlx")
    else:
        repos.add("mlx-community/parakeet-tdt-0.6b-v3")

    # VAD smart turn (optional but commonly enabled)
    vad_model = env.get("VAD_SMART_TURN_MODEL_PATH", "pipecat-ai/smart-turn-v2").strip()
    if vad_model:
        repos.add(vad_model)

    # SpeechBrain speaker recognition
    repos.add("speechbrain/spkrec-ecapa-voxceleb")

    return repos


def fetch_repo(repo: str) -> None:
    owner, name = repo.split("/", 1)
    out_dir = HF_HUB_DIR / f"models--{owner}--{name}"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {repo} -> {out_dir}")
    snapshot_download(repo_id=repo, local_dir=str(out_dir))


def main() -> int:
    env = read_env(SERVER_DIR / ".env")

    # Ensure offline mode disabled for fetch
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    repos = collect_required_repos(env)
    print("Fetching required repos for offline cache:\n  - " + "\n  - ".join(sorted(repos)))

    for repo in sorted(repos):
        try:
            fetch_repo(repo)
        except Exception as e:
            print(f"[fetch] ERROR: {repo}: {e}")
            return 1

    print("\nAll required repos fetched. Rebuild the app to include them:")
    print("  cd app/src-tauri && npx tauri build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

