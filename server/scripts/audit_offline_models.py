#!/usr/bin/env python3
"""
Audit required offline models/assets based on server/.env and known defaults.

Checks that all referenced Hugging Face models exist in the bundled offline cache
`server/models/hf_cache/hub/` and that Kokoro MLX/ONNX assets are present.

Prints a concise report of FOUND/MISSING items and suggests snapshot_download
commands to fetch missing repos before packaging.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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


def any_snapshot_exists(repo: str) -> bool:
    """Check if any snapshot directory exists for a HF repo under hf_cache/hub."""
    # Convert repo "owner/name" -> models--owner--name
    owner, name = repo.split("/", 1)
    repo_dir = HF_HUB_DIR / f"models--{owner}--{name}"
    snaps = repo_dir / "snapshots"
    return snaps.exists() and any(p.is_dir() for p in snaps.iterdir())


def main() -> int:
    env = read_env(SERVER_DIR / ".env")

    # Determine engines
    stt_engine = env.get("VOICE_AGENT_STT_ENGINE", env.get("STT_ENGINE", "parakeet_batch")).lower()
    tts_engine = env.get("VOICE_AGENT_TTS_ENGINE", env.get("TTS_ENGINE", "kokoro_mlx")).lower()

    # Required repos based on env
    hf_repos: set[str] = set()

    # STT
    if "whisper" in stt_engine:
        hf_repos.add("mlx-community/whisper-medium-mlx")
    else:
        hf_repos.add("mlx-community/parakeet-tdt-0.6b-v3")

    # Smart turn VAD (optional)
    vad_model = env.get("VAD_SMART_TURN_MODEL_PATH", "pipecat-ai/smart-turn-v2").strip()
    if vad_model:
        hf_repos.add(vad_model)

    # Speaker recognition via SpeechBrain
    hf_repos.add("speechbrain/spkrec-ecapa-voxceleb")

    # Kokoro MLX voices are shipped as .pt in server/models/kokoro-mlx/voices
    # ONNX fallback assets are shipped in server/models/kokoro/

    missing = []
    print("\n=== Offline Model Audit ===")
    print(f"- STT engine: {stt_engine}")
    print(f"- TTS engine: {tts_engine}")
    print(f"- HF cache dir: {HF_HUB_DIR}")

    for repo in sorted(hf_repos):
        ok = any_snapshot_exists(repo)
        status = "FOUND" if ok else "MISSING"
        print(f"  {status:7}  {repo}")
        if not ok:
            missing.append(repo)

    # Kokoro MLX/ONNX assets
    kokoro_mlx_dir = SERVER_DIR / "models" / "kokoro-mlx"
    kokoro_onnx_dir = SERVER_DIR / "models" / "kokoro"
    kokoro_mlx_ok = (
        (kokoro_mlx_dir / "config.json").exists()
        and (kokoro_mlx_dir / "kokoro-v1_0.safetensors").exists()
    )
    kokoro_onnx_ok = (
        (kokoro_onnx_dir / "kokoro-v1.0.onnx").exists()
        and (kokoro_onnx_dir / "voices-v1.0.bin").exists()
    )
    print(f"  {'FOUND' if kokoro_mlx_ok else 'MISSING'}  kokoro-mlx weights (server/models/kokoro-mlx)")
    print(f"  {'FOUND' if kokoro_onnx_ok else 'MISSING'}  kokoro-onnx weights (server/models/kokoro)")

    if missing:
        print("\nMissing HF repos. To fetch snapshots offline, run:")
        print("  pip install huggingface_hub")
        for repo in missing:
            print(
                "  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{repo}', local_dir='{HF_HUB_DIR}/models--{repo.replace('/', '--')}')\""
            )
        print("\nAfter downloads complete, rebuild the app:")
        print("  cd app/src-tauri && npx tauri build")
        return 1

    print("\nAll required models are present for offline run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

