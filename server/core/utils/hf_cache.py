import os
from pathlib import Path
from typing import Optional


def _server_root() -> Path:
    # This file lives at server/core/utils/hf_cache.py
    # server root is two parents up from this file
    return Path(__file__).resolve().parents[2]


def resolve_hf_repo_local_dir(repo_id: str) -> Optional[Path]:
    """Resolve a HuggingFace repo id to a local cache directory.

    Tries the env var HUGGINGFACE_HUB_CACHE first, then falls back to
    server/models/hf_cache/hub. If a matching repo dir containing
    a config.json is found, returns its Path; otherwise None.
    """
    try:
        cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if cache_root:
            base = Path(cache_root)
        else:
            base = _server_root() / "models" / "hf_cache" / "hub"

        # Repo directories in HF cache are normalized as models--ORG--NAME
        repo_dir = base / f"models--{repo_id.replace('/', '--')}"
        cfg = repo_dir / "config.json"
        if cfg.exists():
            return repo_dir

        # Fallback: try using huggingface_hub to locate the cached file
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            cfg_path = hf_hub_download(repo_id, "config.json")
            repo_dir = Path(cfg_path).parent
            if repo_dir.exists():
                return repo_dir
        except Exception:
            pass
    except Exception:
        pass
    return None

