from typing import Optional

from .hf_cache import resolve_hf_repo_local_dir


def resolve_hf_model_path(repo_id: str) -> Optional[str]:
    """Return an absolute local cache directory for a HF repo id, or None."""
    p = resolve_hf_repo_local_dir(repo_id)
    return str(p) if p is not None else None

