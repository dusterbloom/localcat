"""
Compatibility wrapper for the existing MemoryStore implementation.
Phase 1 modularization step 1A: provide a stable import path.
"""

from typing import Optional

# Re-export from the current implementation to avoid behavior changes
from memory_store import MemoryStore, Paths  # type: ignore

__all__ = [
    "MemoryStore",
    "Paths",
]
