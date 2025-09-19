"""
HotMem modular package (Phase 1 scaffolding)

This package provides a modular surface for the memory subsystem.
Initially, it re-exports existing implementations to avoid behavior changes.
"""

from .store import MemoryStore, Paths  # re-export for compatibility

__all__ = [
    "MemoryStore",
    "Paths",
]

