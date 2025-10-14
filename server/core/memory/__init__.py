"""
Memory management system for voice agents.

Provides lazy exports to avoid importing heavyweight Pipecat dependencies
during module discovery or unit testing.
"""

__all__ = ["HotMemService"]


def __getattr__(name):
    if name == "HotMemService":
        from .hotmem_service import HotMemService

        return HotMemService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
