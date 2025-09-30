"""
Memory management system for voice agents.

Components:
- hotpath_processor: Core memory processing with USGS pattern extraction
- memory_store: Persistent memory storage and retrieval
- session_tracker: Session-based memory management
- hotmem_service: Pipecat-compatible tool-based memory service
"""

from .hotmem_service import HotMemService

__all__ = ['HotMemService']