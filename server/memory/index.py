"""
In-memory index scaffolding (Phase 1 step 1A)

Provides a minimal helper to rebuild entity indices from the store without
changing existing behavior. This is a future seam for moving indexing out of
HotMemory.
"""

from collections import defaultdict
from typing import Dict, Set, Tuple

from .store import MemoryStore


class HotIndex:
    def __init__(self):
        # entity -> set of (s, r, d)
        self.entity_index: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)

    def rebuild_from_store(self, store: MemoryStore) -> int:
        count = 0
        edges = store.get_all_edges()
        for s, r, d, conf in edges:
            if conf > 0.1:
                self.entity_index[s].add((s, r, d))
                self.entity_index[d].add((s, r, d))
                count += 1
        return count

