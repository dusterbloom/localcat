"""
RelationStats: auto-detection of single-valued relations from SQLite.

Computes, caches, and exposes whether a relation should be treated as
single-valued (i.e., at most one active dst per src/user) using observed data.

Zero-maintenance: recomputed periodically; no hardcoded lists.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional
from loguru import logger


class RelationStats:
    """Data-driven relation statistics accessor with TTL cache."""

    def __init__(self, store):
        self.store = store
        self._cache: Dict[str, bool] = {}
        self._last_build: float = 0.0
        self._ttl: float = float(os.getenv("MEMORY_RELATION_STATS_TTL", "900"))  # seconds
        # Threshold: fraction of sources (users) that have >1 active dst for a rel
        # If below threshold → consider single-valued
        try:
            self._exclusivity_threshold = float(os.getenv("MEMORY_SINGLE_VALUED_RATIO", "0.2"))
        except Exception:
            self._exclusivity_threshold = 0.2

    def _needs_refresh(self) -> bool:
        return (time.time() - self._last_build) > self._ttl or not self._cache

    def _rebuild(self) -> None:
        """Recompute exclusivity from SQLite edge table.

        SQL logic: for each rel, compute the fraction of src with count(dst)>1.
        """
        try:
            cur = self.store.sql.cursor()
            # For performance, limit to active edges
            rows = cur.execute(
                """
                SELECT rel, src, COUNT(DISTINCT dst) AS c
                FROM edge
                WHERE status >= 0
                GROUP BY rel, src
                """
            ).fetchall()
        except Exception as e:
            logger.debug(f"[RelationStats] rebuild failed: {e}")
            return

        per_rel_total: Dict[str, int] = {}
        per_rel_multi: Dict[str, int] = {}

        for rel, src, c in rows:
            per_rel_total[rel] = per_rel_total.get(rel, 0) + 1
            if int(c or 0) > 1:
                per_rel_multi[rel] = per_rel_multi.get(rel, 0) + 1

        new_cache: Dict[str, bool] = {}
        for rel, total in per_rel_total.items():
            multi = per_rel_multi.get(rel, 0)
            ratio = (multi / total) if total > 0 else 1.0
            new_cache[rel] = (ratio <= self._exclusivity_threshold)

        self._cache = new_cache
        self._last_build = time.time()
        logger.debug(f"[RelationStats] rebuilt {len(self._cache)} rels; single-valued: {sum(1 for v in self._cache.values() if v)}")

    def is_single_valued(self, rel: Optional[str]) -> bool:
        if not rel:
            return False
        if self._needs_refresh():
            self._rebuild()
        return bool(self._cache.get(rel, False))

