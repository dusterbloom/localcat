"""
Retrieval module — simplified.

Two sources: FTS conversation search + graph entity lookup.
Simple recency-weighted scoring, greeting gate, dedup.
"""

from typing import List, Tuple, Any, Dict, Optional, Set
import time
import math
import os
from loguru import logger
from .memory_constants import WEIGHT_MIN_ACTIVE, RECENCY_HALF_LIFE_MS


class Retrieval:
    def __init__(self, host: Any):
        """host must expose: entity_index, recency_buffer, store."""
        self.host = host

    def retrieve(self, query: str, entities: List[str], turn_id: int,
                 max_bullets: int = 3, intent: Optional[Dict] = None) -> List[str]:
        """
        Retrieve memory bullets using FTS + graph lookup.

        Returns:
            List of formatted bullet strings (max max_bullets).
        """
        q = (query or "").strip().lower()
        if not q:
            return []

        # Gate: suppress memory for greetings/smalltalk
        greeting_terms = ("hello", "hi", "hey", "good morning", "good afternoon",
                          "good evening", "howdy", "greetings", "yo")
        smalltalk_terms = ("how are you", "how's it going", "what's up",
                           "how do you do", "how are you doing")

        is_greeting = any(term in q for term in greeting_terms) and len(q.split()) <= 4
        is_smalltalk = any(term in q for term in smalltalk_terms)

        if is_greeting or is_smalltalk:
            name_indicators = ("name", "who are you", "what's your name",
                               "what is your name", "called", "identity")
            if not any(ind in q for ind in name_indicators):
                logger.info(f"[Retrieval] Greeting/smalltalk — no memory needed: '{q}'")
                return []

        now_ms = int(time.time() * 1000)
        seen: Set[str] = set()
        scored: List[Tuple[float, str]] = []

        # Source 1: FTS conversation search
        try:
            fts_results = self.host.store.search_fts(query, limit=10)
            for text, _eid, ts in fts_results:
                key = text.strip().lower()
                if key in seen or len(key) < 10:
                    continue
                seen.add(key)
                score = self._recency_score(ts, now_ms)
                scored.append((score, text))
        except Exception as e:
            logger.warning(f"[Retrieval] FTS search failed: {e}")

        # Source 2: Graph entity lookup
        try:
            for entity in (entities or [])[:5]:
                if entity not in self.host.entity_index:
                    continue
                for s, r, d in list(self.host.entity_index[entity])[:10]:
                    bullet = self._humanize_fact(s, r, d)
                    if not bullet:
                        continue
                    key = bullet.strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)

                    # Get edge weight for scoring
                    try:
                        edge_id = self.host.store.edge_id(s, r, d)
                        cur = self.host.store.sql.cursor()
                        row = cur.execute(
                            "SELECT weight, updated_at FROM edge WHERE id = ?",
                            (edge_id,)
                        ).fetchone()
                        weight = float(row[0]) if row else 0.3
                        ts = int(row[1]) if row else now_ms
                    except Exception:
                        weight, ts = 0.3, now_ms

                    if weight < WEIGHT_MIN_ACTIVE:
                        continue

                    score = weight * self._recency_score(ts, now_ms)
                    scored.append((score, bullet))
        except Exception as e:
            logger.warning(f"[Retrieval] Graph lookup failed: {e}")

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        bullets = [text for _, text in scored[:max_bullets]]

        logger.info(f"[Retrieval] Returning {len(bullets)} bullets from {len(scored)} candidates")
        return bullets

    def _recency_score(self, ts: int, now_ms: int) -> float:
        """Exponential decay based on RECENCY_HALF_LIFE_MS."""
        age_ms = max(0, now_ms - ts)
        if RECENCY_HALF_LIFE_MS <= 0:
            return 1.0
        return math.exp(-0.693 * age_ms / RECENCY_HALF_LIFE_MS)

    def _humanize_fact(self, s: str, r: str, d: str) -> str:
        """Convert (s, r, d) triple to English fragment."""
        if not s or not r or not d:
            return ""

        # Skip noisy relations
        skip_rels = {"said", "asked", "told", "mentioned", "wants_to", "going_to",
                     "has_intent", "command", "request"}
        if r.lower() in skip_rels:
            return ""

        # Format: "subject relation destination"
        r_display = r.replace("_", " ")

        # Fix agreement for "you"
        if s.lower() == "you":
            if r_display in ("is", "has", "does", "was"):
                verb_map = {"is": "are", "has": "have", "does": "do", "was": "were"}
                r_display = verb_map.get(r_display, r_display)

        return f"{s} {r_display} {d}"
