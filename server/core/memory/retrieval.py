"""
Retrieval module (Phase 1D)

Implements the existing retrieval policy:
- Entity-first selection with relation priority and recency
- Fallback to recent facts
- Returns up to 3 bullets

This adapter is behavior-preserving: it reads from the host's indices and
store exactly as the previous implementation did.
"""

from typing import List, Tuple, Any, Dict, Optional, Set
import time


import os


class Retrieval:
    def __init__(self, host: Any):
        """host must expose: entity_index, recency_buffer, store."""
        self.host = host

    def retrieve(self, query: str, entities: List[str], turn_id: int, max_bullets: int = 3, intent: Optional[Dict] = None) -> List[str]:
        """
        Retrieve memory bullets using hybrid strategy with fair budget allocation.

        Args:
            query: User query text
            entities: Extracted entities from query
            turn_id: Current conversation turn ID
            max_bullets: Maximum bullets to return
            intent: Optional intent classification result for routing

        Returns:
            List of formatted bullet strings
        """
        # Source control via env (defaults to graph only for backward compatibility)
        enabled_sources = [s.strip() for s in os.getenv("MEMORY_SOURCES", "graph").split(",") if s.strip()]

        # Lightweight intent gating for greetings
        q = (query or "").strip().lower()
        greeting_terms = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening", "top of the morning", "howdy")
        is_greeting = any(term in q for term in greeting_terms) and len(q.split()) <= 4 if q else False
        relation_allowlist: Optional[Set[str]] = {"name"} if is_greeting else None

        # Determine source priority based on query characteristics and intent
        source_priority = self._get_source_priority(query, intent)

        # Budget allocation strategy: Give each source a fair chance
        # This prevents graph from starving convo/summary
        budget = self._allocate_budget(max_bullets, enabled_sources)

        # Collect candidates from all enabled sources concurrently
        all_candidates: List[Tuple[float, str, str]] = []  # (score, bullet, source)
        seen = set()

        for source in enabled_sources:
            if source == "graph" and budget.get("graph", 0) > 0:
                graph_bullets = self._graph_retrieve(
                    query, entities, turn_id, budget["graph"], seen.copy(), relation_allowlist
                )
                # Score graph bullets based on position (earlier = higher priority)
                for idx, bullet in enumerate(graph_bullets):
                    priority_boost = 1.0 if source_priority[0] == "graph" else 0.5
                    score = priority_boost * (100 - idx * 10)
                    all_candidates.append((score, bullet, "graph"))

            elif source == "convo" and budget.get("convo", 0) > 0:
                convo_bullets = self._convo_retrieve(query, budget["convo"], seen.copy())
                for idx, bullet in enumerate(convo_bullets):
                    # Convo/FTS matches get a boost for relevance (they matched the search query)
                    priority_boost = 1.2 if source_priority[0] == "convo" else 1.1
                    score = priority_boost * (100 - idx * 10)
                    all_candidates.append((score, bullet, "convo"))

            elif source == "summary" and budget.get("summary", 0) > 0:
                summary_bullets = self._summary_retrieve(budget["summary"], seen.copy())
                for idx, bullet in enumerate(summary_bullets):
                    # Summary gets moderate boost (contextual but not query-matched)
                    priority_boost = 1.05 if source_priority[0] == "summary" else 1.0
                    score = priority_boost * (100 - idx * 10)
                    all_candidates.append((score, bullet, "summary"))

        # Re-rank all candidates by score and deduplicate
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        final_bullets = []
        seen_bullets = set()
        for score, bullet, source in all_candidates:
            if bullet not in seen_bullets:
                final_bullets.append(bullet)
                seen_bullets.add(bullet)
                if len(final_bullets) >= max_bullets:
                    break

        return final_bullets[:max_bullets]

    def _allocate_budget(self, max_bullets: int, enabled_sources: List[str]) -> Dict[str, int]:
        """
        Allocate retrieval budget across sources to prevent starvation.

        Strategy:
        - Get MORE candidates from each source than max_bullets
        - Let re-ranking decide final selection
        - This ensures diversity without starving any source
        """
        budget = {}

        if max_bullets <= 0:
            return budget

        # Count active sources
        active_sources = [s for s in ["graph", "convo", "summary"] if s in enabled_sources]

        if not active_sources:
            return budget

        # Give each source a generous budget to ensure candidates
        # We'll let re-ranking pick the best max_bullets from all sources
        # This is key: don't limit sources too early!
        per_source_budget = max(max_bullets, 3)  # At least 3 per source, or max_bullets if higher

        for source in active_sources:
            budget[source] = per_source_budget

        return budget

    def _graph_retrieve(self, query: str, entities: List[str], turn_id: int, max_bullets: int, seen: set, allowed_relations: Optional[Set[str]] = None) -> List[str]:
        out: List[str] = []
        # Prefer fact bullets based on query entities
        ent_set = [e for e in entities if e]
        non_you = [e for e in ent_set if e != "you"]
        include_you = any(e == "you" for e in ent_set)
        query_entities = non_you[:4]
        if include_you:
            query_entities.append("you")

        pred_pri = {
            "lives_in": 100,
            "works_at": 95,
            "born_in": 90,
            "moved_from": 85,
            "participated_in": 80,
            "friend_of": 78,
            "name": 75,
            "has": 60,
        }

        WEIGHT_MIN = 0.25  # Align with status thresholding used by store
        # Stricter minimum positive support for noisy relations
        REL_MIN_POS: Dict[str, int] = {
            "also_known_as": 2,
        }

        now_ms = int(time.time() * 1000)
        for entity in query_entities:
            if entity in self.host.entity_index:
                candidates = list(self.host.entity_index[entity])
                scored: List[Tuple[float, int, str, str, str]] = []

                # Build quick lookup for (s,r)-> dst meta once per relation
                meta_cache: Dict[Tuple[str, str], Dict[str, Tuple[float, int, int, int, int]]] = {}

                for s, r, d in candidates:
                    # Retrieve neighbor meta for this (s,r) only once
                    key = (s, r)
                    if key not in meta_cache:
                        try:
                            neigh = self.host.store.neighbors(s, r)
                            meta_cache[key] = {
                                dst: (float(w), int(nts), int(pos), int(neg), int(st))
                                for (dst, w, nts, pos, neg, st) in neigh
                            }
                        except Exception:
                            meta_cache[key] = {}

                    w, ts, pos, neg, status = 1.0, 0, 0, 0, 1
                    meta = meta_cache.get(key, {}).get(d)
                    if meta is not None:
                        w, ts, pos, neg, status = meta

                    # Skip edges that are stale/negated/weak
                    if status <= 0:
                        continue
                    if pos <= neg:
                        continue
                    if w < WEIGHT_MIN:
                        continue
                    # Optional relation allowlist (e.g., for greetings)
                    if allowed_relations is not None and r not in allowed_relations:
                        continue
                    # Disallow AKA unless it's explicitly about the user
                    if r == "also_known_as" and s.lower() != "you":
                        continue
                    # Per‑relation support requirements
                    min_pos = REL_MIN_POS.get(r, 1)
                    if pos < min_pos:
                        # Additional guard for families of noisy relations
                        if ("happen" in r or "feel" in r) and pos < 2:
                            continue
                        if r in ("quality", "quantity"):
                            continue

                    pri = pred_pri.get(r, 50)
                    # Composite score: priority × weight × support × recency
                    support = 1.0 + min(max(pos, 0), 5) * 0.1  # dampen large pos
                    age_ms = max(0, now_ms - int(ts)) if ts else 0
                    half_life_ms = 7 * 24 * 3600 * 1000  # 7 days
                    recency_factor = (2 ** (-(age_ms / half_life_ms))) if ts else 0.8
                    score = float(pri) * float(max(w, 0.01)) * support * recency_factor
                    scored.append((score, ts, s, r, d))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for _score, _ts, s, r, d in scored:
                    fact = f"{s} {r} {d}"
                    if fact not in seen:
                        suffix = self._ago_suffix(_ts)
                    human = self._humanize_fact(s, r, d)
                    if human:
                        out.append(f"• [graph] {human}{suffix}")
                        seen.add(fact)
                        if len(out) >= max_bullets:
                            return out

        # Fallback to recency unless an allowlist is active
        if allowed_relations is None:
            for item in reversed(list(self.host.recency_buffer)[-10:]):
                fact = f"{item.s} {item.r} {item.d}"
                if fact not in seen:
                    age = self._ago_suffix(item.timestamp if hasattr(item, 'timestamp') else 0)
                    human = self._humanize_fact(item.s, item.r, item.d)
                    if human:
                        out.append(f"• [graph] {human}{age}")
                        seen.add(fact)
                        if len(out) >= max_bullets:
                            break

        return out[:max_bullets]

    def _convo_retrieve(self, query: str, max_bullets: int, seen: set) -> List[str]:
        out: List[str] = []
        try:
            # Simple FTS search over prior mentions; limit small to keep latency predictable
            hits = self.host.store.search_fts(query, limit=max_bullets * 2)
        except Exception:
            hits = []
        for text, eid, ts in hits:
            # Filter to only conversation entries (not summary)
            if eid == "summary":
                continue
            s = text.strip().replace("\n", " ")
            if not s:
                continue
            bullet = f"• [convo] {s[:120]}{self._ago_suffix(ts)}"  # keep short
            if bullet in seen:
                continue
            seen.add(bullet)
            out.append(bullet)
            if len(out) >= max_bullets:
                break
        return out

    def _summary_retrieve(self, max_bullets: int, seen: set) -> List[str]:
        out: List[str] = []
        try:
            rows = self.host.store.get_recent_chunks_by_eid("summary", limit=max_bullets * 2)
        except Exception:
            rows = []
        for text, ts in rows:
            s = text.strip().replace("\n", " ")
            if not s:
                continue
            bullet = f"• [summary] {s[:160]}{self._ago_suffix(ts)}"
            if bullet in seen:
                continue
            seen.add(bullet)
            out.append(bullet)
            if len(out) >= max_bullets:
                break
        return out

    def _ago_suffix(self, ts_ms: int) -> str:
        try:
            if not ts_ms:
                return ""
            now_ms = int(time.time() * 1000)
            delta = max(0, now_ms - int(ts_ms))
            sec = delta // 1000
            if sec < 60:
                return f" ({sec}s ago)"
            mins = sec // 60
            if mins < 60:
                return f" ({mins}m ago)"
            hours = mins // 60
            if hours < 24:
                return f" ({hours}h ago)"
            days = hours // 24
            # compact format: include days and remaining hours
            rem_h = hours % 24
            if rem_h > 0:
                return f" ({days}d {rem_h}h ago)"
            return f" ({days}d ago)"
        except Exception:
            return ""

    def _humanize_fact(self, s: str, r: str, d: str) -> str:
        """Convert (s,r,d) to a compact English fragment.

        Applies conservative filtering for conversational/command relations and
        fixes common agreement issues for second-person subjects.
        """
        meaningless_entities = {"it", "this", "that", "there", "here", "been", "know"}
        wh_words = {"what", "who", "when", "where", "why", "how", "which"}
        if s.lower() in meaningless_entities or d.lower() in meaningless_entities:
            return ""
        # Drop obvious punctuation artifacts
        if "," in s or "," in d:
            return ""
        if s.lower() in wh_words or d.lower() in wh_words:
            return ""

        stop_relations = {
            "and",
            "know",
            "remember",
            "say",
            "tell",
            "think",
            "ask",
            "quality",
            "quantity",
            "tell_about",
            "talk",
            "talk_about",
            "delete",
            "remove",
            # keep also_known_as with support gating; suppress generic variants
            "known",
            "known_as",
        }
        if r in stop_relations:
            return ""

        if r == "name":
            if s.lower() == "you":
                return f"your name is {d}"
            return f"{s}'s name is {d}"
        if r == "has":
            if s.lower() == "you":
                return f"you have {d}"
            return f"{s} has {d}"
        if r == "also_known_as":
            # Only meaningful for user identity
            if s.lower() != "you":
                return ""
            return f"{s} aka {d}"
        if r == "is":
            if s.lower() in meaningless_entities or d.lower().startswith("what "):
                return ""
            if s.lower() == "you":
                return f"you are {d}"
            return f"{s} is {d}"
        if r.startswith("v:"):
            return f"{s} {r[2:]} {d}"
        # Common relation fixes for second person
        if s.lower() == "you":
            if r == "lives_in":
                return f"you live in {d}"
            if r == "works_at":
                return f"you work at {d}"
            if r == "works_in":
                return f"you work in {d}"

        return f"{s} {r.replace('_', ' ')} {d}"

    def _is_temporal_query(self, query: str) -> bool:
        """Detect if query is asking about time-based information."""
        q = query.lower()
        temporal_keywords = {
            "yesterday", "today", "recently", "last week", "last month",
            "earlier", "before", "ago", "just now", "this morning",
            "this afternoon", "this evening", "last night", "past"
        }
        return any(kw in q for kw in temporal_keywords)

    def _is_semantic_query(self, query: str) -> bool:
        """Detect if query is asking about topics/concepts rather than facts."""
        q = query.lower()
        semantic_indicators = {
            "about", "related to", "regarding", "concerning",
            "hobbies", "interests", "preferences", "likes", "favorites",
            "thoughts on", "opinion", "views", "feelings"
        }
        return any(ind in q for ind in semantic_indicators)

    def _get_source_priority(self, query: str, intent: Optional[Dict] = None) -> List[str]:
        """
        Determine source priority order based on query characteristics and intent.

        Returns list of sources in priority order: ['graph', 'convo', 'summary']
        """
        # Intent-based routing (highest priority)
        if intent and not intent.get('fallback', False):
            intent_type = intent.get('intent', '')

            # Memory lookup intents should prioritize conversation history
            if intent_type in ['lookup_memory', 'recall_information', 'retrieve_information']:
                return ['convo', 'summary', 'graph']

            # Update/delete intents should prioritize graph for accuracy
            if intent_type in ['memory_update', 'delete_memory', 'store_information']:
                return ['graph', 'convo', 'summary']

        # Query pattern-based routing (medium priority)
        if self._is_temporal_query(query):
            # Temporal queries benefit most from conversation history
            return ['convo', 'summary', 'graph']

        if self._is_semantic_query(query):
            # Semantic queries benefit from summaries and conversation
            return ['summary', 'convo', 'graph']

        # Default: factual queries work best with graph-first
        return ['graph', 'convo', 'summary']
