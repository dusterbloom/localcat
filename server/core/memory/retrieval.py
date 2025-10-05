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
from loguru import logger

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
        logger.debug(f"[Retrieval] enabled_sources={enabled_sources} query='{query[:50]}'")

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
                logger.debug(f"[Retrieval] graph_bullets count={len(graph_bullets)}")
                # Score graph bullets based on position (earlier = higher priority)
                for idx, bullet in enumerate(graph_bullets):
                    priority_boost = 1.0 if source_priority[0] == "graph" else 0.5
                    score = priority_boost * (100 - idx * 10)
                    all_candidates.append((score, bullet, "graph"))

            elif source == "convo" and budget.get("convo", 0) > 0:
                convo_bullets = self._convo_retrieve(query, budget["convo"], seen.copy())
                logger.debug(f"[Retrieval] convo_bullets count={len(convo_bullets)}")
                for idx, bullet in enumerate(convo_bullets):
                    # Convo/FTS matches get a boost for relevance (they matched the search query)
                    priority_boost = 1.2 if source_priority[0] == "convo" else 1.1
                    score = priority_boost * (100 - idx * 10)
                    all_candidates.append((score, bullet, "convo"))

            elif source == "summary" and budget.get("summary", 0) > 0:
                summary_bullets = self._summary_retrieve(budget["summary"], seen.copy())
                logger.debug(f"[Retrieval] summary_bullets count={len(summary_bullets)}")
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

        # Log source distribution in final results
        source_counts = {}
        for bullet in final_bullets:
            for src in ["graph", "convo", "summary"]:
                if f"[{src}]" in bullet:
                    source_counts[src] = source_counts.get(src, 0) + 1
        logger.debug(f"[Retrieval] final_bullets={len(final_bullets)} source_counts={source_counts}")

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
        # Identity scope
        current_user = getattr(self.host, 'current_user_id', None)
        current_session = getattr(self.host, 'current_session_id', None)
        edge_scope_cache: Dict[str, bool] = {}
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
                    # Provenance scope: keep only edges that belong to current user (or current session)
                    edge_id = self.host.store.edge_id(s, r, d)
                    allowed_edge = edge_scope_cache.get(edge_id)
                    if allowed_edge is None:
                        allowed_edge = False
                        try:
                            prov = self.host.store.get_edge_provenance(edge_id)  # List[(text, session_id, turn_id, ts)]
                        except Exception:
                            prov = []
                        # If we know the current user, require any provenance session to belong to them
                        if current_user:
                            for (_text, sess_id, _turn, _ts) in prov:
                                if self.host.store.is_session_owned_by_user(sess_id, current_user):
                                    allowed_edge = True
                                    break
                        # Fallback: if no user scope available, allow edges from current session only
                        elif current_session:
                            allowed_edge = any(sess_id == current_session for (_text, sess_id, _turn, _ts) in prov)
                        edge_scope_cache[edge_id] = allowed_edge
                    if not allowed_edge:
                        continue

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
            # Sanitize query for FTS5: remove punctuation and special chars
            # FTS5 syntax errors on: , . ? ! ( ) " ' - and other special chars
            import re
            sanitized = re.sub(r'[^\w\s]', ' ', query)  # Keep only alphanumeric and spaces
            sanitized = ' '.join(sanitized.split())  # Normalize whitespace

            if not sanitized.strip():
                logger.debug(f"[Retrieval._convo] Query sanitized to empty string")
                return []

            # Prefer user/session-scoped FTS to prevent cross-user leakage
            user_id = getattr(self.host, 'current_user_id', None)
            session_id = getattr(self.host, 'current_session_id', None)
            allowed = [e for e in [user_id, session_id] if e]
            if allowed and hasattr(self.host.store, 'search_fts_scoped'):
                hits = self.host.store.search_fts_scoped(sanitized, allowed, limit=max_bullets * 2)
            else:
                # Fallback to global FTS
                hits = self.host.store.search_fts(sanitized, limit=max_bullets * 2)
            logger.debug(f"[Retrieval._convo] FTS returned {len(hits)} hits for query='{sanitized[:30]}'")
        except Exception as e:
            logger.warning(f"[Retrieval._convo] FTS search failed: {e}")
            hits = []
        # Exclude enrollment/fixed phrases from retrieval context
        excluded = []
        try:
            import os
            ex_raw = os.getenv("EXCLUDED_MEMORY_PHRASES", "").strip()
            fixed = os.getenv("ENROLLMENT_FIXED_PHRASE", "").strip()
            excluded = [p.strip().lower() for p in ex_raw.split("||") if p.strip()]
            if fixed:
                excluded.append(fixed.lower())
        except Exception:
            pass
        for text, eid, ts in hits:
            logger.debug(f"[Retrieval._convo] Processing hit: eid='{eid}' text='{text[:40]}'")
            # Filter to only conversation entries (not summary)
            # Summaries are stored with eid starting with "summary:" or "summary"
            if eid and (eid == "summary" or eid.startswith("summary:")):
                logger.debug(f"[Retrieval._convo] Skipping summary: {eid}")
                continue
            s = text.strip().replace("\n", " ")
            if not s:
                logger.debug(f"[Retrieval._convo] Skipping empty text")
                continue
            if excluded:
                tl = s.lower()
                if any(p in tl for p in excluded):
                    logger.debug("[Retrieval._convo] Skipping excluded phrase hit")
                    continue
            bullet = f"• [convo] {s[:120]}{self._ago_suffix(ts)}"  # keep short
            if bullet in seen:
                logger.debug(f"[Retrieval._convo] Skipping duplicate bullet")
                continue
            seen.add(bullet)
            out.append(bullet)
            logger.debug(f"[Retrieval._convo] Added bullet: {bullet[:60]}")
            if len(out) >= max_bullets:
                break
        logger.debug(f"[Retrieval._convo] Returning {len(out)} bullets")
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
        # Role-aware display mapping
        def _display(x: str) -> str:
            try:
                if x.startswith('you:'):
                    return x.split(':', 1)[1]
                if x.startswith('agent:'):
                    return x.split(':', 1)[1]
            except Exception:
                pass
            return x

        ds = _display(s)
        dd = _display(d)

        meaningless_entities = {"it", "this", "that", "there", "here", "been", "know"}
        wh_words = {"what", "who", "when", "where", "why", "how", "which"}
        if ds.lower() in meaningless_entities or dd.lower() in meaningless_entities:
            return ""
        # Drop obvious punctuation artifacts
        if "," in ds or "," in dd:
            return ""
        if ds.lower() in wh_words or dd.lower() in wh_words:
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
            return f"{ds} is named {dd}"
        if r == "has":
            return f"{ds} has {dd}"
        if r == "also_known_as":
            # Only meaningful for user identity
            if ds.lower() not in ("you", _display(getattr(self.host, 'user_eid', 'you'))):
                return ""
            return f"{ds} aka {dd}"
        if r == "is":
            if ds.lower() in meaningless_entities or dd.lower().startswith("what "):
                return ""
            return f"{ds} is {dd}"
        if r.startswith("v:"):
            return f"{ds} {r[2:]} {dd}"
        # Common relation fixes (remove underscore)
        if r == "lives_in":
            return f"{ds} lives in {dd}"
        if r == "works_at":
            return f"{ds} works at {dd}"
        if r == "works_in":
            return f"{ds} works in {dd}"

        return f"{ds} {r.replace('_', ' ')} {dd}"

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
