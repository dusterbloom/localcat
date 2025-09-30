"""
Retrieval module (Phase 1D)

Implements the existing retrieval policy:
- Entity-first selection with relation priority and recency
- Fallback to recent facts
- Returns up to 3 bullets

This adapter is behavior-preserving: it reads from the host's indices and
store exactly as the previous implementation did.
"""

from typing import List, Tuple, Any, Dict
import time


import os


class Retrieval:
    def __init__(self, host: Any):
        """host must expose: entity_index, recency_buffer, store."""
        self.host = host

    def retrieve(self, query: str, entities: List[str], turn_id: int, max_bullets: int = 3) -> List[str]:
        bullets: List[str] = []
        seen = set()

        # Source order control via env (defaults to graph only)
        sources = [s.strip() for s in os.getenv("MEMORY_SOURCES", "graph").split(",") if s.strip()]

        # 1) Graph retrieval
        if "graph" in sources and len(bullets) < max_bullets:
            bullets.extend(self._graph_retrieve(query, entities, turn_id, max_bullets - len(bullets), seen))

        # 2) Summary retrieval
        if "summary" in sources and len(bullets) < max_bullets:
            bullets.extend(self._summary_retrieve(max_bullets - len(bullets), seen))

        # 3) Conversation retrieval via FTS (if indexed)
        if "convo" in sources and len(bullets) < max_bullets:
            bullets.extend(self._convo_retrieve(query, max_bullets - len(bullets), seen))

        return bullets[:max_bullets]

    def _graph_retrieve(self, query: str, entities: List[str], turn_id: int, max_bullets: int, seen: set) -> List[str]:
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
                    # Per‑relation support requirements
                    min_pos = REL_MIN_POS.get(r, 1)
                    if pos < min_pos:
                        # Additional guard for families of noisy relations
                        if ("happen" in r or "feel" in r) and pos < 2:
                            continue
                        if r in ("quality", "quantity"):
                            continue

                    pri = pred_pri.get(r, 50)
                    # Composite score: priority × weight × support, tie-break by recency
                    support = 1.0 + min(max(pos, 0), 5) * 0.1  # dampen large pos
                    score = float(pri) * float(max(w, 0.01)) * support
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

        # Fallback to recent facts if we still need context
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
        meaningless_entities = {"it", "this", "that", "there", "here", "been"}
        wh_words = {"what", "who", "when", "where", "why", "how", "which"}
        if s.lower() in meaningless_entities or d.lower() in meaningless_entities:
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
