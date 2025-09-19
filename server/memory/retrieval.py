"""
Retrieval module (Phase 1D)

Implements the existing retrieval policy:
- Entity-first selection with relation priority and recency
- Fallback to recent facts
- Returns up to 3 bullets

This adapter is behavior-preserving: it reads from the host's indices and
store exactly as the previous implementation did.
"""

from typing import List, Tuple, Any
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

        for entity in query_entities:
            if entity in self.host.entity_index:
                candidates = list(self.host.entity_index[entity])
                scored: List[Tuple[int, int, str, str, str]] = []
                for s, r, d in candidates:
                    ts = 0
                    try:
                        neigh = self.host.store.neighbors(s, r)
                        for (dst, _w, nts, _p, _n, _st) in neigh:
                            if dst == d:
                                ts = int(nts)
                                break
                    except Exception:
                        ts = 0
                    pri = pred_pri.get(r, 50)
                    scored.append((pri, ts, s, r, d))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for _pri, _ts, s, r, d in scored:
                    fact = f"{s} {r} {d}"
                    if fact not in seen:
                        suffix = self._ago_suffix(_ts)
                        out.append(f"• {fact}{suffix}")
                        seen.add(fact)
                        if len(out) >= max_bullets:
                            return out

        # Fallback to recent facts if we still need context
        for item in reversed(list(self.host.recency_buffer)[-10:]):
            fact = f"{item.s} {item.r} {item.d}"
            if fact not in seen:
                age = self._ago_suffix(item.timestamp if hasattr(item, 'timestamp') else 0)
                if item.r == "name":
                    formatted = f"• {item.s}'s name is {item.d}"
                elif item.r == "has":
                    formatted = f"• {item.s} has {item.d}"
                elif item.r == "is":
                    formatted = f"• {item.s} is {item.d}"
                elif item.r.startswith("v:"):
                    formatted = f"• {item.s} {item.r[2:]} {item.d}"
                else:
                    formatted = f"• {item.s} {item.r.replace('_', ' ')} {item.d}"

                out.append(formatted + age)
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
            bullet = f"• recently: {s[:120]}{self._ago_suffix(ts)}"  # keep short
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
            bullet = f"• summary: {s[:160]}{self._ago_suffix(ts)}"
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
