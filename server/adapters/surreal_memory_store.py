"""
SurrealMemoryStore
------------------
Thin SurrealDB-backed adapter implementing the subset of the MemoryStore
API used by HotMemoryFacade and MemoryRetrieverOptimized.

Notes
- Designed to be optional: import and connect lazily.
- Batches writes via enqueue_* buffers and persists on flush().
- Uses Surreal TYPE RELATION for edges and a SEARCH index for mentions.

Env variables
- SURREALDB_URL (e.g., ws://127.0.0.1:8000/rpc)
- SURREALDB_USER, SURREALDB_PASS
- SURREALDB_NAMESPACE, SURREALDB_DATABASE
"""

from __future__ import annotations

import os
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple


class _SurrealClient:
    def __init__(self, url: str, user: str, password: str, ns: str, db: str):
        self._url = url
        self._user = user
        self._password = password
        self._ns = ns
        self._db = db
        self._client = None
        self._connected = False

    async def connect(self):
        if self._connected:
            return
        try:
            from surrealdb import Surreal  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "surrealdb client not installed. `pip install surrealdb`"
            ) from e
        self._client = Surreal(self._url)
        await self._client.connect()
        await self._client.signin({"user": self._user, "pass": self._password})
        await self._client.use(self._ns, self._db)
        self._connected = True

    async def query(self, q: str, vars: Optional[Dict[str, Any]] = None):
        await self.connect()
        return await self._client.query(q, vars or {})


class SurrealMemoryStore:
    """SurrealDB-backed store with a compatible surface for HotMem."""

    def __init__(self, url: str, user: str, password: str, ns: str, db: str):
        self._db = _SurrealClient(url, user, password, ns, db)
        # Write buffers
        self._aliases: List[Tuple[str, str]] = []
        self._edge_events: List[Tuple[str, str, str, float, int, Optional[str], Dict[str, Any]]] = []
        self._edge_meta: List[Tuple[str, str, str, str, Optional[str], Optional[str], Dict[str, Any], int]] = []
        self._mentions: List[Tuple[str, str, int, str, int]] = []

    # ---------- Helpers ----------
    @staticmethod
    def edge_id(s: str, r: str, d: str) -> str:
        return hashlib.sha1(f"{s}|{r}|{d}".encode()).hexdigest()

    # ---------- Aliases ----------
    def enqueue_alias(self, alias: str, eid: str) -> None:
        self._aliases.append((alias, eid))

    async def _flush_aliases(self):
        if not self._aliases:
            return
        q = (
            "INSERT INTO alias { alias: $alias, canonical: $canonical } ON DUPLICATE KEY UPDATE canonical = $canonical;"
        )
        for alias, eid in self._aliases:
            await self._db.query(q, {"alias": alias, "canonical": eid})
        self._aliases.clear()

    async def resolve_alias_async(self, alias: str) -> Optional[str]:
        res = await self._db.query(
            "SELECT canonical FROM alias WHERE alias = $a LIMIT 1;", {"a": alias}
        )
        try:
            rows = res[0]["result"]
            return rows[0].get("canonical") if rows else None
        except Exception:
            return None

    def resolve_alias(self, alias: str) -> Optional[str]:
        # Synchronous convenience (best-effort). Not used on hot path.
        # Callers that need certainty should use async path.
        return None

    # ---------- Edges ----------
    def observe_edge(self, s: str, r: str, d: str, conf: float, now_ts: int) -> None:
        self._edge_events.append((s, r, d, float(conf), int(now_ts), None, {}))

    def enqueue_edge_meta(
        self,
        s: str,
        r: str,
        d: str,
        prov: str = "",
        lang: Optional[str] = None,
        span: Optional[str] = None,
        props: Optional[Dict[str, Any]] = None,
        ts: Optional[int] = None,
    ) -> None:
        self._edge_meta.append((s, r, d, prov or "", lang, span, props or {}, int(ts or time.time())))

    async def _flush_edges(self):
        if not self._edge_events and not self._edge_meta:
            return
        # Upsert entities and relate edges; simple baseline
        for s, r, d, conf, ts, sid, props in self._edge_events:
            await self._db.query(
                "CREATE entity:$_s SET name=$sname ON DUPLICATE;",
                {"_s": s, "sname": s},
            )
            await self._db.query(
                "CREATE entity:$_d SET name=$dname ON DUPLICATE;",
                {"_d": d, "dname": d},
            )
            await self._db.query(
                (
                    "RELATE entity:$_s->edge->entity:$_d SET predicate=$p, confidence=$c, ts=time::unix($ts), session_id=$sid;"
                ),
                {"_s": s, "_d": d, "p": r, "c": conf, "ts": int(ts / 1000), "sid": sid},
            )
            if props:
                # Store props as an upsert on the relation record (requires a selection by id or filter)
                await self._db.query(
                    (
                        "UPDATE edge SET props = merge(if props != NONE then props else {}, $props) WHERE in = entity:$_s AND out = entity:$_d AND predicate = $p;"
                    ),
                    {"_s": s, "_d": d, "p": r, "props": props},
                )
        self._edge_events.clear()

        for s, r, d, prov, lang, span, props, ts in self._edge_meta:
            await self._db.query(
                (
                    "UPDATE edge SET props = merge(if props != NONE then props else {}, $props) WHERE in = entity:$_s AND out = entity:$_d AND predicate = $p;"
                ),
                {"_s": s, "_d": d, "p": r, "props": {"prov": prov, "lang": lang, "span": span, **(props or {})}},
            )
        self._edge_meta.clear()

    # ---------- Mentions / FTS ----------
    def enqueue_mention(self, eid: str, text: str, ts: float, sid: str, tid: int) -> None:
        self._mentions.append((eid, text[:500], int(ts), sid, int(tid)))

    async def _flush_mentions(self):
        if not self._mentions:
            return
        q = (
            "INSERT INTO mention { eid: $eid, text: $text, ts: time::unix($ts), session_id: $sid, turn_id: $tid };"
        )
        for eid, text, ts, sid, tid in self._mentions:
            await self._db.query(q, {"eid": eid, "text": text, "ts": int(ts / 1000), "sid": sid, "tid": tid})
        self._mentions.clear()

    def search_fts_detailed(self, query: str, limit: int = 10) -> List[Tuple[str, str, int]]:
        # Synchronous proxy for compatibility: best-effort empty result without a running Surreal instance
        return []

    async def search_fts_detailed_async(self, query: str, limit: int = 10) -> List[Tuple[str, str, int]]:
        res = await self._db.query(
            "SELECT text, eid, ts FROM mention WHERE text @ $q ORDER BY ts DESC LIMIT $n;",
            {"q": query, "n": int(limit)},
        )
        try:
            rows = res[0]["result"]
            out: List[Tuple[str, str, int]] = []
            for row in rows:
                out.append((str(row.get("text", "")), str(row.get("eid", "")), int(row.get("ts", 0))))
            return out
        except Exception:
            return []

    # ---------- Bulk reads for rebuild ----------
    def get_all_edges(self, min_status: int = 0) -> List[Tuple[str, str, str, float]]:
        # Optional: provide a synchronous baseline (empty) for compatibility
        return []

    async def get_all_edges_async(self) -> List[Tuple[str, str, str, float]]:
        res = await self._db.query("SELECT in, out, predicate, confidence FROM edge LIMIT 10000;")
        try:
            rows = res[0]["result"]
            out: List[Tuple[str, str, str, float]] = []
            for row in rows:
                s = row.get("in").split(":", 1)[-1]
                d = row.get("out").split(":", 1)[-1]
                p = row.get("predicate")
                c = float(row.get("confidence", 0.5) or 0.5)
                out.append((s, p, d, c))
            return out
        except Exception:
            return []

    def get_all_edge_meta(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        return []

    # ---------- Flush ----------
    def flush(self) -> None:
        # Provide a synchronous no-op to keep compatibility on environments without surreal client.
        # Callers that need durability should call flush_async().
        pass

    async def flush_async(self) -> None:
        await self._flush_aliases()
        await self._flush_edges()
        await self._flush_mentions()

    # ---------- Constructors ----------
    @classmethod
    def from_env(cls) -> "SurrealMemoryStore":
        url = os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
        user = os.getenv("SURREALDB_USER", "root")
        pw = os.getenv("SURREALDB_PASS", "root")
        ns = os.getenv("SURREALDB_NAMESPACE", "localcat")
        db = os.getenv("SURREALDB_DATABASE", "memory")
        return cls(url, user, pw, ns, db)

