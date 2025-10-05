"""
LocalCat: Compiled memory store (SQLite + LMDB), micro-batched, non-blocking
- SQLite: entities, edges history, mentions, BM25 FTS5
- LMDB: alias map and adjacency lists for O(1) hot-lookups
"""

import os
import lmdb
import msgpack
import sqlite3
import json
import hashlib
import time
import shutil
import contextlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from loguru import logger


@dataclass
class Paths:
    sqlite_path: str = None
    lmdb_dir: str = None

    def __post_init__(self):
        # Only set defaults if not explicitly provided (including ":memory:")
        if self.sqlite_path is None:
            self.sqlite_path = os.getenv("HOTMEM_SQLITE", "memory.db")
        # For LMDB, None means disabled, so don't set a default in that case
        if self.lmdb_dir is None:
            env_lmdb = os.getenv("HOTMEM_LMDB_DIR")
            if env_lmdb:
                self.lmdb_dir = env_lmdb
            # else keep it as None to disable LMDB
        logger.debug(f"Paths initialized: sqlite_path={self.sqlite_path!r}, lmdb_dir={self.lmdb_dir!r}")


def _now_i() -> int:
    return int(time.time())


class MemoryStore:
    """
    Durable mirror of operational RAM memory:
      - enqueue_* methods never block the hot loop
      - flush_if_needed() batches writes every N ops / M ms
      - alias / adjacency reads are O(1) via LMDB (memory-mapped)
      - Automatic corruption recovery
    """
    def __init__(self, paths: Paths = None):
        self.paths = paths or Paths()
        self._init_with_recovery()
        
        # Batch queues
        self._aliases: List[Tuple[str, str]] = []
        # Edge queue supports optional meta JSON (9th element)
        self._edges: List[Tuple] = []
        self._mentions: List[Tuple[str, str, int, str, int]] = []

        # Provenance queues
        self._turns: List[Tuple[str, str, str, int, int]] = []  # (id, text, sid, tid, ts)
        self._edge_sources: List[Tuple[str, str, int]] = []  # (edge_id, turn_id, ts)

        self._last = time.time()

        # Performance monitoring
        self.metrics = defaultdict(list)
    
    def _init_with_recovery(self):
        """Initialize databases with automatic corruption recovery"""
        try:
            self._init_databases()
        except Exception as e:
            logger.error(f"Database corruption detected: {e}")
            self._recover_from_corruption()
    
    def _init_databases(self):
        """Initialize SQLite and LMDB databases"""
        # SQLite with optimal settings for write performance
        logger.debug(f"Connecting to SQLite database at: {self.paths.sqlite_path!r}")
        self.sql = sqlite3.connect(self.paths.sqlite_path, check_same_thread=False)

        # Use WAL mode only for file-based databases, not for :memory:
        if self.paths.sqlite_path != ":memory:":
            journal_mode = "WAL"
        else:
            journal_mode = "MEMORY"

        self.sql.executescript(f"""
            PRAGMA journal_mode={journal_mode};
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=268435456;  -- 256MB memory map
            PRAGMA foreign_keys=ON;  -- Enable foreign key constraints
            
            CREATE TABLE IF NOT EXISTS entity(
              id TEXT PRIMARY KEY, 
              name TEXT, 
              aliases TEXT,
              created_at INT, 
              updated_at INT
            );
            
            CREATE TABLE IF NOT EXISTS edge(
              id TEXT PRIMARY KEY,
              src TEXT, 
              rel TEXT, 
              dst TEXT,
              weight REAL DEFAULT 1.0, 
              pos INT DEFAULT 0, 
              neg INT DEFAULT 0,
              status INT DEFAULT 1,  -- 1=active, 0=stale, -1=archived, -9=deleted
              updated_at INT,
              meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edge_src ON edge(src);
            CREATE INDEX IF NOT EXISTS idx_edge_status ON edge(status);
            
            CREATE TABLE IF NOT EXISTS mention(
              id TEXT PRIMARY KEY, 
              eid TEXT, 
              text TEXT,
              ts INT, 
              session_id TEXT, 
              turn_id INT
            );
            CREATE INDEX IF NOT EXISTS idx_mention_eid ON mention(eid);
            
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
              USING fts5(
                text,
                eid UNINDEXED,
                rel UNINDEXED,
                dst UNINDEXED,
                ts UNINDEXED,
                tokenize='porter'
              );

            -- Conversation-first provenance tables
            CREATE TABLE IF NOT EXISTS conversation_turn(
              id TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              session_id TEXT NOT NULL,
              turn_id INT NOT NULL,
              ts INT NOT NULL,
              UNIQUE(session_id, turn_id)
            );
            CREATE INDEX IF NOT EXISTS idx_turn_session ON conversation_turn(session_id, turn_id);
            CREATE INDEX IF NOT EXISTS idx_turn_ts ON conversation_turn(ts DESC);

            CREATE TABLE IF NOT EXISTS edge_source(
              edge_id TEXT NOT NULL,
              turn_id TEXT NOT NULL,
              extracted_at INT NOT NULL,
              PRIMARY KEY (edge_id, turn_id),
              FOREIGN KEY (edge_id) REFERENCES edge(id) ON DELETE CASCADE,
              FOREIGN KEY (turn_id) REFERENCES conversation_turn(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_source_edge ON edge_source(edge_id);
            CREATE INDEX IF NOT EXISTS idx_source_turn ON edge_source(turn_id);
        """)
        
        # LMDB with proper settings (skip if lmdb_dir is None)
        if self.paths.lmdb_dir:
            os.makedirs(self.paths.lmdb_dir, exist_ok=True)
            self.lenv = lmdb.open(
                self.paths.lmdb_dir,
                map_size=2_147_483_648,  # 2GB
                max_dbs=8,
                subdir=True,
                sync=False,  # Don't sync on every write
                writemap=True  # Use writemap for better performance
            )
        else:
            self.lenv = None

        if self.lenv:
            self.db_alias = self.lenv.open_db(b"alias")
            self.db_adj = self.lenv.open_db(b"adj")
        else:
            self.db_alias = None
            self.db_adj = None

        # Migrate: ensure 'meta' column exists on edge (safe no-op if present)
        try:
            self.sql.execute("ALTER TABLE edge ADD COLUMN meta TEXT")
            self.sql.commit()
        except Exception:
            pass
    
    def _recover_from_corruption(self):
        """Recover from database corruption"""
        logger.info("Starting database recovery...")
        
        # Backup corrupted files
        if os.path.exists(self.paths.lmdb_dir):
            backup_dir = f"{self.paths.lmdb_dir}.corrupted.{int(time.time())}"
            shutil.move(self.paths.lmdb_dir, backup_dir)
            logger.info(f"Backed up corrupted LMDB to {backup_dir}")
        
        # Re-initialize databases
        self._init_databases()
        
        # Try to rebuild LMDB from SQLite
        try:
            self._rebuild_lmdb_from_sqlite()
            logger.info("Recovery completed: rebuilt LMDB from SQLite")
        except Exception as e:
            logger.warning(f"Could not rebuild from SQLite: {e}")
            logger.info("Starting with fresh databases")
    
    def _rebuild_lmdb_from_sqlite(self):
        """Rebuild LMDB indices from SQLite (source of truth)"""
        with self.lenv.begin(write=True) as txn:
            cur = self.sql.cursor()
            
            # Rebuild alias index
            for (eid, aliases_json) in cur.execute("SELECT id, aliases FROM entity WHERE aliases IS NOT NULL"):
                if aliases_json:
                    for alias in aliases_json.split(','):
                        txn.put(f"alias:{alias}".encode(), eid.encode(), db=self.db_alias, overwrite=True)
            
            # Rebuild adjacency index
            for (src, rel, dst, w, pos, neg, status, ts) in cur.execute(
                "SELECT src, rel, dst, weight, pos, neg, status, updated_at FROM edge WHERE status >= 0"
            ):
                key = f"adj:{src}|{rel}".encode()
                old = txn.get(key, db=self.db_adj)
                arr = msgpack.loads(old) if old else []
                arr.extend([dst, float(w), int(ts), int(pos), int(neg), int(status)])
                txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)
    
    @staticmethod
    def edge_id(s, r, d) -> str:
        """Compute a canonical edge ID.

        Canonicalization aligns with extraction refiners so that callers using
        raw text (e.g., 'I', 'Google') can still resolve the stored edge IDs.
        - Lowercase subject/object and strip leading determiners / trailing "'s".
        - Map first-person pronouns (i/me/my/mine/myself) → 'you'.
        - Relation is used as-is (typically already canonicalized like 'v:work_at').
        """
        def _canon_entity(x: str) -> str:
            if not isinstance(x, str):
                return str(x)
            t = x.strip().lower()
            if t in {"i", "me", "my", "mine", "myself"}:
                return "you"
            # strip leading simple determiners/possessives
            for det in ("the", "a", "an", "my", "your", "his", "her", "our", "their", "its"):
                if t.startswith(det + " "):
                    t = t[len(det) + 1 :]
                    break
            if t.endswith("'s"):
                t = t[:-2]
            return t.strip()

        s2 = _canon_entity(s)
        d2 = _canon_entity(d)
        r2 = r if isinstance(r, str) else str(r)
        return hashlib.sha1(f"{s2}|{r2}|{d2}".encode()).hexdigest()
    
    # ---------- Enqueue (non-blocking) ----------
    def enqueue_alias(self, alias: str, eid: str) -> None:
        self._aliases.append((alias, eid))
    
    def enqueue_edge_row(self, s, r, d, weight, pos, neg, status, ts, meta: Optional[Dict[str, Any]] = None):
        meta_json = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else (meta if isinstance(meta, str) else None)
        self._edges.append((s, r, d, float(weight), int(pos), int(neg), int(status), int(ts), meta_json))
    
    def enqueue_mention(self, eid: str, text: str, ts: float, sid: str, tid: int) -> None:
        self._mentions.append((eid, text[:500], int(ts), sid, int(tid)))  # Limit text length

    @staticmethod
    def turn_id(session_id: str, turn_id: int) -> str:
        """Generate stable turn ID from session + turn number"""
        return hashlib.sha1(f"{session_id}|{turn_id}".encode()).hexdigest()

    def enqueue_turn(self, text: str, session_id: str, turn_id: int, ts: int) -> str:
        """
        Store conversation turn (non-blocking, idempotent)

        Args:
            text: Full conversation text
            session_id: Session identifier
            turn_id: Turn number within session
            ts: Timestamp in milliseconds

        Returns:
            Turn ID (hash) for linking to edges
        """
        tid = self.turn_id(session_id, turn_id)
        self._turns.append((tid, text[:2000], session_id, turn_id, ts))  # Limit text to 2KB
        return tid

    def enqueue_edge_source(self, edge_id: str, turn_id: str, ts: int) -> None:
        """
        Link edge to conversation turn (non-blocking)

        Args:
            edge_id: Edge ID from self.edge_id(s, r, d)
            turn_id: Turn ID from self.enqueue_turn()
            ts: Extraction timestamp in milliseconds
        """
        self._edge_sources.append((edge_id, turn_id, ts))

    def flush_if_needed(self, max_ops: int = 16, max_ms: int = 500) -> None:
        total_ops = (len(self._aliases) + len(self._edges) + len(self._mentions) +
                     len(self._turns) + len(self._edge_sources))
        elapsed_ms = (time.time() - self._last) * 1000

        if total_ops >= max_ops or elapsed_ms >= max_ms:
            self.flush()
    
    # ---------- Flush (batched) ----------
    def flush(self) -> None:
        if not (self._aliases or self._edges or self._mentions or
                self._turns or self._edge_sources):
            return
        
        start = time.perf_counter()
        
        try:
            with contextlib.ExitStack() as stack:
                # Single transaction for both databases when LMDB enabled
                txn = stack.enter_context(self.lenv.begin(write=True)) if self.lenv else None
                cur = self.sql.cursor()
                
                # Batch process aliases
                for alias, eid in self._aliases:
                    if txn is not None:
                        txn.put(f"alias:{alias}".encode(), eid.encode(), db=self.db_alias, overwrite=True)
                    # Update entity aliases in SQLite
                    cur.execute(
                        "INSERT INTO entity(id, name, aliases, created_at, updated_at) "
                        "VALUES(?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET aliases = aliases || ',' || ?, updated_at = ?",
                        (eid, eid, alias, _now_i(), _now_i(), alias, _now_i())
                    )
                
                # Batch process edges with adjacency updates
                for row in self._edges:
                    if len(row) == 9:
                        s, r, d, w, pos, neg, status, ts, meta_json = row
                    else:
                        s, r, d, w, pos, neg, status, ts = row
                        meta_json = None
                    # Update LMDB adjacency if available
                    if txn is not None:
                        key = f"adj:{s}|{r}".encode()
                        old = txn.get(key, db=self.db_adj)
                        if old:
                            arr = msgpack.loads(old)
                            # Check if edge already exists and update
                            found = False
                            for i in range(0, len(arr), 6):
                                if arr[i] == d:
                                    arr[i+1] = w
                                    arr[i+2] = ts
                                    arr[i+3] = pos
                                    arr[i+4] = neg
                                    arr[i+5] = status
                                    found = True
                                    break
                            if not found:
                                arr.extend([d, w, ts, pos, neg, status])
                        else:
                            arr = [d, w, ts, pos, neg, status]
                        txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)
                    
                    # Update SQLite
                    eid = self.edge_id(s, r, d)
                    cur.execute("""
                        INSERT INTO edge(id, src, rel, dst, weight, pos, neg, status, updated_at, meta)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            weight=excluded.weight,
                            pos=excluded.pos,
                            neg=excluded.neg,
                            status=excluded.status,
                            updated_at=excluded.updated_at,
                            meta=COALESCE(excluded.meta, edge.meta)
                    """, (eid, s, r, d, w, pos, neg, status, int(ts), meta_json))
                
                # Batch process mentions
                for eid, text, ts, sid, tid in self._mentions:
                    mid = hashlib.sha1(f"{eid}|{ts}|{sid}|{tid}".encode()).hexdigest()
                    cur.execute(
                        "INSERT OR IGNORE INTO mention(id, eid, text, ts, session_id, turn_id) "
                        "VALUES(?, ?, ?, ?, ?, ?)",
                        (mid, eid, text, int(ts), sid, tid)
                    )
                    # Update FTS index
                    cur.execute(
                        "INSERT INTO chunks_fts(text, eid, rel, dst, ts) VALUES(?, ?, ?, ?, ?)",
                        (text, eid, "", "", int(ts))
                    )

                # Batch process conversation turns
                for tid, text, sid, turn_num, ts in self._turns:
                    cur.execute(
                        "INSERT OR IGNORE INTO conversation_turn(id, text, session_id, turn_id, ts) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (tid, text, sid, turn_num, ts)
                    )
                    # Index conversation in FTS for convo retrieval
                    cur.execute(
                        "INSERT INTO chunks_fts(text, eid, rel, dst, ts) VALUES(?, ?, ?, ?, ?)",
                        (text, "conversation", "", "", ts)
                    )

                # Batch process edge sources
                for edge_id, turn_id, ts in self._edge_sources:
                    cur.execute(
                        "INSERT OR IGNORE INTO edge_source(edge_id, turn_id, extracted_at) "
                        "VALUES(?, ?, ?)",
                        (edge_id, turn_id, ts)
                    )

                self.sql.commit()
                
        except Exception as e:
            logger.error(f"Flush failed: {e}")
            # Don't lose data on error - will retry next flush
            return
        
        # Clear queues only on success
        self._aliases.clear()
        self._edges.clear()
        self._mentions.clear()
        self._turns.clear()
        self._edge_sources.clear()
        self._last = time.time()
        
        # Track performance
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics['flush_ms'].append(elapsed_ms)
        if len(self.metrics['flush_ms']) > 100:
            self.metrics['flush_ms'] = self.metrics['flush_ms'][-100:]
    
    # ---------- Fast reads ----------
    def resolve_alias(self, alias: str) -> Optional[str]:
        with self.lenv.begin() as txn:
            v = txn.get(f"alias:{alias}".encode(), db=self.db_alias)
        return v.decode() if v else None
    
    def neighbors(self, s: str, r: str) -> List[Tuple[str, float, int, int, int, int]]:
        with self.lenv.begin() as txn:
            raw = txn.get(f"adj:{s}|{r}".encode(), db=self.db_adj)
        if not raw:
            return []
        arr = msgpack.loads(raw)
        out = []
        for i in range(0, len(arr), 6):
            dst, w, ts, pos, neg, status = arr[i:i+6]
            out.append((dst, float(w), int(ts), int(pos), int(neg), int(status)))
        return out
    
    # ---------- Edge lifecycle ops (hot-path safe) ----------
    @staticmethod
    def _status_from_weight(w: float) -> int:
        return 1 if w >= 0.25 else (0 if w >= 0.10 else -1)
    
    @staticmethod
    def _alpha(conf: float, base: float = 0.15, lo: float = 0.05, hi: float = 0.35) -> float:
        return max(lo, min(hi, base * conf))
    
    def observe_edge(self, s: str, r: str, d: str, conf: float, now_ts: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """Create/reinforce (s,r,d) with positive evidence."""
        if self.lenv is not None:
            # For immediate updates, we write directly to LMDB
            with self.lenv.begin(write=True) as txn:
                key = f"adj:{s}|{r}".encode()
                old = txn.get(key, db=self.db_adj)
                arr = msgpack.loads(old) if old else []

                found = False
                w = conf
                pos = 1
                neg = 0

                # Scan for existing edge
                for i in range(0, len(arr), 6):
                    if arr[i] == d:
                        # Exponential weighted average toward 1.0
                        a = self._alpha(conf, base=0.15)
                        w = (1 - a) * float(arr[i+1]) + a * 1.0
                        arr[i+1] = w
                        arr[i+2] = now_ts
                        pos = int(arr[i+3]) + 1
                        arr[i+3] = pos
                        arr[i+5] = self._status_from_weight(w)
                        found = True
                        break

                if not found:
                    # New edge
                    w = min(0.75, conf)
                    arr.extend([d, w, now_ts, 1, 0, 1])

                txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)

            # Enqueue for SQLite persistence
            self.enqueue_edge_row(s, r, d, w, pos, neg, self._status_from_weight(w), now_ts, meta)
            self.flush_if_needed()
        else:
            # LMDB disabled: enqueue a conservative SQLite update only
            w = min(0.75, conf)
            pos = 1
            neg = 0
            self.enqueue_edge_row(s, r, d, w, pos, neg, self._status_from_weight(w), now_ts, meta)
            self.flush_if_needed()
    
    def negate_edge(self, s: str, r: str, d: str, conf: float, now_ts: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """Demote (s,r,d) with negative/contradicting evidence."""
        if self.lenv is not None:
            with self.lenv.begin(write=True) as txn:
                key = f"adj:{s}|{r}".encode()
                old = txn.get(key, db=self.db_adj)
                arr = msgpack.loads(old) if old else []

                found = False
                w = 0.1
                pos = 0
                neg = 1

                for i in range(0, len(arr), 6):
                    if arr[i] == d:
                        # Exponential weighted average toward 0.0
                        a = self._alpha(conf, base=0.20, hi=0.50)
                        w = (1 - a) * float(arr[i+1]) + a * 0.0
                        arr[i+1] = w
                        arr[i+2] = now_ts
                        neg = int(arr[i+4]) + 1
                        arr[i+4] = neg
                        pos = int(arr[i+3])
                        arr[i+5] = self._status_from_weight(w)
                        found = True
                        break

                if not found:
                    arr.extend([d, 0.10, now_ts, 0, 1, 0])  # Weak & stale

                txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)

            self.enqueue_edge_row(s, r, d, w, pos, neg, self._status_from_weight(w), now_ts, meta)
            self.flush_if_needed()
        else:
            # LMDB disabled: conservative SQLite update only
            w = 0.10
            pos = 0
            neg = 1
            self.enqueue_edge_row(s, r, d, w, pos, neg, self._status_from_weight(w), now_ts, meta)
            self.flush_if_needed()
    
    def hard_forget(self, s: str, r: str = None, d: str = None) -> None:
        """Explicit user forget (purge from LMDB + tombstone in SQLite)."""
        with self.lenv.begin(write=True) as txn:
            if r is None:
                # Delete all edges from source
                cur = self.sql.cursor()
                for (rel,) in cur.execute("SELECT DISTINCT rel FROM edge WHERE src=?", (s,)):
                    txn.delete(f"adj:{s}|{rel}".encode(), db=self.db_adj)
                cur.execute("UPDATE edge SET weight=0, status=-9 WHERE src=?", (s,))
                self.sql.commit()
            elif d is None:
                # Delete all edges with (s,r,*)
                txn.delete(f"adj:{s}|{r}".encode(), db=self.db_adj)
                cur = self.sql.cursor()
                cur.execute("UPDATE edge SET weight=0, status=-9 WHERE src=? AND rel=?", (s, r))
                self.sql.commit()
            else:
                # Delete specific edge
                key = f"adj:{s}|{r}".encode()
                old = txn.get(key, db=self.db_adj)
                if old:
                    arr = msgpack.loads(old)
                    arr2 = []
                    for i in range(0, len(arr), 6):
                        if arr[i] != d:
                            arr2.extend(arr[i:i+6])
                    if arr2:
                        txn.put(key, msgpack.dumps(arr2), db=self.db_adj, overwrite=True)
                    else:
                        txn.delete(key, db=self.db_adj)
                
                cur = self.sql.cursor()
                cur.execute("UPDATE edge SET weight=0, status=-9 WHERE src=? AND rel=? AND dst=?", (s, r, d))
                self.sql.commit()
    
    # ---------- Search operations ----------
    def search_fts(self, query: str, limit: int = 10) -> List[Tuple[str, str, int]]:
        """Full-text search using SQLite FTS5, including timestamp for recency."""
        cur = self.sql.cursor()
        results: List[Tuple[str, str, int]] = []
        for (text, eid, ts) in cur.execute(
            "SELECT text, eid, ts FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit)
        ):
            results.append((str(text), str(eid), int(ts)))
        return results

    def search_fts_scoped(self, query: str, eids: List[str], limit: int = 10) -> List[Tuple[str, str, int]]:
        """FTS search restricted to specific eids (user/session scoped)."""
        if not eids:
            return self.search_fts(query, limit)
        cur = self.sql.cursor()
        results: List[Tuple[str, str, int]] = []
        try:
            placeholders = ",".join(["?"] * len(eids))
            sql = f"SELECT text, eid, ts FROM chunks_fts WHERE chunks_fts MATCH ? AND eid IN ({placeholders}) ORDER BY rank LIMIT ?"
            params = [query, *eids, int(limit)]
            for (text, eid, ts) in cur.execute(sql, params):
                results.append((str(text), str(eid), int(ts)))
        except Exception as e:
            logger.warning(f"search_fts_scoped failed: {e}")
        return results

    def is_session_owned_by_user(self, session_id: str, user_id: str) -> bool:
        """Return True if there is at least one mention for (eid=user_id, session_id=session_id)."""
        if not session_id or not user_id:
            return False
        try:
            cur = self.sql.cursor()
            row = cur.execute(
                "SELECT 1 FROM mention WHERE session_id = ? AND eid = ? LIMIT 1",
                (session_id, user_id),
            ).fetchone()
            return bool(row)
        except Exception:
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                import statistics
                metrics[key] = {
                    'mean': statistics.mean(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values),
                    'count': len(values)
                }
        return metrics

    # ---------- Recent items helpers ----------
    def get_recent_chunks_by_eid(self, eid: str, limit: int = 5) -> List[Tuple[str, int]]:
        """Return recent FTS chunks for a given eid ordered by timestamp desc.

        Returns list of (text, ts).
        """
        cur = self.sql.cursor()
        out: List[Tuple[str, int]] = []
        try:
            for (text, ts) in cur.execute(
                "SELECT text, ts FROM chunks_fts WHERE eid = ? ORDER BY ts DESC LIMIT ?",
                (eid, int(limit)),
            ):
                out.append((str(text), int(ts)))
        except Exception:
            pass
        return out

    def get_recent_chunks_by_user(self, user_id: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Return recent FTS chunks across all sessions for a given user ordered by timestamp desc.

        Returns list of (text, ts).
        """
        return self.get_recent_chunks_by_eid(user_id, limit)

    def get_sessions_by_user(self, user_id: str) -> List[str]:
        """Return all session IDs for a given user."""
        cur = self.sql.cursor()
        sessions = []
        try:
            for (session_id,) in cur.execute(
                "SELECT DISTINCT session_id FROM mention WHERE eid = ? ORDER BY session_id",
                (user_id,),
            ):
                sessions.append(str(session_id))
        except Exception:
            pass
        return sessions

    def get_recent_chunks_by_session(self, user_id: str, session_id: str, limit: int = 5) -> List[Tuple[str, int]]:
        """Return recent FTS chunks for a specific user session ordered by timestamp desc.

        Returns list of (text, ts).
        """
        cur = self.sql.cursor()
        out: List[Tuple[str, int]] = []
        try:
            for (text, ts) in cur.execute(
                "SELECT text, ts FROM mention WHERE eid = ? AND session_id = ? ORDER BY ts DESC LIMIT ?",
                (user_id, session_id, int(limit)),
            ):
                out.append((str(text), int(ts)))
        except Exception:
            pass
        return out

    # ---------- Bulk reads for rebuild ----------
    def get_all_edges(self, min_status: int = 0) -> List[Tuple[str, str, str, float]]:
        """Return (src, rel, dst, weight) for all edges with status >= min_status.

        Used to rebuild in-memory indices at startup without blocking hot path later.
        """
        cur = self.sql.cursor()
        rows = cur.execute(
            "SELECT src, rel, dst, weight FROM edge WHERE status >= ?",
            (int(min_status),)
        ).fetchall()
        return [(str(s), str(r), str(d), float(w)) for (s, r, d, w) in rows]

    # ---------- Provenance Query Helpers ----------

    def get_edge_provenance(self, edge_id: str) -> List[Tuple[str, str, int, int]]:
        """
        Get all conversation turns that produced this edge

        Args:
            edge_id: Edge ID from self.edge_id(s, r, d)

        Returns:
            List of (text, session_id, turn_id, extracted_at) tuples
            Ordered by most recent first
        """
        cur = self.sql.cursor()
        return cur.execute("""
            SELECT t.text, t.session_id, t.turn_id, es.extracted_at
            FROM edge_source es
            JOIN conversation_turn t ON es.turn_id = t.id
            WHERE es.edge_id = ?
            ORDER BY es.extracted_at DESC
        """, (edge_id,)).fetchall()

    def get_turn_extractions(self, session_id: str, turn_id: int) -> List[Tuple[str, str, str, float]]:
        """
        Get all edges extracted from a conversation turn

        Args:
            session_id: Session identifier
            turn_id: Turn number within session

        Returns:
            List of (src, rel, dst, weight) tuples
        """
        tid = self.turn_id(session_id, turn_id)
        cur = self.sql.cursor()
        return cur.execute("""
            SELECT e.src, e.rel, e.dst, e.weight
            FROM edge_source es
            JOIN edge e ON es.edge_id = e.id
            WHERE es.turn_id = ?
            ORDER BY e.weight DESC
        """, (tid,)).fetchall()

    def get_conversation(self, session_id: str, limit: int = 100) -> List[Tuple[int, str, int]]:
        """
        Retrieve full conversation by session

        Args:
            session_id: Session identifier
            limit: Maximum turns to return

        Returns:
            List of (turn_id, text, timestamp) tuples ordered by turn
        """
        cur = self.sql.cursor()
        return cur.execute("""
            SELECT turn_id, text, ts
            FROM conversation_turn
            WHERE session_id = ?
            ORDER BY turn_id ASC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    def get_edge_sources_count(self, edge_id: str) -> int:
        """
        Count how many conversation turns produced this edge
        Useful for confidence scoring (more sources = higher confidence)

        Args:
            edge_id: Edge ID

        Returns:
            Number of distinct source conversations
        """
        cur = self.sql.cursor()
        result = cur.execute("""
            SELECT COUNT(*) FROM edge_source WHERE edge_id = ?
        """, (edge_id,)).fetchone()
        return result[0] if result else 0
