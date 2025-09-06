"""
LocalCat: Compiled memory store (SQLite + LMDB), micro-batched, non-blocking
- SQLite: entities, edges history, mentions, BM25 FTS5
- LMDB: alias map and adjacency lists for O(1) hot-lookups
"""

import os
import lmdb
import msgpack
import sqlite3
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
        self.sqlite_path = self.sqlite_path or os.getenv("HOTMEM_SQLITE", "memory.db")
        self.lmdb_dir = self.lmdb_dir or os.getenv("HOTMEM_LMDB_DIR", "graph.lmdb")


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
        self._edges: List[Tuple[str, str, str, float, int, int, int, int]] = []
        self._mentions: List[Tuple[str, str, int, str, int]] = []
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
        # Ensure directory for SQLite file exists
        try:
            sql_dir = os.path.dirname(self.paths.sqlite_path or '')
            if sql_dir:
                os.makedirs(sql_dir, exist_ok=True)
        except Exception:
            pass
        # SQLite with optimal settings for write performance
        self.sql = sqlite3.connect(self.paths.sqlite_path, check_same_thread=False)
        self.sql.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=268435456;  -- 256MB memory map
            
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
              updated_at INT
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
        """)
        
        # LMDB with proper settings
        os.makedirs(self.paths.lmdb_dir, exist_ok=True)
        self.lenv = lmdb.open(
            self.paths.lmdb_dir, 
            map_size=2_147_483_648,  # 2GB
            max_dbs=8, 
            subdir=True,
            sync=False,  # Don't sync on every write
            writemap=True  # Use writemap for better performance
        )
        self.db_alias = self.lenv.open_db(b"alias")
        self.db_adj = self.lenv.open_db(b"adj")
    
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
        return hashlib.sha1(f"{s}|{r}|{d}".encode()).hexdigest()
    
    # ---------- Enqueue (non-blocking) ----------
    def enqueue_alias(self, alias: str, eid: str) -> None:
        self._aliases.append((alias, eid))
    
    def enqueue_edge_row(self, s, r, d, weight, pos, neg, status, ts):
        self._edges.append((s, r, d, float(weight), int(pos), int(neg), int(status), int(ts)))
    
    def enqueue_mention(self, eid: str, text: str, ts: float, sid: str, tid: int) -> None:
        self._mentions.append((eid, text[:500], int(ts), sid, int(tid)))  # Limit text length
    
    def flush_if_needed(self, max_ops: int = 16, max_ms: int = 500) -> None:
        total_ops = len(self._aliases) + len(self._edges) + len(self._mentions)
        elapsed_ms = (time.time() - self._last) * 1000
        
        if total_ops >= max_ops or elapsed_ms >= max_ms:
            self.flush()
    
    # ---------- Flush (batched) ----------
    def flush(self) -> None:
        if not (self._aliases or self._edges or self._mentions):
            return
        
        start = time.perf_counter()
        
        try:
            with contextlib.ExitStack() as stack:
                # Single transaction for both databases
                txn = stack.enter_context(self.lenv.begin(write=True))
                cur = self.sql.cursor()
                
                # Batch process aliases
                for alias, eid in self._aliases:
                    txn.put(f"alias:{alias}".encode(), eid.encode(), db=self.db_alias, overwrite=True)
                    # Update entity aliases in SQLite
                    cur.execute(
                        "INSERT INTO entity(id, name, aliases, created_at, updated_at) "
                        "VALUES(?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET aliases = aliases || ',' || ?, updated_at = ?",
                        (eid, eid, alias, _now_i(), _now_i(), alias, _now_i())
                    )
                
                # Batch process edges with adjacency updates
                for s, r, d, w, pos, neg, status, ts in self._edges:
                    # Update LMDB adjacency
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
                        INSERT INTO edge(id, src, rel, dst, weight, pos, neg, status, updated_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            weight=excluded.weight,
                            pos=excluded.pos,
                            neg=excluded.neg,
                            status=excluded.status,
                            updated_at=excluded.updated_at
                    """, (eid, s, r, d, w, pos, neg, status, int(ts)))
                
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
                
                self.sql.commit()
                
        except Exception as e:
            logger.error(f"Flush failed: {e}")
            # Don't lose data on error - will retry next flush
            return
        
        # Clear queues only on success
        self._aliases.clear()
        self._edges.clear()
        self._mentions.clear()
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
    
    def observe_edge(self, s: str, r: str, d: str, conf: float, now_ts: int) -> None:
        """Create/reinforce (s,r,d) with positive evidence."""
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
        self.enqueue_edge_row(s, r, d, w, pos, neg, self._status_from_weight(w), now_ts)
        self.flush_if_needed()
    
    def negate_edge(self, s: str, r: str, d: str, conf: float, now_ts: int) -> None:
        """Demote (s,r,d) with negative/contradicting evidence."""
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
        
        self.enqueue_edge_row(s, r, d, w, pos, neg, self._status_from_weight(w), now_ts)
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
    def search_fts(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Full-text search using SQLite FTS5 (text, eid).

        Kept stable for existing callers. See `search_fts_detailed` to also get timestamps.
        """
        cur = self.sql.cursor()
        results: List[Tuple[str, str]] = []
        for (text, eid) in cur.execute(
            "SELECT text, eid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit)
        ):
            results.append((str(text or ''), str(eid or '')))
        return results

    def search_fts_detailed(self, query: str, limit: int = 10) -> List[Tuple[str, str, int]]:
        """Full-text search returning (text, eid, ts).

        Uses the same FTS table which stores `ts` as an UNINDEXED column.
        """
        cur = self.sql.cursor()
        results: List[Tuple[str, str, int]] = []
        for (text, eid, ts) in cur.execute(
            "SELECT text, eid, ts FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit)
        ):
            try:
                results.append((str(text or ''), str(eid or ''), int(ts or 0)))
            except Exception:
                results.append((str(text or ''), str(eid or ''), 0))
        return results
    
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
