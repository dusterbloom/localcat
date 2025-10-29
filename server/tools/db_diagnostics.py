#!/usr/bin/env python3
"""
DB Diagnostics for LocalCat Memory

Summarizes the state of the memory SQLite DB:
 - Resolves DB path from args/env/.env (server/.env)
 - Table counts (edge, edge_source, conversation_turn, mention, chunks_content)
 - Users present (mention.eid) and sessions per user
 - Edge visibility per user (via ownership gate)
 - Slot tagging coverage (chunks_content.slot)

Usage:
  python server/tools/db_diagnostics.py [--db /absolute/path/to/memory.db] [--user peppi] [--limit 10]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import Optional, Tuple


def _load_env_db_path() -> Optional[str]:
    # 1) Env vars
    for key in ("MEMORY_SQLITE_PATH", "HOTMEM_SQLITE", "SQLITE_PATH", "SQLITE"):
        v = os.getenv(key)
        if v:
            return os.path.expanduser(os.path.expandvars(v))
    # 2) server/.env keys
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#') or '=' not in s:
                        continue
                    k, v = s.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"\'')
                    if k in ("MEMORY_SQLITE_PATH", "HOTMEM_SQLITE", "SQLITE_PATH", "SQLITE") and v:
                        return os.path.expanduser(os.path.expandvars(v))
        except Exception:
            pass
    # 3) fallback
    return os.path.abspath("memory.db")


def _sizeof(path: str) -> str:
    try:
        b = os.path.getsize(path)
        for unit in ['B','KB','MB','GB','TB']:
            if b < 1024.0:
                return f"{b:.1f}{unit}"
            b /= 1024.0
        return f"{b:.1f}PB"
    except Exception:
        return "unknown"


def _table_exists(cur, name: str) -> bool:
    try:
        row = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        return bool(row)
    except Exception:
        return False


def _count(cur, table: str) -> int:
    try:
        row = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _edge_count_visible_to_user(cur, user_id: str) -> int:
    try:
        row = cur.execute(
            """
            SELECT COUNT(DISTINCT es.edge_id)
            FROM edge_source es
            JOIN conversation_turn t ON t.id = es.turn_id
            JOIN mention m ON m.session_id = t.session_id
            WHERE m.eid = ?
            """,
            (user_id,),
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _sessions_by_user(cur, user_id: str, limit: int = 10):
    try:
        rows = cur.execute(
            """
            SELECT session_id, MAX(ts) AS last_ts, COUNT(*) AS rows
            FROM mention WHERE eid = ? GROUP BY session_id ORDER BY last_ts DESC LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()
        return rows
    except Exception:
        return []


def _slot_coverage(cur):
    try:
        rows = cur.execute(
            "SELECT slot, COUNT(*) FROM chunks_content GROUP BY slot ORDER BY COUNT(*) DESC"
        ).fetchall()
        return rows
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=None, help="Path to memory SQLite DB")
    ap.add_argument("--user", type=str, default=None, help="User ID to inspect")
    ap.add_argument("--limit", type=int, default=10, help="Limit for sample outputs")
    args = ap.parse_args()

    db_path = args.db or _load_env_db_path()
    db_path = os.path.abspath(db_path)

    print("=== Memory DB Diagnostics ===")
    print(f"DB Path: {db_path}")
    print(f"Exists: {os.path.exists(db_path)}  Size: {_sizeof(db_path) if os.path.exists(db_path) else 'N/A'}")

    if not os.path.exists(db_path):
        print("DB file does not exist. Check MEMORY_SQLITE_PATH/HOTMEM_SQLITE and restart server.")
        sys.exit(0)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Table existence
    tables = ["edge", "edge_source", "conversation_turn", "mention", "entity", "chunks_content", "chunks_fts"]
    print("\nTables present:")
    for t in tables:
        print(f" - {t}: {_table_exists(cur, t)}")

    # Counts
    print("\nCounts:")
    for t in tables:
        if _table_exists(cur, t):
            print(f" - {t}: {_count(cur, t)}")

    # Users present
    users = []
    if _table_exists(cur, "mention"):
        users = cur.execute(
            "SELECT eid, COUNT(*), COUNT(DISTINCT session_id) FROM mention GROUP BY eid ORDER BY COUNT(*) DESC"
        ).fetchall()
    print("\nUsers (eid) in mention:")
    for eid, rows, sess in users:
        print(f" - {eid}: rows={rows} sessions={sess}")

    # Slot coverage
    if _table_exists(cur, "chunks_content"):
        print("\nSlot coverage (chunks_content.slot):")
        for slot, cnt in _slot_coverage(cur):
            print(f" - {slot}: {cnt}")

    # Per-user visibility summary
    if args.user:
        uid = args.user
        print(f"\nVisibility for user '{uid}':")
        print(f" - visible edges (via ownership): {_edge_count_visible_to_user(cur, uid)}")
        sessions = _sessions_by_user(cur, uid, args.limit)
        if sessions:
            print(f" - recent sessions (limit {args.limit}):")
            for sid, last_ts, rows in sessions:
                print(f"    • {sid} last_ts={last_ts} rows={rows}")
        else:
            print(" - no sessions found in mention")

    con.close()


if __name__ == "__main__":
    main()

