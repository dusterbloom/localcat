#!/usr/bin/env python3
"""
TTL/Archival maintenance job for edges.

Policies (configurable):
- Demote edges with status=1 not updated in N days (default 30)
- Purge edges with status<=0 not updated in M days (default 90)

Dry-run support.

Examples:
  source .venv/bin/activate && cd server
  python scripts/admin/ttl_job.py --demote-days 30 --purge-days 90 --dry
  python scripts/admin/ttl_job.py --demote-days 30 --purge-days 120
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from components.memory.memory_store import MemoryStore, Paths  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--demote-days', type=int, default=30)
    ap.add_argument('--purge-days', type=int, default=90)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    now_ms = int(time.time() * 1000)
    demote_ms = args.demote_days * 24 * 60 * 60 * 1000
    purge_ms = args.purge_days * 24 * 60 * 60 * 1000

    store = MemoryStore(paths=Paths())
    cur = store.sql.cursor()

    # Demote old active edges
    demote_rows = cur.execute(
        "SELECT src, rel, dst, updated_at FROM edge WHERE status=1 AND updated_at > 0 AND updated_at <= ?",
        (int((now_ms - demote_ms) / 1000),)
    ).fetchall()

    print(f"Demote candidates: {len(demote_rows)} (status=1, older than {args.demote_days} days)")
    for (s, r, d, upd) in demote_rows:
        if args.dry:
            print(f"  DRY demote ({s}, {r}, {d}) last={upd}")
            continue
        store.negate_edge(str(s), str(r), str(d), conf=0.5, now_ts=now_ms)

    # Purge stale edges (status<=0) older than purge_days
    purge_rows = cur.execute(
        "SELECT src, rel, dst, status, updated_at FROM edge WHERE status <= 0 AND updated_at > 0 AND updated_at <= ?",
        (int((now_ms - purge_ms) / 1000),)
    ).fetchall()
    print(f"Purge candidates: {len(purge_rows)} (status<=0, older than {args.purge_days} days)")
    for (s, r, d, status, upd) in purge_rows:
        if args.dry:
            print(f"  DRY purge ({s}, {r}, {d}) status={status} last={upd}")
            continue
        store.hard_forget(str(s), str(r), str(d))

    if not args.dry:
        store.flush()
    print("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

