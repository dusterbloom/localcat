#!/usr/bin/env python3
"""
Ingest SQuAD contexts into LocalCat memory (Enhanced FTS) for retrieval A/B.

- Indexes SQuAD paragraphs as conversation documents in Enhanced FTS.
- Writes to a workspace-local SQLite+LMDB (HotPath fidelity) or a custom path.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Set


def main():
    ap = argparse.ArgumentParser(description="Ingest SQuAD contexts into LocalCat Enhanced FTS")
    ap.add_argument("--split", default="train", help="HF datasets split: train or validation")
    ap.add_argument("--limit", type=int, default=1000, help="Max contexts to ingest")
    ap.add_argument("--db", default="./data/squad.db", help="SQLite database path")
    ap.add_argument("--lmdb", default="./data/squad.lmdb", help="LMDB directory path (created)")
    args = ap.parse_args()

    # Import server modules
    import sys
    ROOT = Path(__file__).resolve().parents[2] / "server"
    sys.path.insert(0, str(ROOT))
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.enhanced_fts import EnhancedFTS

    # Prepare store
    paths = Paths(sqlite_path=args.db, lmdb_dir=args.lmdb)
    store = MemoryStore(paths)
    fts = EnhancedFTS(store)

    # Load SQuAD via datasets
    from datasets import load_dataset
    ds = load_dataset("squad", split=args.split)

    seen: Set[str] = set()
    session_id = "squad"
    turn = 0
    ingested = 0
    now_ms = int(time.time() * 1000)

    for ex in ds:
        ctx = (ex.get("context") or "").strip()
        if not ctx:
            continue
        # Deduplicate identical contexts
        key = hash(ctx)
        if key in seen:
            continue
        seen.add(key)
        try:
            fts.index_conversation(ctx, session_id=session_id, turn_id=turn, timestamp=now_ms)
            turn += 1
            ingested += 1
        except Exception:
            pass
        if ingested >= int(args.limit):
            break

    print(f"Ingested {ingested} SQuAD contexts into {args.db} (+ LMDB at {args.lmdb})")


if __name__ == "__main__":
    main()

