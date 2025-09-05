#!/usr/bin/env python3
"""
Inspect HotMem storage and retrieval without running the full server.

Usage:
  python server/debug_hotmem_state.py --sqlite memory.db --lmdb graph.lmdb --query "What is my name?"

Shows:
  - Edge count and sample edges from SQLite
  - Preview memory bullets for a query (non-mutating)
"""

import argparse
import os
from typing import Optional

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory


def main(sqlite: Optional[str], lmdb_dir: Optional[str], query: Optional[str], limit: int):
    paths = Paths(sqlite_path=sqlite, lmdb_dir=lmdb_dir)
    store = MemoryStore(paths)
    hot = HotMemory(store)
    hot.rebuild_from_store()

    edges = store.get_all_edges()
    print(f"Total edges: {len(edges)}")
    for i, (s, r, d, w) in enumerate(edges[:limit], 1):
        print(f"  {i:>2}. ({s}, {r}, {d})  w={w:.2f}")

    if query:
        print("\nPreview retrieval (non-mutating):")
        preview = hot.preview_bullets(query)
        print(f"  Entities: {preview['entities']}")
        for b in preview['bullets']:
            print(f"  {b}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=os.getenv("HOTMEM_SQLITE", "memory.db"))
    ap.add_argument("--lmdb", dest="lmdb_dir", default=os.getenv("HOTMEM_LMDB_DIR", "graph.lmdb"))
    ap.add_argument("--query", help="Optional query text to preview bullets", default=None)
    ap.add_argument("--limit", type=int, default=20, help="How many edges to print")
    args = ap.parse_args()
    main(args.sqlite, args.lmdb_dir, args.query, args.limit)

