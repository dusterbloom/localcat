#!/usr/bin/env python3
"""
Manual retrieval dump tool

Usage:
  uv run python server/scripts/print_retrieval.py --query "Where do I live?" \
      --sqlite server/data/memory.db --lmdb server/data/graph.lmdb

Prints expanded entities, all candidates (KG + FTS), MMR pool, and final bullets.
Honors HOTMEM_RETRIEVAL_DEBUG env var for extra logging inside retriever.
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--entities", nargs="*", default=["you"], help="Seed entities, default=['you']")
    p.add_argument("--sqlite", default=os.getenv("HOTMEM_SQLITE", "memory.db"))
    p.add_argument("--lmdb", default=os.getenv("HOTMEM_LMDB_DIR", "graph.lmdb"))
    p.add_argument("--seed-edge", action="append", default=[], help="Seed KG edge as 'src:rel:dst' (can pass multiple)")
    p.add_argument("--seed-summary", default=None, help="Seed a summary mention text (e.g., 'You live in Sardinia')")
    args = p.parse_args()

    store = MemoryStore(Paths(sqlite_path=args.sqlite, lmdb_dir=args.lmdb))
    facade = HotMemoryFacade(store)

    # Optional seeding
    if args.seed_edge:
        now = int(__import__('time').time() * 1000)
        for i, spec in enumerate(args.seed_edge):
            try:
                s, r, d = spec.split(":", 2)
            except ValueError:
                print(f"Skipping malformed --seed-edge '{spec}', expected 'src:rel:dst'")
                continue
            store.observe_edge(s, r, d, 0.8, now + i)
    if args.seed_summary:
        sid = "session_cli"
        ts = int(__import__('time').time() * 1000)
        store.enqueue_mention(eid=f"summary:{sid}", text=args.seed_summary, ts=ts, sid=sid, tid=0)
    # Persist any seeds
    if args.seed_edge or args.seed_summary:
        store.flush()
    facade.rebuild_from_store()

    print(f"Rebuilt: entities={len(facade.entity_index)} edges=sum({sum(len(v) for v in facade.entity_index.values())})")
    result = facade.retriever.retrieve_context(args.query, args.entities, turn_id=1)

    # Dump everything
    print("\n=== Expanded Entities ===")
    # retriever returns expanded only inside RetrievalResult, but we can re-expand
    expanded = facade.retriever._expand_query_entities(args.entities, args.query)
    print(expanded)

    print("\n=== Candidates (raw) ===")
    cands = facade.retriever._gather_candidate_triples(args.query, expanded, intent=None)
    for i, (sc, ts, k, p) in enumerate(cands):
        if k == 'kg':
            s, r, d = p[:3]
            print(f"[{i:02d}] kind=kg score={sc:.3f} ts={ts} triple=({s}, {r}, {d})")
        else:
            print(f"[{i:02d}] kind={k} score={sc:.3f} ts={ts} text={p}")

    print("\n=== MMR Pool & Bullets ===")
    bullets = facade.retriever._apply_mmr_selection(args.query, cands, turn_id=1)
    print(f"Bullets selected: {len(bullets)}")
    for i, b in enumerate(bullets):
        print(f"  - {b}")


if __name__ == "__main__":
    sys.exit(main())
