#!/usr/bin/env python3
"""
Evaluate HotMem retrieval scoring with/without LEANN.

Usage:
  python3 scripts/eval_retrieval.py --leann=false "what is my dog's name"
"""
import argparse
import os
import tempfile

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths


def seed(store: MemoryStore):
    now = 1_700_000_000_000
    store.observe_edge('you', 'name', 'Alex', 0.9, now)
    store.observe_edge('you', 'lives_in', 'Berlin', 0.9, now)
    store.observe_edge('dog', 'name', 'Potola', 0.8, now)
    store.observe_edge('you', 'owns', 'dog', 0.8, now)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('query')
    ap.add_argument('--leann', default='false')
    args = ap.parse_args()

    os.environ['HOTMEM_USE_LEANN'] = args.leann

    with tempfile.TemporaryDirectory() as tdir:
        store = MemoryStore(Paths(sqlite_path=os.path.join(tdir, 'memory.db'), lmdb_dir=os.path.join(tdir, 'graph.lmdb')))
        hot = HotMemory(store)
        # Seed first, then rebuild hot indices from store so retrieval can find facts
        seed(store)
        hot.rebuild_from_store()

        bullets, _ = hot.process_turn(args.query, session_id='eval', turn_id=1)
        print("Query:", args.query)
        print("Bullets:")
        for b in bullets:
            print(" -", b)


if __name__ == '__main__':
    main()
