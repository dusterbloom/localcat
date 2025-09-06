#!/usr/bin/env python3
"""
Evaluate HotMem extraction on a small complex-sentence set.

Usage:
  python3 scripts/eval_extraction.py
"""
import os
import tempfile
from typing import List, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths


TEST_CASES = [
    ("Casablanca", "I came to Casablanca for the waters, even though I was misinformed because Casablanca is in the desert, and I stayed because Ilsa walked into my gin joint."),
    ("Hamlet", "Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and, by opposing, end them, is the question."),
    ("Pride & Prejudice", "Mr. Darcy owns Pemberley estate which has ten thousand acres, he rides horses every morning, and his sister Georgiana plays piano beautifully while their aunt Lady Catherine lives in Kent."),
    ("Great Gatsby", "Jay Gatsby, whose real name is James Gatz, was born in North Dakota, worked on Dan Cody's yacht for five years, and now lives in West Egg where he throws lavish parties."),
    ("Conversation", "Yesterday I met Sarah who works at Microsoft in Seattle, she graduated from Stanford in 2019, drives a Tesla, and said her brother Tom lives in Portland and teaches at Reed College."),
]


def run():
    with tempfile.TemporaryDirectory() as tdir:
        store = MemoryStore(Paths(sqlite_path=os.path.join(tdir, 'memory.db'), lmdb_dir=os.path.join(tdir, 'graph.lmdb')))
        hot = HotMemory(store)
        hot.prewarm('en')

        print("HotMem Extraction Evaluation (complex sentences)\n" + "="*60)
        for i, (name, text) in enumerate(TEST_CASES, 1):
            bullets, triples = hot.process_turn(text, session_id="eval", turn_id=i)
            print(f"\n{i}. {name}")
            print(f"Input: {text}")
            print(f"Triples ({len(triples)}):")
            for j, (s, r, d) in enumerate(triples, 1):
                print(f"  {j}. ({s}) -[{r}]-> ({d})")
            print(f"Bullets ({len(bullets)}):")
            for j, b in enumerate(bullets, 1):
                print(f"  {j}. {b}")


if __name__ == '__main__':
    run()

