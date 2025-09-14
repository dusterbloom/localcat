#!/usr/bin/env python3
"""
Run Level1to3_text.md through the memory pipeline (no audio) and show persisted triples.

- Uses current .env defaults, but disables LEANN/fusion to focus on extraction + persistence.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade


def parse_level1to3(md_path: Path) -> List[Tuple[str, str]]:
    raw = md_path.read_text(encoding='utf-8')
    entries: List[Tuple[str, str]] = []
    cur = None
    for line in raw.splitlines():
        if line.startswith('###'):
            cur = line.strip('# ').strip()
            continue
        if cur and line.strip() and not line.strip().startswith('#'):
            entries.append((cur, line.strip()))
            cur = None
    return entries


def run() -> int:
    # Load .env from server root
    env_path = Path(__file__).resolve().parents[2] / '.env'
    if env_path.exists():
        load_dotenv(str(env_path), override=True)

    # Force lite coref and disable fusion for this run
    os.environ['HOTMEM_COREF_MODE'] = 'lite'
    os.environ['HOTMEM_USE_COREF'] = 'true'
    os.environ['HOTMEM_USE_LEANN'] = 'false'
    os.environ['HOTMEM_RETRIEVAL_FUSION'] = 'false'

    # Prepare store under /tmp
    store = MemoryStore(paths=Paths(
        sqlite_path=str(Path('/tmp/hotmem_l1to3.db')),
        lmdb_dir=str(Path('/tmp/hotmem_l1to3.lmdb')),
    ))
    facade = HotMemoryFacade(store)

    md = Path(__file__).resolve().parents[3] / 'backlog' / 'docs' / 'Level1to3_text.md'
    entries = parse_level1to3(md)

    session_id = 'level1to3'
    total_ms = 0.0
    print(f"Found {len(entries)} texts in {md}")
    for i, (title, text) in enumerate(entries, 1):
        t0 = time.perf_counter()
        bullets, stored = facade.process_turn(text, session_id=session_id, turn_id=i)
        ms = (time.perf_counter() - t0) * 1000
        total_ms += ms
        print(f"\n[{i:02}] {title}")
        print(f"Time: {ms:.1f}ms | stored={len(stored)} | bullets={len(bullets)}")
        if stored:
            for s, r, d in stored[:8]:
                print(f"  • ({s}, {r}, {d})")
        else:
            print("  • (no persisted triples)")
    print(f"\nTotal time: {total_ms:.1f}ms")
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
