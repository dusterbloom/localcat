#!/usr/bin/env python3
"""
A/B test: Coref ON vs OFF for Enhanced Level3 end-to-end pipeline (no audio).

Measures per-turn runtime, stored triple counts, and bullet counts.

Usage:
  source .venv/bin/activate && cd server
  python scripts/e2e/ab_coref.py --preset trf
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade


TEXTS = [
    "John Smith works at Google in San Francisco. He manages the AI team and develops new products.",
    "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees.",
    "Where do I live again?",
    "My name is Alex and I live in Berlin.",
]


def preset_env(preset: str) -> None:
    os.environ.setdefault('DEFAULT_EXTRACTION_STRATEGY', 'enhanced_level3')
    os.environ.setdefault('HOTMEM_ROUTE_TO_REGISTRY', 'true')
    if preset == 'trf':
        os.environ['ENHANCED_LEVEL3_SPACY_MODEL'] = 'en_core_web_trf'
        os.environ.setdefault('ENHANCED_LEVEL3_ENTITY_CONF', '0.65')
        os.environ.setdefault('ENHANCED_LEVEL3_RELATION_CONF', '0.55')
        os.environ.setdefault('ENHANCED_LEVEL3_TARGET_REL', '40')
    else:
        os.environ.setdefault('ENHANCED_LEVEL3_SPACY_MODEL', 'en_core_web_sm')


def run_variant(preset: str, use_coref: bool) -> Dict[str, Any]:
    # Apply preset, toggle coref via env
    preset_env(preset)
    os.environ['HOTMEM_USE_COREF'] = 'true' if use_coref else 'false'

    # Fresh temp store per variant
    suffix = 'coref' if use_coref else 'nocoref'
    paths = Paths(
        sqlite_path=os.path.abspath(os.path.join('/tmp', f'hotmem_e2e_{suffix}.db')),
        lmdb_dir=os.path.abspath(os.path.join('/tmp', f'hotmem_e2e_{suffix}.lmdb')),
    )
    store = MemoryStore(paths=paths)
    facade = HotMemoryFacade(store)

    results: List[Dict[str, Any]] = []
    session_id = f'e2e_ab_{suffix}'
    t_all = time.perf_counter()
    for i, t in enumerate(TEXTS, 1):
        t0 = time.perf_counter()
        bullets, stored = facade.process_turn(t, session_id, i)
        ms = (time.perf_counter() - t0) * 1000
        results.append({
            'text': t,
            'time_ms': round(ms, 1),
            'stored': stored,
            'bullets': bullets,
        })
    total_ms = (time.perf_counter() - t_all) * 1000
    return {
        'use_coref': use_coref,
        'preset': preset,
        'total_ms': round(total_ms, 1),
        'results': results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=['sm', 'trf'], default='trf')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    A = run_variant(args.preset, use_coref=False)
    B = run_variant(args.preset, use_coref=True)

    if args.json:
        print(json.dumps({'A_nocoref': A, 'B_coref': B}, indent=2))
        return 0

    def summarize(run: Dict[str, Any]) -> None:
        print(f"\nVariant: {'coref=ON' if run['use_coref'] else 'coref=OFF'} | preset={run['preset']}")
        print(f"Total time: {run['total_ms']:.1f}ms")
        for r in run['results']:
            print(f"  - {r['time_ms']:.1f}ms | stored={len(r['stored'])} | bullets={len(r['bullets'])}")

    summarize(A)
    summarize(B)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

