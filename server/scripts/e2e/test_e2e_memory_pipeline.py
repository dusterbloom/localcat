#!/usr/bin/env python3
"""
End-to-end memory pipeline test (no audio):
- Intent classification
- Extraction via registry (Enhanced Level3 default)
- Query routing and storage gating
- Augmented retrieval and context bullet building

Usage:
  source server/.venv/bin/activate
  python scripts/e2e/test_e2e_memory_pipeline.py --preset trf

Options:
  --preset sm|trf    Use small spaCy or transformer preset (default: sm)
  --json             Print machine-readable JSON summary
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from components.memory.memory_store import MemoryStore, Paths  # type: ignore
from components.memory.hotmemory_facade import HotMemoryFacade  # type: ignore
from components.memory.memory_intent import get_intent_classifier  # type: ignore


def preset_env(preset: str) -> None:
    """Apply convenient env presets for Enhanced Level3."""
    os.environ.setdefault('DEFAULT_EXTRACTION_STRATEGY', 'enhanced_level3')
    os.environ.setdefault('HOTMEM_ROUTE_TO_REGISTRY', 'true')
    if preset == 'trf':
        os.environ['ENHANCED_LEVEL3_SPACY_MODEL'] = 'en_core_web_rtf'
        os.environ.setdefault('ENHANCED_LEVEL3_ENTITY_CONF', '0.65')
        os.environ.setdefault('ENHANCED_LEVEL3_RELATION_CONF', '0.55')
        os.environ.setdefault('ENHANCED_LEVEL3_TARGET_REL', '40')
        os.environ.setdefault('ENHANCED_LEVEL3_EXTRA_VERBS', 'argue,weigh,consider,foster,prove,seek,reveal,mitigate,preserve')
    else:
        os.environ.setdefault('ENHANCED_LEVEL3_SPACY_MODEL', 'en_core_web_sm')
        os.environ.setdefault('ENHANCED_LEVEL3_ENTITY_CONF', '0.70')
        os.environ.setdefault('ENHANCED_LEVEL3_RELATION_CONF', '0.65')


def run_turn(facade: HotMemoryFacade, text: str, session: str, turn_id: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bullets, stored = facade.process_turn(text, session, turn_id)
    dt = (time.perf_counter() - t0) * 1000
    return {
        'text': text,
        'time_ms': round(dt, 1),
        'bullets': bullets,
        'stored_triples': stored,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=['sm', 'trf'], default='sm')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    # Load .env if present
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    # Apply presets
    preset_env(args.preset)

    # Fresh test store under /tmp
    paths = Paths(
        sqlite_path=os.path.abspath(os.path.join('/tmp', 'hotmem_e2e_memory.db')),
        lmdb_dir=os.path.abspath(os.path.join('/tmp', 'hotmem_e2e_graph.lmdb')),
    )
    store = MemoryStore(paths=paths)
    facade = HotMemoryFacade(store)

    # Test texts (declarative + question)
    texts = [
        "John Smith works at Google in San Francisco. He manages the AI team and develops new products.",
        "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees.",
        "Where do I live again?",  # pure question → retrieval only
        "My name is Alex and I live in Berlin.",
    ]

    # Run turns
    session_id = 'e2e_session_1'
    results: List[Dict[str, Any]] = []
    for i, t in enumerate(texts, 1):
        results.append(run_turn(facade, t, session_id, i))

    # Print summary
    if args.json:
        print(json.dumps({'preset': args.preset, 'results': results}, indent=2))
        return 0

    print(f"Preset: {args.preset}")
    for r in results:
        print("\n=== TURN ===")
        print(f"Text: {r['text']}")
        print(f"Time: {r['time_ms']}ms")
        print(f"Stored triples: {len(r['stored_triples'])}")
        for s, p, o in r['stored_triples'][:6]:
            print(f"  • ({s}, {p}, {o})")
        print(f"Bullets: {len(r['bullets'])}")
        for b in r['bullets'][:6]:
            print(f"  - {b}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

