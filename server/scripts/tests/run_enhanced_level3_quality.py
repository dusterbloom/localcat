#!/usr/bin/env python3
"""
Runner for Enhanced Level3 quality checks with logs and timing.

Shows evidence of triples, predicates, and timing per case.
"""

import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from components.extraction.extraction_strategies import EnhancedLevel3ExtractionStrategy


def run_case(text: str, name: str):
    os.environ['ENHANCED_LEVEL3_SPACY_MODEL'] = 'en_core_web_sm'
    os.environ['ENHANCED_LEVEL3_ENTITY_CONF'] = '0.65'
    os.environ['ENHANCED_LEVEL3_RELATION_CONF'] = '0.55'
    strat = EnhancedLevel3ExtractionStrategy()
    assert strat.is_available(), 'Enhanced Level3 strategy unavailable'
    t0 = time.perf_counter()
    triples = strat.extract(text)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n=== {name} ===")
    print(f"Time: {ms:.2f}ms  Triples: {len(triples)}")
    for s, r, d in triples:
        print(f"  - ({s} | {r} | {d})")
    return triples, ms


def main():
    ok = True

    # Case 1: work_at + manage
    t1 = "John Smith works at Google in San Francisco. He manages the AI team and develops new products."
    triples1, ms1 = run_case(t1, 'Case 1: work_at + manage')
    preds1 = {p for (_, p, _) in triples1}
    if not (('work_at' in preds1) and ('work_in' in preds1)):
        print("[FAIL] Missing work_at/work_in in Case 1")
        ok = False
    if any(p in preds1 for p in ('subject_of', 'prepositional_object_of', 'preposition:in')):
        print("[FAIL] Found generic UD predicates in Case 1")
        ok = False

    # Case 2: play + watch_from/under
    t2 = (
        "In the bustling city park, a group of children played tag while their parents "
        "watched from wooden benches under tall oak trees."
    )
    triples2, ms2 = run_case(t2, 'Case 2: park scene')
    preds2 = {p for (_, p, _) in triples2}
    if not ({'play', 'play_in', 'watch_from', 'watch_under'} <= preds2):
        print("[FAIL] Missing expected verb/verb_prep in Case 2")
        ok = False

    # Case 3: live_in
    t3 = "I live in Berlin."
    triples3, ms3 = run_case(t3, 'Case 3: live_in')
    if not any(p == 'live_in' and d.lower() == 'berlin' for _, p, d in triples3):
        print("[FAIL] Missing live_in Berlin in Case 3")
        ok = False

    print("\n=== Summary ===")
    print(f"Case 1: {ms1:.2f}ms, predicates={list(preds1)}")
    print(f"Case 2: {ms2:.2f}ms, predicates={list(preds2)}")
    print(f"Case 3: {ms3:.2f}ms, triples={len(triples3)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

