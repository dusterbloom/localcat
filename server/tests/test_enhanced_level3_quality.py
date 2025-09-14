#!/usr/bin/env python3
"""Regression tests for Enhanced Level3 quality and speed characteristics.

These tests assert that Enhanced Level3 returns clean semantic relations
without generic UD artifacts and preserves key verb_prep nuances.
"""

import os
import time

from components.extraction.extraction_strategies import EnhancedLevel3ExtractionStrategy


def setup_module(module):
    # Small model for fast tests
    os.environ['ENHANCED_LEVEL3_SPACY_MODEL'] = 'en_core_web_sm'
    os.environ['ENHANCED_LEVEL3_ENTITY_CONF'] = '0.65'
    os.environ['ENHANCED_LEVEL3_RELATION_CONF'] = '0.55'


def _triples(text: str):
    strat = EnhancedLevel3ExtractionStrategy()
    assert strat.is_available(), "Enhanced Level3 not available"
    t0 = time.perf_counter()
    triples = strat.extract(text)
    ms = (time.perf_counter() - t0) * 1000
    return triples, ms


def test_work_at_and_play_scene():
    t1 = "John Smith works at Google in San Francisco. He manages the AI team and develops new products."
    triples, ms = _triples(t1)
    preds = {p for (_, p, _) in triples}
    assert 'work_at' in preds
    assert 'work_in' in preds
    assert 'subject_of' not in preds and 'prepositional_object_of' not in preds
    assert ms < 50.0  # small model should be very fast

    t2 = (
        "In the bustling city park, a group of children played tag while their parents "
        "watched from wooden benches under tall oak trees."
    )
    triples2, _ = _triples(t2)
    sro = set(triples2)
    assert ('a group children', 'play', 'tag') in {(s.replace('of ', ''), p, o) for (s,p,o) in sro} or any(p=='play' for _,p,_ in sro)
    # Ensure verb_prep nuances present
    assert any(p == 'play_in' for _, p, _ in sro)
    assert any(p == 'watch_from' for _, p, _ in sro)
    assert any(p == 'watch_under' for _, p, _ in sro)


def test_live_in_core_fact():
    t = "I live in Berlin."
    triples, _ = _triples(t)
    # We accept either 'I' or 'you' depending on higher layers; relation must be live_in
    assert any(p == 'live_in' for _, p, _ in triples)
    assert any(o.lower() == 'berlin' for _, _, o in triples)

