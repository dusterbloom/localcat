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

def test_copula_extraction():
    """Test copula relation extraction (e.g., 'is', 'are', 'was')"""
    t = "My wife's name is Sarah. She is a software engineer at Google since 2020."
    triples, ms = _triples(t)
    
    # Check for copula relations
    copula_rels = [(s, p, o) for s, p, o in triples if p == 'is']
    assert len(copula_rels) >= 2, f"Expected at least 2 copula relations, got {len(copula_rels)}: {copula_rels}"
    
    # Specific checks
    name_rel = any('wife' in s.lower() and 'name' in o.lower() for s, p, o in copula_rels)
    assert name_rel, f"No 'wife name is Sarah' relation found in: {copula_rels}"
    
    engineer_rel = any('she' in s.lower() and 'engineer' in o.lower() for s, p, o in copula_rels)
    assert engineer_rel, f"No 'she is engineer' relation found in: {copula_rels}"
    
    assert ms < 100.0, f"Extraction too slow: {ms:.1f}ms"

def test_temporal_copula_integration():
    """Test copula + temporal extraction (e.g., 'since 2020')"""
    t = "My wife is at Google since 2020. She works there as a manager."
    triples, ms = _triples(t)
    
    # Check for copula with temporal
    all_rels = [(s, p, o) for s, p, o in triples]
    copula_rels = [r for r in all_rels if p == 'is']
    temporal_rels = [r for r in all_rels if any(word in o.lower() for word in ['since', '2020'])]
    
    assert len(copula_rels) >= 1, f"No copula relations found: {copula_rels}"
    assert len(temporal_rels) >= 1, f"No temporal relations found: {temporal_rels}"
    
    # Ensure wife-Google relation exists
    wife_google = any('wife' in s.lower() and 'google' in o.lower() for s, p, o in all_rels)
    assert wife_google, f"No 'wife is at Google' relation in: {all_rels}"
    
    # Temporal should link to employment duration
    temporal_link = any('since' in o.lower() or '2020' in o.lower() for s, p, o in all_rels if 'wife' in s.lower())
    assert temporal_link, f"No temporal link for wife in: {all_rels}"
    
    assert ms < 100.0, f"Temporal copula extraction too slow: {ms:.1f}ms"


def test_live_in_core_fact():
    t = "I live in Berlin."
    triples, _ = _triples(t)
    # We accept either 'I' or 'you' depending on higher layers; relation must be live_in
    assert any(p == 'live_in' for _, p, _ in triples)
    assert any(o.lower() == 'berlin' for _, _, o in triples)

