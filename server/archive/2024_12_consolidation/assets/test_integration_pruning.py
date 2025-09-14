#!/usr/bin/env python3
"""
Integration test: entity pruning and basic triple sanity

Validates that the extraction pipeline prunes overlapping/numeric entities
and avoids storing low-value micro triples (e.g., single-digit tails).
"""

import os
import sys
import os.path as op

# Ensure 'server' is on sys.path when running from repo root
_here = op.dirname(op.abspath(__file__))
_server_root = op.dirname(_here)
if _server_root not in sys.path:
    sys.path.insert(0, _server_root)

from components.memory.config import create_config
from components.extraction.memory_extractor import MemoryExtractor


def is_numeric_like(s: str) -> bool:
    s2 = (s or '').strip().replace(',', '').replace('.', '')
    return s2.isdigit()


def test_entity_pruning_and_triple_sanity():
    # Ensure GLiREL can be optionally disabled if not available; we test pruning regardless
    os.environ.setdefault('HOTMEM_USE_GLIREL', 'true')

    text = (
        "The iPhone 15 Pro, manufactured by Apple Inc. in Cupertino, California, "
        "costs 999 dollars and competes with Samsung Galaxy S24 Ultra produced in South Korea by Samsung Electronics."
    )

    config = create_config()
    extractor = MemoryExtractor(config.get_extractor_config())

    result = extractor.extract(text)

    # Entities: ensure '15' alone is not present if 'iPhone 15 Pro' exists
    ents = set([e.strip().lower() for e in result.entities])
    assert any('iphone 15 pro' in e for e in ents), "Expected composite product entity"
    assert '15' not in ents, "Numeric-only subspan should be pruned"

    # Triples sanity: no head/tail that are numeric-only or very short tokens
    bad = []
    for (s, r, d) in result.triples:
        if is_numeric_like(s) or is_numeric_like(d):
            bad.append((s, r, d))
        if len((s or '').strip()) < 2 or len((d or '').strip()) < 2:
            bad.append((s, r, d))

    assert not bad, f"Found low-value triples: {bad[:5]}"

    # Expect at least some structured relations from UD/GLiREL
    assert len(result.triples) >= 5, "Should extract a minimal set of relations"

if __name__ == "__main__":
    try:
        test_entity_pruning_and_triple_sanity()
        print("✅ Integration pruning test passed")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        raise
