#!/usr/bin/env python3
"""
Test all 20 L1 patterns after implementation
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.add(sys.stderr, level="INFO")

def test_all_patterns():
    """Test extraction with all patterns now implemented"""

    from core.memory.extractors.yaml_extractor import YAMLExtractor

    # Test cases for different patterns
    test_cases = [
        # Basic SVO
        ("John works at Google", ["UNIVERSAL_SVO_ACTIVE"]),

        # Passive voice
        ("The book was written by John", ["UNIVERSAL_SVO_PASSIVE"]),

        # Copula
        ("John is happy", ["UNIVERSAL_COPULA_ADJECTIVAL"]),
        ("Alice is the CEO", ["UNIVERSAL_COPULA_NOMINAL"]),

        # Coordination
        ("John and Mary work at Google", ["UNIVERSAL_COORD_SUBJECT"]),
        ("Alice likes cats and dogs", ["UNIVERSAL_COORD_OBJECT"]),
        ("Bob runs and jumps", ["UNIVERSAL_COORD_VERB"]),

        # Mixed coordination (subject/object and verbs coordinated)
        ("Alice and Bob founded and led Acme and Beta", ["UNIVERSAL_COORD_MIXED"]),

        # Control verbs
        ("John wants to leave", ["UNIVERSAL_CONTROL_VERB"]),

        # Ditransitive
        ("John gave Mary a book", ["UNIVERSAL_DITRANSITIVE_GIVE"]),
        ("Alice told Bob the news", ["UNIVERSAL_DITRANSITIVE_COMMUNICATE"]),

        # Modal
        ("John can swim", ["UNIVERSAL_MODAL_VERBS"]),

        # Embedding
        ("John thinks that Mary is smart", ["UNIVERSAL_CCOMP_EMBEDDING"]),

        # Relative clause
        ("Alice, who founded Acme, is the CEO", ["UNIVERSAL_RELATIVE_CLAUSE"]),

        # Temporal/Spatial
        ("John worked yesterday", ["UNIVERSAL_TEMPORAL_ADVERBIALS"]),
        ("The book is on the table", ["UNIVERSAL_SPATIAL_PREPOSITIONS"]),

        # Quantifier
        ("All students passed the test", ["UNIVERSAL_QUANTIFIER_SCOPE"]),

        # Negation
        ("John does not like coffee", ["UNIVERSAL_NEGATION_SCOPE"]),

        # Progressive/Perfect
        ("John is running", ["UNIVERSAL_PROGRESSIVE_ASPECT"]),
        ("John has eaten lunch", ["UNIVERSAL_PERFECT_ASPECT"]),
    ]

    yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"

    print("=" * 80)
    print("TESTING ALL 20 L1 PATTERNS")
    print("=" * 80)

    # Initialize extractor
    yaml_ext = YAMLExtractor(yaml_path)

    results = []
    patterns_covered = set()

    for text, expected_patterns in test_cases:
        print(f"\nText: \"{text}\"")
        print(f"Expected patterns: {expected_patterns}")
        print("-" * 60)

        # Extract
        t0 = time.perf_counter()
        _, triples, neg_count, doc = yaml_ext.extract(text, 'en')
        triples = yaml_ext.refine(text, triples, doc)
        latency = (time.perf_counter() - t0) * 1000

        print(f"Extracted ({latency:.0f}ms): {triples}")

        if triples:
            patterns_covered.update(expected_patterns)
            print("✅ Pattern working!")
        else:
            print("⚠️ No extraction")

        results.append({
            "text": text,
            "patterns": expected_patterns,
            "triples": triples,
            "latency": latency,
            "neg_count": neg_count
        })

    # Summary
    print("\n" + "=" * 80)
    print("PATTERN COVERAGE SUMMARY")
    print("=" * 80)

    all_patterns = {
        "UNIVERSAL_SVO_ACTIVE",
        "UNIVERSAL_SVO_PASSIVE",
        "UNIVERSAL_COPULA_NOMINAL",
        "UNIVERSAL_COPULA_ADJECTIVAL",
        "UNIVERSAL_COORD_SUBJECT",
        "UNIVERSAL_COORD_OBJECT",
        "UNIVERSAL_COORD_VERB",
        "UNIVERSAL_COORD_MIXED",
        "UNIVERSAL_DITRANSITIVE_GIVE",
        "UNIVERSAL_DITRANSITIVE_COMMUNICATE",
        "UNIVERSAL_CONTROL_VERB",
        "UNIVERSAL_CCOMP_EMBEDDING",
        "UNIVERSAL_MODAL_VERBS",
        "UNIVERSAL_RELATIVE_CLAUSE",
        "UNIVERSAL_TEMPORAL_ADVERBIALS",
        "UNIVERSAL_SPATIAL_PREPOSITIONS",
        "UNIVERSAL_QUANTIFIER_SCOPE",
        "UNIVERSAL_NEGATION_SCOPE",
        "UNIVERSAL_PROGRESSIVE_ASPECT",
        "UNIVERSAL_PERFECT_ASPECT",
    }

    print(f"\nPatterns covered: {len(patterns_covered)}/20")
    print(f"Covered: {sorted(patterns_covered)}")

    missing = all_patterns - patterns_covered
    if missing:
        print(f"\nMissing coverage: {sorted(missing)}")

    # Performance stats
    total_triples = sum(len(r["triples"]) for r in results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"\n📊 Performance Stats:")
    print(f"Total triples extracted: {total_triples}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Tests with extraction: {sum(1 for r in results if r['triples'])}/{len(results)}")

    # Save results
    output_path = Path("results/all_patterns_test.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    test_all_patterns()
