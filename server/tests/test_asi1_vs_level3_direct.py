#!/usr/bin/env python3
"""
Direct comparison of ASI1 and Level3 extraction systems
"""
import os
import time
from asi1_processor import ULTRAGROKSpacyV821Processor
from level3_universal_kg import UniversalKGExtractor

def test_comparison():
    # Test text
    text = "John Smith works at Google in San Francisco. He manages the AI team and develops new products."

    print("="*60)
    print("ASI1 vs Level3 Direct Comparison")
    print("="*60)
    print(f"Test text: {text}\n")

    # Test ASI1
    print("1. ASI1 Extraction (ULTRAGROKSpacyV821Processor)")
    print("-"*40)
    try:
        asi1 = ULTRAGROKSpacyV821Processor()
        start = time.time()
        asi1_result = asi1.process_spacy_semantics(text)
        asi1_time = (time.time() - start) * 1000

        print(f"Time: {asi1_time:.2f}ms")
        print(f"Triples found: {len(asi1_result.get('triples', []))}")
        if asi1_result.get('triples'):
            for t in asi1_result['triples'][:3]:
                print(f"  - {t.subj} | {t.pred} | {t.obj}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n2. Level3 Extraction (UniversalKGExtractor)")
    print("-"*40)
    try:
        level3 = UniversalKGExtractor()
        start = time.time()
        level3_result = level3.extract(text)
        level3_time = (time.time() - start) * 1000

        print(f"Time: {level3_time:.2f}ms")
        print(f"Relations found: {len(level3_result.relations)}")
        if level3_result.relations:
            for r in level3_result.relations[:3]:
                print(f"  - {r.subject} | {r.predicate} | {r.object}")
    except Exception as e:
        print(f"Error: {e}")

    # Check if they share patterns
    print("\n3. Pattern Analysis")
    print("-"*40)

    # Check ASI1 for Level3 patterns
    print("ASI1 uses YAML-based patterns with spaCy dependency matching")
    print("Level3 uses direct spaCy dependency parsing with ASI1 guards embedded")

    # Key finding
    print("\n🔍 KEY FINDING:")
    print("-"*40)
    print("Level3 has ASI1 guards hardcoded into its extraction logic:")
    print("  - Line 189: '# ASI1 meaningful_attribute filtering'")
    print("  - Line 194: '# ASI1 guard: meaningful_attribute = true'")
    print("  - Line 204: '# ASI1 guard: avoid adjective that's same as noun stem'")
    print("  - Line 246: '# ASI1 guard: avoid_over_segmentation = true'")
    print("\nLevel3 appears to be a reimplementation of ASI1 concepts")
    print("with the guards/filters directly embedded in Python code")
    print("rather than loaded from YAML patterns.")

    print("\n4. Architecture Difference")
    print("-"*40)
    print("ASI1: YAML patterns → spaCy matching → triples")
    print("Level3: spaCy parsing → hardcoded ASI1 filters → relations")
    print("\nBoth use the same conceptual filtering (ASI1 guards)")
    print("but Level3 implements them directly in code for speed.")

if __name__ == "__main__":
    # Disable debug output
    os.environ['ASI_DEBUG'] = 'false'
    test_comparison()