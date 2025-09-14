#!/usr/bin/env python3
"""
Clean comparison of ASI1 and Level3 extraction - NO DEBUG OUTPUT
"""
import os
import sys
import time

# Disable all debug output
os.environ['ASI_DEBUG'] = 'false'
os.environ['DEBUG'] = 'false'

# Redirect stderr to suppress debug messages
class SuppressDebug:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._original_stderr

def test_comparison():
    from asi1_processor import ULTRAGROKSpacyV821Processor
    from level3_universal_kg import UniversalKGExtractor

    # Test text
    text = "John Smith works at Google in San Francisco. He manages the AI team and develops new products."

    print("="*60)
    print("ASI1 vs Level3 CLEAN Comparison")
    print("="*60)
    print(f"Test text: {text}\n")

    # Test ASI1
    print("1. ASI1 Extraction (ULTRAGROKSpacyV821Processor)")
    print("-"*40)
    try:
        with SuppressDebug():
            asi1 = ULTRAGROKSpacyV821Processor()

        start = time.time()
        with SuppressDebug():
            asi1_result = asi1.process_spacy_semantics(text)
        asi1_time = (time.time() - start) * 1000

        print(f"Time: {asi1_time:.2f}ms")
        print(f"Triples found: {len(asi1_result.get('triples', []))}")

        if asi1_result.get('triples'):
            print("\nExtracted triples:")
            for t in asi1_result['triples']:
                confidence = getattr(t, 'confidence', 1.0)
                print(f"  [{confidence:.2f}] {t.subj} | {t.pred} | {t.obj}")
        else:
            print("No triples extracted")

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
            print("\nExtracted relations:")
            for r in level3_result.relations[:10]:  # Show first 10
                print(f"  [{r.confidence:.2f}] {r.subject} | {r.predicate} | {r.object}")
        else:
            print("No relations extracted")

    except Exception as e:
        print(f"Error: {e}")

    # Compare outputs
    print("\n3. QUALITY COMPARISON")
    print("-"*40)

    asi1_count = len(asi1_result.get('triples', []))
    level3_count = len(level3_result.relations)

    print(f"ASI1 extracted: {asi1_count} triples")
    print(f"Level3 extracted: {level3_count} relations")

    if asi1_count == 0 and level3_count > 0:
        print("\n⚠️ MAJOR DIFFERENCE: ASI1 extracts nothing, Level3 extracts many!")
        print("They are NOT producing the same output.")
    elif asi1_count > 0 and level3_count > 0:
        print("\nBoth extracted relations. Checking similarity...")
        # Compare predicates used
        asi1_preds = set(t.pred for t in asi1_result.get('triples', []))
        level3_preds = set(r.predicate for r in level3_result.relations)
        print(f"ASI1 predicates: {asi1_preds}")
        print(f"Level3 predicates: {list(level3_preds)[:10]}")  # First 10

if __name__ == "__main__":
    test_comparison()
