#!/usr/bin/env python3

import time
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from components.extraction.yaml_ud_loader_v2 import YAMLUDLoaderV2
import spacy

def test_v8_2_1_spacy():
    """Test the ULTRAGROK V8.2.1 SPACY patterns with our V2 loader"""

    print("=== ULTRAGROK V8.2.1 SPACY COMPATIBILITY TEST ===")
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading YAML patterns with V2 loader...")
    yaml_path = "/Users/peppi/Dev/localcat/server/ULTRAGROK_V8.2.1_SPACY.yaml"
    loader = YAMLUDLoaderV2(yaml_path)
    patterns_info = loader.get_patterns_info()

    print(f"✓ Loaded {patterns_info['total_patterns']} patterns")
    print(f"Pattern priorities: {patterns_info['pattern_priorities']}")
    print()

    # Test sentences from easy to complex
    test_sentences = [
        # V8.2.1 Expected Examples
        "John gave Mary book",           # Expected: 2 triples (core_event + transfer_event)
        "Book was read by John",         # Expected: 1 triple (passive_event)
        "John walked to store",          # Expected: 1 triple (goal_motion)
        "Book is on table",              # Expected: 1 triple (static_location)
        "John worked yesterday",         # Expected: 1 triple (temporal_adverb)
        "John is the president",         # Expected: 1 triple (nominal_attribution)

        # Complex examples for natural complexity scaling
        "John walked to store through park yesterday", # Expected: 3 triples
        "Solution seems very effective",               # Expected: 2 triples
        "Meeting is in conference room",               # Expected: 1 triple
    ]

    total_time = 0
    total_triples = 0

    print("Testing natural complexity scaling (V8.2.1 promise)...")
    print("-" * 60)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i:2d}. '{sentence}'")

        start_time = time.time()
        doc = nlp(sentence)
        triples = loader.extract_triples(doc)
        end_time = time.time()

        extraction_time = (end_time - start_time) * 1000
        total_time += extraction_time
        total_triples += len(triples)

        print(f"    Time: {extraction_time:.2f}ms | Triples: {len(triples)}")

        if triples:
            for j, triple in enumerate(triples, 1):
                confidence = getattr(triple, 'confidence', 'N/A')
                triple_type = getattr(triple, 'type', 'unknown')
                print(f"      {j}. ({triple.subject}, {triple.predicate}, {triple.object}) "
                      f"[{triple_type}, conf={confidence}]")
        else:
            print("      ⚠️  NO TRIPLES EXTRACTED")
        print()

    avg_time = total_time / len(test_sentences)
    avg_triples = total_triples / len(test_sentences)

    print("=== V8.2.1 SPACY PERFORMANCE SUMMARY ===")
    print(f"Total sentences: {len(test_sentences)}")
    print(f"Total triples: {total_triples}")
    print(f"Average triples per sentence: {avg_triples:.2f}")
    print(f"Total extraction time: {total_time:.2f}ms")
    print(f"Average time per sentence: {avg_time:.2f}ms")

    # V8.2.1 Quality Assessment
    print("\n=== QUALITY ASSESSMENT ===")

    if total_triples == 0:
        print("❌ FAILED: Zero extraction - patterns not matching spaCy dependencies")
        return False

    if avg_time > 500:
        print(f"⚠️  WARNING: Average extraction time {avg_time:.2f}ms > 500ms target")
    else:
        print(f"✓ PASSED: Extraction speed {avg_time:.2f}ms < 500ms target")

    # Natural complexity scaling check
    simple_sentences = test_sentences[:3]  # First 3 are simple
    complex_sentences = test_sentences[6:]  # Last 3 are complex

    simple_triples = 0
    complex_triples = 0

    for sentence in simple_sentences:
        doc = nlp(sentence)
        triples = loader.extract_triples(doc)
        simple_triples += len(triples)

    for sentence in complex_sentences:
        doc = nlp(sentence)
        triples = loader.extract_triples(doc)
        complex_triples += len(triples)

    avg_simple = simple_triples / len(simple_sentences) if simple_sentences else 0
    avg_complex = complex_triples / len(complex_sentences) if complex_sentences else 0

    print(f"Simple sentences (avg): {avg_simple:.1f} triples")
    print(f"Complex sentences (avg): {avg_complex:.1f} triples")

    if avg_complex > avg_simple:
        print("✓ PASSED: Natural complexity scaling - complex > simple")
    else:
        print("⚠️  WARNING: Complex sentences should extract more triples than simple")

    print(f"\nV8.2.1 SPACY Compatibility: {'✓ SUCCESS' if total_triples > 0 else '❌ FAILED'}")
    return total_triples > 0

if __name__ == "__main__":
    success = test_v8_2_1_spacy()
    sys.exit(0 if success else 1)