#!/usr/bin/env python3
"""Test negation propagation from AUX to governing predicate"""

import asyncio
import os
import sys
from loguru import logger

# Add server root to path for imports
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.memory.hotpath_processor import HotPathMemoryProcessor
from pipecat.frames.frames import TranscriptionFrame

async def test_aux_to_verb_negation_propagation():
    """Test that negation propagates from AUX to governing verb"""

    print("\n" + "="*60)
    print("TESTING AUX→VERB NEGATION PROPAGATION")
    print("="*60)

    # Initialize memory processor with fresh state
    processor = HotPathMemoryProcessor(
        user_id="test-negation-prop",
        sqlite_path=":memory:",
        lmdb_dir="/tmp/test_neg_prop_lmdb"
    )

    print("\n✅ Memory processor initialized")

    # Test cases: (text, phrase_that_should_NOT_be_in_any_edge, description)
    test_cases = [
        # Core cases - negation attached to AUX should propagate to main verb
        ("I'm not interested in classic cars", "interested in classic", "Negation on 'am' → 'interested'"),
        ("I don't like horror movies", "like horror", "Negation on 'do' → 'like'"),
        ("He hasn't been eating meat", "eat meat", "AUX chain: hasn't been eating"),
        ("She wasn't happy with the service", "happy", "Negation on 'was' → 'happy'"),
        ("We couldn't find the solution", "find", "Negation on 'could' → 'find'"),
        ("They wouldn't accept the offer", "accept", "Negation on 'would' → 'accept'"),
        
        # Mixed polarity - should store positive but not negative parts
        ("I like pizza but not pineapple", "pineapple", "Mixed positive/negative"),
        ("She enjoys music but not loud music", "loud music", "Mixed with adjectives"),
        
        # Copula cases with negation
        ("It isn't working properly", "working", "Negation on 'is' → 'working'"),
        ("You aren't allowed to enter", "allow", "Negation on 'are' → 'allow'"),
        
        # Complex AUX chains
        ("He hasn't been seen recently", "see", "Perfect passive: hasn't been seen"),
        ("They couldn't have been notified", "notify", "Modal perfect passive"),
    ]

    print("\n📝 Processing negation propagation test cases:")
    for text, forbidden_phrase, description in test_cases:
        frame = TranscriptionFrame(
            text=text,
            user_id="test-negation-prop",
            timestamp="0"
        )

        # Process the frame
        await processor.process_frame(frame, direction=None)
        print(f"  ✓ Processed: '{text}' ({description})")

    # Check what was stored
    print("\n🧠 Verifying negation propagation:")
    edges = processor.hot.store.get_all_edges()
    print(f"  Total edges stored: {len(edges)}")

    # Display all edges for debugging
    if edges:
        print("  Stored edges:")
        for src, rel, dest, weight in edges:
            print(f"    • {src} --[{rel}]--> {dest} (weight: {weight:.3f})")

    # Verify negations were NOT stored
    print("\n✅ Propagation verification tests:")
    failures = []

    for text, forbidden_phrase, description in test_cases:
        # Check if any edge contains the phrase that should NOT be there
        found_forbidden = False
        for src, rel, dest, weight in edges:
            edge_str = f"{src} {rel} {dest}".lower()
            if forbidden_phrase.lower() in edge_str:
                found_forbidden = True
                failures.append(f"❌ FAIL: '{text}' created unwanted edge: {src} --[{rel}]--> {dest}")
                break
        
        if not found_forbidden:
            print(f"  ✓ PASS: '{text}' correctly avoided storing '{forbidden_phrase}'")

    # Test that positive parts are still stored in mixed cases
    print("\n🔍 Testing positive preservation in mixed cases:")
    mixed_cases = [
        ("I like pizza but not pineapple", "like pizza", "Should store positive 'like pizza'"),
        ("She enjoys music but not loud music", "enjoy music", "Should store positive 'enjoy music'"),
    ]
    
    for text, expected_phrase, description in mixed_cases:
        found_expected = False
        for src, rel, dest, weight in edges:
            edge_str = f"{src} {rel} {dest}".lower()
            if expected_phrase.lower() in edge_str:
                found_expected = True
                print(f"  ✓ PASS: '{text}' correctly stored '{expected_phrase}'")
                break
        
        if not found_expected:
            failures.append(f"❌ FAIL: '{text}' missing expected positive edge containing '{expected_phrase}'")

    # Final results
    print("\n" + "="*60)
    if failures:
        print("❌ SOME NEGATION PROPAGATION TESTS FAILED:")
        for failure in failures:
            print(f"  {failure}")
        print("="*60)
        return False
    else:
        print("✅ ALL NEGATION PROPAGATION TESTS PASSED")
        print("  • Negation correctly propagates from AUX to governing predicate")
        print("  • Mixed positive/negative statements handled correctly")
        print("  • Complex AUX chains handled properly")
        print("="*60)
        return True

async def test_positive_control():
    """Test that positive statements are still stored correctly"""
    
    print("\n" + "="*60)
    print("TESTING POSITIVE CONTROL (SHOULD STORE)")
    print("="*60)

    # Initialize memory processor with fresh state
    processor = HotPathMemoryProcessor(
        user_id="test-positive",
        sqlite_path=":memory:",
        lmdb_dir="/tmp/test_positive_lmdb"
    )

    print("\n✅ Memory processor initialized")

    # Positive cases - SHOULD store edges
    test_cases = [
        ("I like science fiction", "like science", "Positive preference"),
        ("She enjoys classical music", "enjoy classical", "Positive activity"),
        ("He works at a tech company", "works_at tech", "Positive employment"),
        ("We live in San Francisco", "lives_in san", "Positive location"),
        ("They have two dogs", "has dog", "Positive possession"),
    ]

    print("\n📝 Processing positive control test cases:")
    for text, expected_phrase, description in test_cases:
        frame = TranscriptionFrame(
            text=text,
            user_id="test-positive",
            timestamp="0"
        )

        # Process the frame
        await processor.process_frame(frame, direction=None)
        print(f"  ✓ Processed: '{text}' ({description})")

    # Check what was stored
    print("\n🧠 Verifying positive storage:")
    edges = processor.hot.store.get_all_edges()
    print(f"  Total edges stored: {len(edges)}")

    # Display all edges for debugging
    if edges:
        print("  Stored edges:")
        for src, rel, dest, weight in edges:
            print(f"    • {src} --[{rel}]--> {dest} (weight: {weight:.3f})")

    # Verify positives WERE stored
    print("\n✅ Positive verification tests:")
    failures = []

    for text, expected_phrase, description in test_cases:
        # Check if any edge contains the expected phrase
        found_expected = False
        for src, rel, dest, weight in edges:
            edge_str = f"{src} {rel} {dest}".lower()
            if expected_phrase.lower() in edge_str:
                found_expected = True
                print(f"  ✓ PASS: '{text}' correctly stored '{expected_phrase}'")
                break
        
        if not found_expected:
            failures.append(f"❌ FAIL: '{text}' missing expected edge containing '{expected_phrase}'")

    # Final results
    print("\n" + "="*60)
    if failures:
        print("❌ SOME POSITIVE CONTROL TESTS FAILED:")
        for failure in failures:
            print(f"  {failure}")
        print("="*60)
        return False
    else:
        print("✅ ALL POSITIVE CONTROL TESTS PASSED")
        print("  • Positive statements are stored correctly")
        print("  • No regression in normal fact extraction")
        print("="*60)
        return True

if __name__ == "__main__":
    # Run both test suites
    propagation_success = asyncio.run(test_aux_to_verb_negation_propagation())
    positive_success = asyncio.run(test_positive_control())

    success = propagation_success and positive_success
    sys.exit(0 if success else 1)
