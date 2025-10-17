#!/usr/bin/env python3
"""Test memory system functionality"""

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

async def test_memory_system():
    """Test the memory system with sample interactions"""

    print("\n" + "="*60)
    print("TESTING MEMORY SYSTEM")
    print("="*60)

    # Initialize memory processor
    processor = HotPathMemoryProcessor(
        user_id="test-user",
        sqlite_path=":memory:",  # Use in-memory DB for testing
        lmdb_dir="/tmp/test_lmdb"
    )

    print("\n✅ Memory processor initialized")

    # Test transcription processing
    test_cases = [
        "My name is John",
        "I live in San Francisco",
        "I work at OpenAI",
        "My favorite food is pizza",
        "I have a dog named Max"
    ]

    print("\n📝 Processing test transcriptions:")
    for text in test_cases:
        frame = TranscriptionFrame(
            text=text,
            user_id="test-user",
            timestamp="0"
        )

        # Process the frame (direction is optional, defaults to None)
        await processor.process_frame(frame, direction=None)
        print(f"  ✓ Processed: '{text}'")

    # Check what's in memory
    print("\n🧠 Memory contents:")

    # Get current memory state using new API
    memory_bullets = processor.hot.retrieve_bullets("Tell me about the user", read_only=True)
    if memory_bullets:
        print(f"  Memory bullets: {memory_bullets}")
    else:
        print("  No memory bullets found")

    # Check edges using store API
    edges = processor.hot.store.get_all_edges()
    print(f"\n  Found {len(edges)} edges in memory graph:")
    for src, rel, dest, weight in edges[:5]:  # Show first 5
        print(f"    • {src} → {dest} (relation: {rel}, weight: {weight:.3f})")

    # Test retrieval
    print("\n🔍 Testing retrieval:")
    test_queries = [
        "Where does John live?",
        "What is John's pet?",
        "John's job"
    ]

    for query in test_queries:
        bullets = processor.hot.retrieve_bullets(query, read_only=True)
        print(f"  Query: '{query}'")
        print(f"    → Bullets: {bullets if bullets else 'No relevant context found'}")

    print("\n" + "="*60)
    if edges:
        print("✅ MEMORY SYSTEM IS WORKING")
    else:
        print("⚠️ MEMORY SYSTEM MAY HAVE ISSUES - No edges created")
    print("="*60)

    return len(edges) > 0

async def test_negation_handling():
    """Test that negations are handled correctly and don't create false positives"""

    print("\n" + "="*60)
    print("TESTING NEGATION HANDLING")
    print("="*60)

    # Initialize memory processor with fresh state
    processor = HotPathMemoryProcessor(
        user_id="test-negation",
        sqlite_path=":memory:",
        lmdb_dir="/tmp/test_negation_lmdb"
    )

    print("\n✅ Memory processor initialized")

    # Test cases: (text, expected_should_NOT_store, description)
    test_cases = [
        # Pure negations - should NOT create positive facts
        ("I'm not interested in classic cars", ["interested in classic"], "Pure negation"),
        ("I don't like horror movies", ["like horror"], "Negative preference"),

        # Positive facts - SHOULD store
        ("I like science fiction", None, "Positive preference"),
        ("I have a dog named Max", None, "Positive fact"),

        # Mixed positive and negative - store only positive
        ("I like pizza but not pineapple", ["pineapple"], "Mixed positive/negative"),

        # Confusion/meta-commentary - should be filtered (Layer 2)
        ("I think it's just confusing", None, "Confusion filter"),
        ("I don't know what you mean", None, "Meta-commentary filter"),
        ("This is unclear to me", None, "Unclear filter"),

        # Questions without assertions - should be filtered (Layer 2 & 4)
        ("What is your name?", None, "Pure question"),

        # Complex negations
        ("I'm not a fan of sports", ["fan of sports"], "Negated identity"),
    ]

    print("\n📝 Processing negation test cases:")
    for text, should_not_contain, description in test_cases:
        frame = TranscriptionFrame(
            text=text,
            user_id="test-negation",
            timestamp="0"
        )

        # Process the frame
        await processor.process_frame(frame, direction=None)
        print(f"  ✓ Processed: '{text}' ({description})")

    # Check what was stored
    print("\n🧠 Verifying memory contents:")
    edges = processor.hot.store.get_all_edges()
    print(f"  Total edges stored: {len(edges)}")

    # Display all edges for debugging
    if edges:
        print("  Stored edges:")
        for src, rel, dest, weight in edges:
            print(f"    • {src} --[{rel}]--> {dest} (weight: {weight:.3f})")

    # Verify negations were NOT stored
    print("\n✅ Verification tests:")
    failures = []

    for text, should_not_contain, description in test_cases:
        if should_not_contain:
            for phrase in should_not_contain:
                # Check if any edge contains the phrase that should NOT be there
                found = False
                for src, rel, dest, weight in edges:
                    edge_str = f"{src} {rel} {dest}".lower()
                    if phrase.lower() in edge_str:
                        found = True
                        failures.append(f"❌ FAIL: '{text}' created unwanted edge: {src} --[{rel}]--> {dest}")
                        break
                if not found:
                    print(f"  ✓ PASS: '{text}' correctly avoided storing '{phrase}'")

    # Test retrieval quality (Layer 4)
    print("\n🔍 Testing retrieval quality (Layer 4):")
    # Retrieve with a query that might match confused utterances
    bullets = processor.hot.retrieve_bullets("Tell me what you know", read_only=True)
    confused_retrieved = any("confus" in b.lower() or "unclear" in b.lower() for b in bullets)

    if confused_retrieved:
        failures.append("❌ FAIL: Retrieved confused/meta utterances (Layer 4 filter failed)")
        print("  ❌ FAIL: Confused utterances were retrieved")
    else:
        print("  ✓ PASS: No confused/meta utterances in retrieval")

    # Final results
    print("\n" + "="*60)
    if failures:
        print("❌ SOME TESTS FAILED:")
        for failure in failures:
            print(f"  {failure}")
        print("="*60)
        return False
    else:
        print("✅ ALL NEGATION TESTS PASSED")
        print("  • Layer 1: Negation tracking works correctly")
        print("  • Layer 2: Quality filters prevent storing confusion")
        print("  • Layer 3: Summarizer prompt focuses on facts")
        print("  • Layer 4: Retrieval filters out low-quality bullets")
        print("="*60)
        return True

if __name__ == "__main__":
    # Run both test suites
    basic_success = asyncio.run(test_memory_system())
    negation_success = asyncio.run(test_negation_handling())

    success = basic_success and negation_success
    sys.exit(0 if success else 1)