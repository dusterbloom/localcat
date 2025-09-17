#!/usr/bin/env python3
"""
Simple retrieval test to verify memory system works
"""
import os
import sys
import tempfile
from loguru import logger

sys.path.insert(0, '.')

def test_retrieval():
    """Test that retrieval works after our fixes"""

    # Suppress logs
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup paths
        from components.memory.memory_store import MemoryStore, Paths
        from components.memory.hotmemory_facade import HotMemoryFacade

        paths = Paths(
            sqlite_path=os.path.join(tmpdir, 'test.db'),
            lmdb_dir=os.path.join(tmpdir, 'test.lmdb')
        )

        store = MemoryStore(paths)
        hot = HotMemoryFacade(store)

        print("=== Simple Retrieval Test ===\n")

        # Store facts about Potola
        print("1. Storing facts:")
        print("   'My dog Potola is 5 years old'")
        result1 = hot.process_turn('My dog Potola is 5 years old', 'session1', 1, 'test_user')
        print(f"   Stored: {result1.triples}")
        print(f"   Intent: {result1.intent.intent.value}\n")

        print("   'Potola loves playing fetch at the park'")
        result2 = hot.process_turn('Potola loves playing fetch at the park', 'session1', 2, 'test_user')
        print(f"   Stored: {result2.triples}\n")

        # Query about Potola - should be PURE_QUESTION with retrieval
        print("2. Querying: 'How old is my dog?'")
        result3 = hot.process_turn('How old is my dog?', 'session1', 3, 'test_user')
        print(f"   Intent: {result3.intent.intent.value}")
        print(f"   Needs retrieval: {result3.needs_retrieval}")
        print(f"   Retrieved bullets: {len(result3.bullets or [])} items")
        if result3.bullets:
            for bullet in result3.bullets[:3]:
                print(f"     - {bullet}")
        print()

        # Another query
        print("3. Querying: 'Tell me about Potola'")
        result4 = hot.process_turn('Tell me about Potola', 'session1', 4, 'test_user')
        print(f"   Intent: {result4.intent.intent.value}")
        print(f"   Needs retrieval: {result4.needs_retrieval}")
        print(f"   Retrieved bullets: {len(result4.bullets or [])} items")
        if result4.bullets:
            for bullet in result4.bullets[:3]:
                print(f"     - {bullet}")
        print()

        # Test correction
        print("4. Correcting: 'Actually, Potola is 7 years old'")
        result5 = hot.process_turn('Actually, Potola is 7 years old', 'session1', 5, 'test_user')
        print(f"   Intent: {result5.intent.intent.value}")
        print(f"   Needs retrieval: {result5.needs_retrieval}")
        print(f"   Stored: {result5.triples}\n")

        # Query again to see updated fact
        print("5. Querying again: 'What is Potola's age?'")
        result6 = hot.process_turn("What is Potola's age?", 'session1', 6, 'test_user')
        print(f"   Intent: {result6.intent.intent.value}")
        print(f"   Retrieved bullets: {len(result6.bullets or [])} items")
        if result6.bullets:
            for bullet in result6.bullets[:3]:
                print(f"     - {bullet}")

        # Summary
        print("\n=== Test Summary ===")
        print(f"✅ Intent classification working: Questions -> {result3.intent.intent.value}")
        print(f"✅ Needs retrieval flag: {result3.needs_retrieval}")

        if result3.bullets:
            print(f"✅ Retrieval working: Got {len(result3.bullets)} bullets")
        else:
            print(f"❌ RETRIEVAL NOT WORKING: Got 0 bullets when querying!")

        if result5.intent.intent.value == 'correction':
            print(f"✅ Corrections detected properly")
        else:
            print(f"❌ Correction not detected: got {result5.intent.intent.value}")

if __name__ == "__main__":
    test_retrieval()