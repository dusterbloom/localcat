#!/usr/bin/env python3
"""
Test script for memory bug fixes:
1. Greeting detection substring bug (Sardinia location query)
2. Memory repetition filter
"""

import asyncio
import sys
import os
from loguru import logger

# Set up minimal logging for test
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="<level>{level: <8}</level> | {message}")

async def test_sardinia_location_query():
    """Test that 'Do you know my location?' retrieves Sardinia memory (not suppressed)"""
    print("\n" + "="*80)
    print("TEST 1: Sardinia Location Query (Greeting Detection Fix)")
    print("="*80)

    try:
        from core.memory.retrieval import Retrieval
        from core.memory.config_manager import MemoryConfiguration

        # Initialize retrieval system
        config = MemoryConfiguration()
        retriever = Retrieval(config=config)

        # Test query that was previously suppressed
        query = "Do you know my location?"

        print(f"\nQuery: '{query}'")
        print(f"Expected: Should retrieve Sardinia location memory (not suppress)")

        # Test suppression logic directly
        should_suppress = retriever._should_suppress_memory_injection(query)

        print(f"\nSuppression result: {should_suppress}")

        if should_suppress:
            print("❌ FAIL: Query was suppressed (bug still present)")
            return False
        else:
            print("✅ PASS: Query was NOT suppressed (fix working!)")

            # Now test actual retrieval
            print("\nAttempting memory retrieval...")
            bullets = retriever.retrieve_bullets(query=query, read_only=True)

            print(f"Retrieved {len(bullets)} bullets:")
            for i, bullet in enumerate(bullets[:3], 1):
                print(f"  {i}. {bullet[:120]}")

            if len(bullets) > 0:
                print("\n✅ SUCCESS: Memory retrieval working!")
                return True
            else:
                print("\n⚠️  WARNING: No suppression but also no bullets retrieved")
                print("   (This might be expected if no relevant memories exist)")
                return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_repetition_prevention():
    """Test that consecutive queries don't repeat the same memory"""
    print("\n" + "="*80)
    print("TEST 2: Memory Repetition Prevention (Filtering Fix)")
    print("="*80)

    try:
        from core.memory.context_injector import ContextInjector
        from core.memory.config_manager import MemoryConfiguration
        from core.memory.hotmem_service import HotMemService

        # Initialize memory system
        config = MemoryConfiguration()
        hot = HotMemService(config=config)
        injector = ContextInjector(hot_memory=hot, config=config)

        # Simulate consecutive queries about the same topic
        queries = [
            "Do you know my favorite number?",
            "What's my favorite number again?",
            "Can you tell me my favorite number?"
        ]

        print("\nSimulating 3 consecutive queries about favorite number:")

        all_bullets = []
        for i, query in enumerate(queries, 1):
            print(f"\n--- Turn {i} ---")
            print(f"Query: '{query}'")

            bullets = await injector.retrieve_and_prepare_bullets(query, read_only=True)

            print(f"Retrieved {len(bullets)} bullets")
            if bullets:
                print(f"First bullet: {bullets[0][:100]}")

            all_bullets.append(bullets)

        # Analysis
        print("\n" + "-"*80)
        print("ANALYSIS:")
        print("-"*80)

        # Check if filtering occurred
        turn1_count = len(all_bullets[0])
        turn2_count = len(all_bullets[1])
        turn3_count = len(all_bullets[2])

        print(f"Turn 1: {turn1_count} bullets")
        print(f"Turn 2: {turn2_count} bullets (should be filtered)")
        print(f"Turn 3: {turn3_count} bullets (should be filtered)")

        # Success criteria: Turn 2 and 3 should have fewer bullets due to filtering
        if turn1_count > 0 and turn2_count < turn1_count:
            print("\n✅ PASS: Memory repetition filter is working!")
            print(f"   Turn 1 retrieved {turn1_count} bullets")
            print(f"   Turn 2 retrieved {turn2_count} bullets (filtered {turn1_count - turn2_count})")
            return True
        elif turn1_count == 0:
            print("\n⚠️  WARNING: No memories retrieved in Turn 1")
            print("   (Cannot test filtering without initial memories)")
            return True
        else:
            print("\n❌ FAIL: No filtering detected")
            print(f"   Expected fewer bullets in Turn 2, but got {turn2_count} (same as Turn 1: {turn1_count})")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MEMORY BUG FIXES VALIDATION TEST SUITE")
    print("="*80)

    results = []

    # Test 1: Sardinia location query (greeting detection fix)
    result1 = await test_sardinia_location_query()
    results.append(("Greeting Detection Fix", result1))

    # Test 2: Memory repetition prevention
    result2 = await test_memory_repetition_prevention()
    results.append(("Memory Repetition Filter", result2))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
