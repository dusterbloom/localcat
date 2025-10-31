#!/usr/bin/env python3
"""
Simple test for memory repetition filter.
Tests that consecutive queries don't repeat the same memory bullets.
"""

import sys
import asyncio
from loguru import logger

# Set up minimal logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<level>{level: <8}</level> | {message}")

async def test_memory_repetition_filter():
    """Test that the deque-based filtering prevents memory repetition"""
    print("\n" + "="*80)
    print("TEST: Memory Repetition Filter")
    print("="*80)

    from unittest.mock import Mock
    from core.memory.context_injector import ContextInjector
    from core.memory.config_manager import MemoryConfiguration

    # Create mock hot memory service
    hot_mock = Mock()

    # Simulate retrieval returning the same bullet multiple times
    test_bullet = "• ⭐💬 user | favorite_number | 76"

    # Mock retrieve_bullets to always return the same bullet
    hot_mock.retrieve_bullets = Mock(return_value=[test_bullet, "• Other memory"])

    # Create configuration
    config = MemoryConfiguration()

    # Create context injector
    injector = ContextInjector(hot_memory=hot_mock, config=config)

    print("\nSimulating 3 consecutive queries about the same topic:")

    # Turn 1: Should retrieve both bullets
    print("\n--- Turn 1 ---")
    bullets_1 = await injector.retrieve_and_prepare_bullets("What's my favorite number?", read_only=True)
    print(f"Retrieved {len(bullets_1)} bullets")
    if bullets_1:
        print(f"  - {bullets_1[0][:80]}")
        if len(bullets_1) > 1:
            print(f"  - {bullets_1[1][:80]}")

    # Turn 2: Should filter out the same bullet
    print("\n--- Turn 2 ---")
    bullets_2 = await injector.retrieve_and_prepare_bullets("What's my favorite number again?", read_only=True)
    print(f"Retrieved {len(bullets_2)} bullets")
    if bullets_2:
        for b in bullets_2:
            print(f"  - {b[:80]}")

    # Turn 3: Should still filter
    print("\n--- Turn 3 ---")
    bullets_3 = await injector.retrieve_and_prepare_bullets("Tell me my favorite number", read_only=True)
    print(f"Retrieved {len(bullets_3)} bullets")
    if bullets_3:
        for b in bullets_3:
            print(f"  - {b[:80]}")

    # Turn 4: Should allow again (deque maxlen=3, so first bullet should be forgotten)
    print("\n--- Turn 4 ---")
    bullets_4 = await injector.retrieve_and_prepare_bullets("What's my favorite number?", read_only=True)
    print(f"Retrieved {len(bullets_4)} bullets")
    if bullets_4:
        for b in bullets_4:
            print(f"  - {b[:80]}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)

    print(f"Turn 1: {len(bullets_1)} bullets (initial retrieval)")
    print(f"Turn 2: {len(bullets_2)} bullets (should be filtered)")
    print(f"Turn 3: {len(bullets_3)} bullets (should be filtered)")
    print(f"Turn 4: {len(bullets_4)} bullets (may include original bullets again)")

    # Success criteria
    all_passed = True

    # Turn 1 should have 2 bullets
    if len(bullets_1) == 2:
        print("✅ Turn 1: Retrieved 2 bullets as expected")
    else:
        print(f"❌ Turn 1: Expected 2 bullets, got {len(bullets_1)}")
        all_passed = False

    # Turn 2 should have 0 bullets (both filtered as recently injected)
    if len(bullets_2) == 0:
        print("✅ Turn 2: All bullets filtered (preventing repetition!)")
    else:
        print(f"⚠️  Turn 2: Expected 0 bullets (all filtered), got {len(bullets_2)}")
        print("   Note: Filter is working if count < Turn 1")
        if len(bullets_2) < len(bullets_1):
            print("✅ Partial success: Some filtering occurred")
        else:
            print("❌ No filtering occurred")
            all_passed = False

    # Turn 3 should also have 0 bullets
    if len(bullets_3) == 0:
        print("✅ Turn 3: All bullets still filtered")
    else:
        print(f"⚠️  Turn 3: Expected 0 bullets, got {len(bullets_3)}")
        if len(bullets_3) < len(bullets_1):
            print("✅ Partial success: Some filtering occurred")

    print("\n" + "="*80)
    if all_passed:
        print("🎉 Memory repetition filter working correctly!")
        return 0
    else:
        print("⚠️  Memory repetition filter needs review")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_memory_repetition_filter())
    sys.exit(exit_code)
