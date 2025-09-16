#!/usr/bin/env python3
"""
Test retrieval timing with real memory data
"""

import os
import sys
import time
from loguru import logger

# Add server to path
sys.path.insert(0, os.path.dirname(__file__))

def test_retrieval():
    """Test retrieval with existing memory data"""

    from components.memory.memory_config import MemoryConfig
    from components.memory.hotmemory_facade import HotMemory

    # Initialize with real database
    config = MemoryConfig()
    hotmem = HotMemory(config)

    # Test queries that would trigger retrieval
    test_queries = [
        ("Where do I live?", "casual"),
        ("Tell me about my work", "casual"),
        ("What did we discuss earlier?", "casual"),
        ("What's my name?", "casual"),
        ("Do you remember what I told you about San Francisco?", "casual"),
    ]

    logger.info("=" * 80)
    logger.info("TESTING RETRIEVAL TIMING WITH REAL DATA")
    logger.info("=" * 80)

    for i, (query, mode) in enumerate(test_queries, 1):
        logger.info(f"\nTest {i}: '{query}'")

        # Process the query
        start = time.perf_counter()
        result = hotmem.process_turn(
            text=query,
            turn_id=i,
            session_id="test_session",
            user_id="test_user",
            mode=mode
        )
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(f"  Total time: {elapsed:.0f}ms")
        logger.info(f"  Bullets returned: {len(result.bullets)}")
        logger.info(f"  Intent: {result.intent.intent.value if result.intent else 'None'}")
        logger.info(f"  Needs retrieval: {result.needs_retrieval}")

        # Small delay between queries
        time.sleep(0.1)

    logger.info("\n" + "=" * 80)
    logger.info("RETRIEVAL TIMING TEST COMPLETE")
    logger.info("Check logs above for SLOW RETRIEVAL and BOTTLENECK warnings")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_retrieval()
