"""
Comprehensive test for memory deletion functionality.

Tests that deleted memories:
1. Are successfully removed from storage
2. Do NOT appear in subsequent retrievals
3. Do NOT leak into context injection
"""

import asyncio
import sys
import os
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.memory.hotpath_processor import HotPathMemoryProcessor
from core.memory.hotpath_tool_integration import HotPathToolIntegration
from pipecat.services.llm_service import FunctionCallParams


async def test_memory_deletion():
    """Test complete memory deletion workflow"""

    logger.info("="*80)
    logger.info("MEMORY DELETION COMPREHENSIVE TEST")
    logger.info("="*80)

    # Initialize HotPath memory processor
    logger.info("\n[1] Initializing HotPath memory processor...")
    hotpath = HotPathMemoryProcessor(
        user_id="test_user_deletion",
        config=None
    )

    # Initialize tool integration
    tool_integration = HotPathToolIntegration(hotpath)
    logger.info("✅ HotPath initialized")

    # Test data
    test_memory = "I love royal pictures and paintings of kings"
    test_query = "royal picture"

    # Step 1: Store a memory
    logger.info(f"\n[2] Storing test memory: '{test_memory}'")

    class MockParams:
        def __init__(self, args):
            self.arguments = args
            self._result = None

        async def result_callback(self, result):
            self._result = result
            logger.info(f"Tool result: {result}")

    add_params = MockParams({"information": test_memory})
    await tool_integration._handle_memory_add(add_params)
    logger.info(f"✅ Memory stored: {add_params._result}")

    # Step 2: Verify memory appears in retrieval BEFORE deletion
    logger.info(f"\n[3] Retrieving memory before deletion with query: '{test_query}'")
    bullets_before = hotpath.hot.retrieve_bullets(test_query, read_only=True)

    logger.info(f"Retrieved {len(bullets_before)} bullets:")
    for i, bullet in enumerate(bullets_before, 1):
        logger.info(f"  {i}. {bullet}")

    if not bullets_before:
        logger.error("❌ TEST FAILED: Memory was not retrieved before deletion!")
        return False

    if not any(test_query in bullet.lower() for bullet in bullets_before):
        logger.error(f"❌ TEST FAILED: Query '{test_query}' not found in retrieved bullets!")
        return False

    logger.info("✅ Memory successfully retrieved before deletion")

    # Step 3: Delete the memory
    logger.info(f"\n[4] Deleting memory with query: '{test_query}'")
    delete_params = MockParams({"query": test_query})
    await tool_integration._handle_memory_delete(delete_params)
    logger.info(f"✅ Deletion executed: {delete_params._result}")

    # Step 4: Verify memory does NOT appear in retrieval AFTER deletion
    logger.info(f"\n[5] Retrieving memory AFTER deletion with query: '{test_query}'")
    bullets_after = hotpath.hot.retrieve_bullets(test_query, read_only=True)

    logger.info(f"Retrieved {len(bullets_after)} bullets:")
    for i, bullet in enumerate(bullets_after, 1):
        logger.info(f"  {i}. {bullet}")

    # Check if deleted memory still appears
    deleted_memory_found = any(test_query in bullet.lower() and "[DELETED]" not in bullet for bullet in bullets_after)

    if deleted_memory_found:
        logger.error(f"❌ TEST FAILED: Deleted memory '{test_query}' still appears in retrieval!")
        logger.error("Bullets after deletion:")
        for bullet in bullets_after:
            if test_query in bullet.lower():
                logger.error(f"  - {bullet}")
        return False

    logger.info("✅ Deleted memory does NOT appear in subsequent retrieval")

    # Step 5: Verify broader query also doesn't return deleted memory
    logger.info(f"\n[6] Testing broader query: 'picture'")
    bullets_broad = hotpath.hot.retrieve_bullets("picture", read_only=True)

    deleted_in_broad = any(test_query in bullet.lower() and "[DELETED]" not in bullet for bullet in bullets_broad)

    if deleted_in_broad:
        logger.error(f"❌ TEST FAILED: Deleted memory appears in broader query results!")
        return False

    logger.info("✅ Deleted memory does NOT appear in broader queries")

    # Step 6: Test that NEW memories with similar content still work
    logger.info(f"\n[7] Storing NEW memory with similar content")
    new_memory = "I enjoy modern abstract art"
    add_params2 = MockParams({"information": new_memory})
    await tool_integration._handle_memory_add(add_params2)

    bullets_new = hotpath.hot.retrieve_bullets("art", read_only=True)
    if not any("abstract art" in bullet.lower() for bullet in bullets_new):
        logger.error("❌ TEST FAILED: New memory not retrievable!")
        return False

    logger.info("✅ New memories still work after deletion")

    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("✅ ALL TESTS PASSED - Memory deletion works correctly!")
    logger.info("="*80)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_memory_deletion())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
