#!/usr/bin/env python3
"""
Test that questions trigger entity extraction and retrieval.

This test verifies the fix for the issue where:
- User says: "My favorite food is steak"
- User asks: "What's my favorite food?"
- System should answer: "Your favorite food is steak"

Before fix: Question skipped entity extraction → no graph traversal → "I don't know"
After fix: Question extracts entities → graph traversal finds answer → "steak"
"""
import sys
import os
from pathlib import Path
import asyncio

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.memory.hotpath_processor import HotPathMemoryProcessor
import time

def test_question_retrieval():
    """Test that questions can retrieve from memory via entity extraction."""

    logger.info("=" * 80)
    logger.info("TEST: Question-Based Memory Retrieval")
    logger.info("=" * 80)

    # Initialize with production database
    os.environ['MEMORY_DB_PATH'] = '/Users/peppi/Library/Application Support/LocalCat/data/memory.db'
    os.environ['MEMORY_LMDB_PATH'] = '/Users/peppi/Library/Application Support/LocalCat/data/memory.lmdb'

    logger.info("\n[1] Initializing HotPath memory processor...")
    hotpath = HotPathMemoryProcessor(
        user_id="test_question_user",
        config=None
    )

    # Store some facts - need to store both hops separately like in production
    logger.info("\n[2] Storing test memories (two turns for complete graph)")
    current_time = int(time.time() * 1000)
    session_id = f"test_session_{current_time}"

    # First turn: establish the preference
    hotpath.hot.store.enqueue_mention(
        session_id,
        "My favorite food is steak",
        current_time,
        session_id,
        1
    )

    bullets1, triples1 = hotpath.hot.process_turn(
        "My favorite food is steak",
        session_id,
        1
    )

    # Second turn: make the value explicit (helps extraction create second hop)
    hotpath.hot.store.enqueue_mention(
        session_id,
        "Steak is my favorite food",
        current_time + 100,
        session_id,
        2
    )

    bullets2, triples2 = hotpath.hot.process_turn(
        "Steak is my favorite food",
        session_id,
        2
    )

    all_triples = triples1 + triples2
    logger.info(f"Stored {len(all_triples)} triples across 2 turns:")
    for triple in all_triples:
        logger.info(f"  {triple}")

    # Now ask a question
    logger.info("\n[3] Asking question: 'What's my favorite food?'")

    # The fix: this should now extract entities even though it's a question
    query = "What's my favorite food?"

    # Extract entities (this is what the fix enables)
    entities, _, _, _, _ = hotpath.hot._cached_extract(query, "en")
    entities = hotpath.hot.extractor.refine_entities(query, entities)
    logger.info(f"Extracted entities: {entities}")

    # Retrieve using the question
    bullets_retrieved = hotpath.hot.retrieve_bullets(query, read_only=True)

    logger.info(f"\nRetrieved {len(bullets_retrieved)} bullets:")
    for i, bullet in enumerate(bullets_retrieved, 1):
        logger.info(f"  {i}. {bullet}")

    # Check if we got the answer
    found_steak = any("steak" in bullet.lower() for bullet in bullets_retrieved)
    found_food = any("food" in bullet.lower() or "favorite" in bullet.lower() for bullet in bullets_retrieved)

    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)

    success = True

    if not entities:
        logger.error("❌ FAIL: No entities extracted from question")
        success = False
    else:
        logger.info(f"✅ PASS: Entities extracted: {entities}")

    if not bullets_retrieved or "[diag]" in bullets_retrieved[0]:
        logger.error("❌ FAIL: No memories retrieved (got diagnostic message)")
        success = False
    else:
        logger.info(f"✅ PASS: Retrieved {len(bullets_retrieved)} memory bullets")

    if not found_steak:
        logger.error("❌ FAIL: Did not find 'steak' in retrieved memories")
        success = False
    else:
        logger.info("✅ PASS: Found 'steak' in retrieved memories")

    if not found_food:
        logger.warning("⚠️  WARNING: Did not find 'food' or 'favorite' in retrieved memories")
    else:
        logger.info("✅ PASS: Found 'food'/'favorite' context")

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("Question-based retrieval is working correctly.")
    else:
        logger.error("❌ TESTS FAILED")
        logger.error("Question-based retrieval needs more work.")
    logger.info("=" * 80)

    return success


if __name__ == "__main__":
    try:
        success = test_question_retrieval()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
