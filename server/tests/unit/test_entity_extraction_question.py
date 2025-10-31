#!/usr/bin/env python3
"""
Test entity extraction from questions.

Root cause: "What's my favorite food?" extracts 0 entities even though
question detection is working. This test verifies entity extraction.
"""
import sys
import os
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.memory.hotpath_processor import HotPathMemoryProcessor

def test_entity_extraction_from_question():
    """Test that entity extraction works for questions."""

    logger.info("=" * 80)
    logger.info("TEST: Entity Extraction from Questions")
    logger.info("=" * 80)

    # Initialize with production database
    os.environ['MEMORY_DB_PATH'] = os.path.expanduser('~/Library/Application Support/LocalCat/data/memory.db')
    os.environ['MEMORY_LMDB_PATH'] = os.path.expanduser('~/Library/Application Support/LocalCat/data/memory.lmdb')

    logger.info("\n[1] Initializing HotPath memory processor...")
    hotpath = HotPathMemoryProcessor(
        user_id="test_entity_extraction",
        config=None
    )

    # Test questions
    test_questions = [
        "What's my favorite food?",
        "What is my favorite food?",
        "What's my name?",
        "Where do I live?",
        "Tell me my favorite color"
    ]

    logger.info(f"\n[2] Testing entity extraction for {len(test_questions)} questions\n")

    results = []
    for question in test_questions:
        logger.info(f"Question: '{question}'")

        # Extract entities using the same path as process_turn
        entities, _, _, doc, entity_aliases = hotpath.hot._cached_extract(question, "en")
        logger.info(f"  Raw entities: {entities}")

        # Apply refinement
        refined_entities = hotpath.hot.extractor.refine_entities(question, entities)
        logger.info(f"  Refined entities: {refined_entities}")

        # Try to extract triples (this is what process_turn does)
        try:
            triples = hotpath.hot.extractor.extract(question, "en")
            logger.info(f"  Triples extracted: {len(triples)}")
            for triple in triples[:3]:  # Show first 3
                logger.info(f"    - {triple}")
        except Exception as e:
            logger.error(f"  Triple extraction failed: {e}")
            triples = []

        results.append({
            'question': question,
            'raw_entities': len(entities),
            'refined_entities': len(refined_entities),
            'triples': len(triples)
        })
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"'{result['question']}'")
        logger.info(f"  Raw: {result['raw_entities']}, Refined: {result['refined_entities']}, Triples: {result['triples']}")

    # Check if any questions extracted entities
    success = any(r['refined_entities'] > 0 for r in results)

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ At least some questions extracted entities")
    else:
        logger.error("❌ FAIL: NO questions extracted any entities!")
        logger.error("This explains why question-based retrieval fails!")
    logger.info("=" * 80)

    return success


if __name__ == "__main__":
    try:
        success = test_entity_extraction_from_question()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
