#!/usr/bin/env python
"""Test script to verify extraction is working correctly after the fix."""

import os
import sys
from pathlib import Path

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables
os.environ['DEFAULT_EXTRACTION_STRATEGY'] = 'enhanced_level3'
os.environ['FALLBACK_EXTRACTION_STRATEGY'] = ''

from components.memory.hotmemory_facade import HotMemoryFacade

def test_extraction():
    """Test that extraction now works correctly."""
    # Initialize the facade
    facade = HotMemoryFacade()

    # Test extraction with "I live in Sardinia"
    test_text = "I live in Sardinia."
    print(f"\nTesting extraction for: '{test_text}'")
    print("=" * 60)

    result = facade.process_turn(
        text=test_text,
        turn_id=1,
        session_id="test_session"
    )

    print(f"Intent: {result.intent.intent.value if result.intent else 'None'}")
    print(f"Needs storage: {result.needs_storage}")
    print(f"Needs retrieval: {result.needs_retrieval}")
    print(f"Extracted triples: {result.triples}")
    print(f"Memory bullets: {result.bullets}")

    # Verify extraction worked
    if result.triples:
        print(f"\n✅ SUCCESS! Extraction is working. Found {len(result.triples)} triples:")
        for triple in result.triples:
            print(f"  - {triple}")
    else:
        print("\n❌ FAILURE! No triples extracted.")

    # Test retrieval with "Do you know where I live?"
    test_query = "Do you know where I live?"
    print(f"\n\nTesting retrieval for: '{test_query}'")
    print("=" * 60)

    result2 = facade.process_turn(
        text=test_query,
        turn_id=2,
        session_id="test_session"
    )

    print(f"Intent: {result2.intent.intent.value if result2.intent else 'None'}")
    print(f"Needs storage: {result2.needs_storage}")
    print(f"Needs retrieval: {result2.needs_retrieval}")
    print(f"Memory bullets retrieved: {result2.bullets}")

    if result2.bullets:
        print(f"\n✅ SUCCESS! Retrieval is working. Found {len(result2.bullets)} memory bullets:")
        for bullet in result2.bullets:
            print(f"  - {bullet}")
    else:
        print("\n⚠️  No memory bullets retrieved (might be expected if memory was cleared).")

if __name__ == "__main__":
    test_extraction()