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

if __name__ == "__main__":
    success = asyncio.run(test_memory_system())
    sys.exit(0 if success else 1)