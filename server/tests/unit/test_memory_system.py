#!/usr/bin/env python3
"""Test memory system functionality"""

import asyncio
import sys
from loguru import logger
from hotpath_processor import HotPathMemoryProcessor
from pipecat.frames.frames import TranscriptionFrame
import os

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

    # Get current memory state
    memory_context = processor.hot.get_context("test-user", k=5)
    if memory_context:
        print(f"  Context: {memory_context}")

    # Check edges
    edges = list(processor.hot.graph.edges(data=True))
    print(f"\n  Found {len(edges)} edges in memory graph:")
    for src, dest, data in edges[:5]:  # Show first 5
        print(f"    • {src} → {dest} ({data.get('relation', 'unknown')})")

    # Test retrieval
    print("\n🔍 Testing retrieval:")
    test_queries = [
        "Where does John live?",
        "What is John's pet?",
        "John's job"
    ]

    for query in test_queries:
        context = processor.hot.get_context(query, k=3)
        print(f"  Query: '{query}'")
        print(f"    → Context: {context if context else 'No relevant context found'}")

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