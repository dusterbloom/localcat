#!/usr/bin/env python3
"""Test the optimized Kokoro chunking logic."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tools.text_formatter import split_text_for_kokoro_streaming


def test_optimized_chunking():
    """Test the new optimized chunking for Kokoro."""

    print("🧪 Testing Optimized Kokoro Chunking")
    print("=" * 50)

    test_cases = [
        {
            "name": "Memory injection filtering",
            "text": "You're right to feel that way! [graph] that centralsystem is scary - 0s ago It's good you mentioned it.",
            "expected_behavior": "Filter out memory context, keep natural sentences"
        },
        {
            "name": "Small sentences grouping",
            "text": "Hello there. How are you? I'm fine. Thanks for asking.",
            "expected_behavior": "Group small sentences together"
        },
        {
            "name": "Optimal size sentence",
            "text": "This is a sentence that is exactly the right size for Kokoro to process efficiently and naturally.",
            "expected_behavior": "Keep as single chunk"
        },
        {
            "name": "Very long sentence",
            "text": "This is an extremely long sentence that goes on and on with lots of details and information that would be too much for Kokoro to process efficiently in a single chunk, so it should be split at natural boundaries.",
            "expected_behavior": "Split at natural boundaries"
        },
        {
            "name": "Mixed content",
            "text": "You're right to feel that way! It's good you mentioned it. • [graph] that centralsystem is scary (0s ago) Let's definitely approach it with caution and think carefully about the implications. Do you want to talk more about that?",
            "expected_behavior": "Filter memory, group appropriately"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"Input: '{test_case['text']}'")
        print(f"Expected: {test_case['expected_behavior']}")

        chunks = split_text_for_kokoro_streaming(test_case['text'])

        print(f"Result: {len(chunks)} chunks")
        for j, chunk in enumerate(chunks, 1):
            print(f"  Chunk {j} ({len(chunk)} chars): '{chunk}'")

        # Analyze results
        total_chars = sum(len(chunk) for chunk in chunks)
        avg_length = total_chars / len(chunks) if chunks else 0

        print(f"Analysis:")
        print(f"  • Total chunks: {len(chunks)}")
        print(f"  • Avg length: {avg_length:.1f} chars")
        print(f"  • Range: {min(len(c) for c in chunks) if chunks else 0}-{max(len(c) for c in chunks) if chunks else 0} chars")

        # Check for memory injection
        has_memory = any('[graph]' in chunk or '•' in chunk or 'ago)' in chunk for chunk in chunks)
        print(f"  • Memory filtered: {'❌ FAILED' if has_memory else '✅ SUCCESS'}")

        # Check chunk sizes are in optimal range (50-120)
        optimal_chunks = sum(1 for chunk in chunks if 50 <= len(chunk) <= 120)
        print(f"  • Optimal size chunks: {optimal_chunks}/{len(chunks)}")


if __name__ == "__main__":
    test_optimized_chunking()