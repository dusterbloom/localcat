#!/usr/bin/env python3
"""
Debug the text preprocessing and chunking logic to identify issues.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tools.text_formatter import (
    sanitize_for_voice,
    sanitize_for_kokoro,
    split_text_for_kokoro_streaming,
    smart_sentence_split
)


def debug_text_processing():
    """Debug text processing with the problematic text."""

    print("🔍 DEBUGGING TEXT PREPROCESSING")
    print("=" * 60)

    # Original problematic text
    original_text = "Of course! Your dog's name is Po and Potola. Is there anything else you'd like to tell me about him ?"

    print(f"📝 Original text: '{original_text}'")
    print(f"   Length: {len(original_text)} characters")

    # Step 1: General voice sanitization
    sanitized_voice = sanitize_for_voice(original_text)
    print(f"\n🧹 After sanitize_for_voice: '{sanitized_voice}'")
    print(f"   Length: {len(sanitized_voice)} characters")

    # Step 2: Kokoro-specific sanitization
    sanitized_kokoro = sanitize_for_kokoro(original_text)
    print(f"\n🎭 After sanitize_for_kokoro: '{sanitized_kokoro}'")
    print(f"   Length: {len(sanitized_kokoro)} characters")

    # Step 3: Split for streaming
    chunks = split_text_for_kokoro_streaming(original_text, min_length=50, max_length=120)
    print(f"\n📦 Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: '{chunk}' ({len(chunk)} chars)")

    # Test with different parameters
    print(f"\n🔧 Testing different chunk parameters:")

    test_params = [
        (30, 80),
        (50, 100),
        (50, 120),
        (70, 150)
    ]

    for min_len, max_len in test_params:
        test_chunks = split_text_for_kokoro_streaming(original_text, min_length=min_len, max_length=max_len)
        print(f"   min={min_len}, max={max_len}: {len(test_chunks)} chunks")
        for i, chunk in enumerate(test_chunks):
            print(f"     {i+1}: '{chunk}' ({len(chunk)} chars)")

    # Analyze potential issues
    print(f"\n🔍 ANALYZING POTENTIAL ISSUES:")

    # Check for problematic characters at sentence ends
    for i, chunk in enumerate(chunks):
        chunk_cleaned = chunk.strip()
        if chunk_cleaned:
            last_char = chunk_cleaned[-1]
            print(f"   Chunk {i+1} ends with: '{last_char}' (ord: {ord(last_char)})")

            # Check for spaces or unusual characters
            if last_char == ' ':
                print(f"     ⚠️  Chunk ends with space!")
            elif last_char not in '.!?':
                print(f"     ⚠️  Chunk doesn't end with proper punctuation!")

            # Check for quotes or special characters
            if '"' in chunk or "'" in chunk or "'" in chunk or '"' in chunk or '"' in chunk:
                print(f"     ⚠️  Chunk contains quotes or special apostrophes!")

    # Test individual problematic parts
    print(f"\n🧪 TESTING INDIVIDUAL PROBLEMATIC PARTS:")

    test_phrases = [
        "Of course!",
        "Your dog's name is Po and Potola.",
        "Is there anything else you'd like to tell me about him ?",
        "Po and Potola.",
        "him ?",
        "about him ?"
    ]

    for phrase in test_phrases:
        sanitized = sanitize_for_kokoro(phrase)
        print(f"   '{phrase}' → '{sanitized}'")
        if phrase != sanitized:
            print(f"     ⚠️  Text was modified during sanitization!")


def test_edge_cases():
    """Test edge cases that might cause artifacts."""

    print(f"\n🚨 TESTING EDGE CASES:")

    edge_cases = [
        "Test.",  # Simple ending
        "Test. ",  # Ending with space
        "Test?",   # Question mark
        "Test! ",  # Exclamation with space
        "Test's thing.",  # Apostrophe
        "Test \"quoted\".",  # Quotes
        "Test (parenthetical).",  # Parentheses
        "Test... ellipsis.",  # Ellipsis
        "Test -- dash.",  # Dashes
        "Test, comma end.",  # Comma
        "Test; semicolon.",  # Semicolon
    ]

    for text in edge_cases:
        chunks = split_text_for_kokoro_streaming(text)
        sanitized = sanitize_for_kokoro(text)
        print(f"   '{text}' → sanitized: '{sanitized}' → chunks: {chunks}")


if __name__ == "__main__":
    debug_text_processing()
    test_edge_cases()