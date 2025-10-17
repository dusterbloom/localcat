#!/usr/bin/env python3
"""Test the text formatter to ensure contractions are handled correctly."""

import sys
import os
# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from text_formatter import sanitize_for_voice, chunk_for_kokoro_ultra_low_latency


def test_contractions():
    """Test various contractions."""
    test_cases = [
        ("You'd like to go there", "Expected: Preserves you'd"),
        ("I'll be there soon", "Expected: Preserves I'll"),
        ("They're coming over", "Expected: Preserves they're"),
        ("We've been waiting", "Expected: Preserves we've"),
        ("It's a beautiful day", "Expected: Preserves it's"),
        ("You're welcome", "Expected: Preserves you're"),
        ("I'd rather not", "Expected: Preserves I'd"),
        ("She'll arrive tomorrow", "Expected: Preserves she'll"),
        ("That'd be great", "Expected: Preserves that'd"),
        ("What's happening?", "Expected: Preserves what's"),
        ("Don't worry about it", "Expected: Preserves don't"),
        ("Can't do that", "Expected: Preserves can't"),
        ("Won't be long", "Expected: Preserves won't"),
        ("Shouldn't have done that", "Expected: Preserves shouldn't"),
    ]

    print("Testing contraction handling in TTS text formatter:\n")
    for text, description in test_cases:
        result = sanitize_for_voice(text)
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print(f"        {description}")
        print()


def test_special_chars():
    """Test special character handling."""
    test_cases = [
        ("Hello `world`", "Should remove backticks"),
        ("Test with emojis 😊 🎉", "Should remove emojis"),
        ("Multiple    spaces", "Should normalize spaces"),
        ("**Bold** text", "Should remove markdown"),
        ("https://example.com link", "Should remove URLs"),
        ("Various apostrophes: ' ' `", "Should normalize apostrophes"),
    ]

    print("\nTesting special character handling:\n")
    for text, description in test_cases:
        result = sanitize_for_voice(text)
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print(f"        {description}")
        print()


def test_chunk_for_kokoro_ultra_low_latency():
    """Test ultra-low latency text chunking for Kokoro TTS."""
    test_cases = [
        # (input, max_chars, description)
        ("Hello world", 25, "Simple short text"),
        ("This is a test sentence that is quite long", 25, "Long sentence splits at clauses"),
        ("Short. Another. More.", 25, "Multiple short sentences"),
        ("The quick brown fox jumps over the lazy dog and runs fast", 25, "Long sentence with conjunctions"),
        ("", 25, "Empty text returns empty list"),
        ("A" * 100, 25, "Very long text without breaks"),
        ("Hello, world! How are you today?", 25, "Text with punctuation"),
    ]

    print("\nTesting ultra-low latency text chunking:\n")
    for text, max_chars, description in test_cases:
        result = chunk_for_kokoro_ultra_low_latency(text, max_chars)
        print(f"Input:  {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"Max chars: {max_chars}")
        print(f"Chunks: {len(result)}")
        for i, chunk in enumerate(result):
            print(f"  [{i+1}] ({len(chunk)} chars): {chunk}")
        print(f"Description: {description}")

        # Validation
        for chunk in result:
            if len(chunk) > max_chars * 1.2:  # Allow 20% overflow for edge cases
                print(f"  ⚠️  WARNING: Chunk exceeds max_chars: {len(chunk)} > {max_chars}")

        if not result and text.strip():
            print(f"  ⚠️  WARNING: No chunks produced from non-empty text")

        print()


def test_chunk_for_kokoro_edge_cases():
    """Test edge cases for chunking."""
    # Test with default max_chars
    result = chunk_for_kokoro_ultra_low_latency("This is a test")
    assert len(result) > 0, "Should produce chunks for normal text"

    # Test empty text
    result = chunk_for_kokoro_ultra_low_latency("")
    assert result == [], "Empty text should return empty list"

    # Test whitespace only
    result = chunk_for_kokoro_ultra_low_latency("   ")
    assert result == [], "Whitespace-only text should return empty list"

    # Test very short text (under max_chars)
    result = chunk_for_kokoro_ultra_low_latency("Hi", max_chars=25)
    assert len(result) == 1, "Short text should produce single chunk"
    assert result[0] == "Hi", "Short text should be preserved exactly"

    # Test text with only punctuation
    result = chunk_for_kokoro_ultra_low_latency("...", max_chars=25)
    assert result == [], "Punctuation-only should return empty list"

    print("✅ All edge case tests passed")


if __name__ == "__main__":
    test_contractions()
    test_special_chars()
    test_chunk_for_kokoro_ultra_low_latency()
    test_chunk_for_kokoro_edge_cases()