#!/usr/bin/env python3
"""Test the text formatter to ensure contractions are handled correctly."""

import sys
import os
# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from text_formatter import sanitize_for_voice


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


if __name__ == "__main__":
    test_contractions()
    test_special_chars()