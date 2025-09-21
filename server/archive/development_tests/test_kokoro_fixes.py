#!/usr/bin/env python3
"""Test script for Kokoro TTS fixes with problematic texts."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from tools.text_formatter import sanitize_for_kokoro, smart_sentence_split


def test_problematic_texts():
    """Test the enhanced sanitization with known problematic texts."""

    # The problematic text from the logs
    problematic_text = 'That\'s an interesting question! As your assistant, I can access information to help you, but I don\'t have direct "internet" access in the way a web browser does. I can still look things up for you. Would you like me to do that?'

    print("=== Testing Problematic Text ===")
    print(f"Original: {problematic_text}")
    print(f"Length: {len(problematic_text)} chars")
    print()

    # Test the enhanced sanitization
    sanitized = sanitize_for_kokoro(problematic_text, max_sentence_length=100)
    print(f"Sanitized: {sanitized}")
    print(f"Length: {len(sanitized)} chars")
    print()

    # Show sentence splitting
    sentences = smart_sentence_split(sanitized, max_length=100)
    print("=== Sentence Splitting ===")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. [{len(sentence)} chars] {sentence}")
    print()

    # Test other challenging cases
    test_cases = [
        'I can help with "complex queries" and i.e. technical questions.',
        'This is a very long sentence that contains multiple clauses, coordinating conjunctions like and, but, or, and should be split intelligently at natural break points to ensure optimal TTS performance.',
        'Short sentence.',
        'Text with (parenthetical expressions) and some — em dashes.',
        'Question marks work? Exclamations too! And periods.',
    ]

    print("=== Additional Test Cases ===")
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text}")
        sanitized = sanitize_for_kokoro(test_text)
        sentences = smart_sentence_split(sanitized, max_length=80)
        for j, sentence in enumerate(sentences, 1):
            print(f"  {j}. [{len(sentence)} chars] {sentence}")


if __name__ == "__main__":
    test_problematic_texts()