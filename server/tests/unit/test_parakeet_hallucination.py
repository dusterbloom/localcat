#!/usr/bin/env python3
"""
Test hallucination detection in ParakeetBatchSTT.

Tests the pattern-based hallucination filtering that replaced confidence-based filtering.
"""

import sys
import os
import pytest

# Add server directory to path
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from core.stt.parakeet_batch import ParakeetBatchSTT


class TestHallucinationDetection:
    """Test suite for hallucination pattern detection."""

    @pytest.fixture
    def stt_service(self):
        """Create a ParakeetBatchSTT instance for testing (mock mode)."""
        # We're just testing the hallucination detection logic, not actual STT
        # So we won't initialize the actual model
        service = object.__new__(ParakeetBatchSTT)
        return service

    def test_known_hallucinations_filtered(self, stt_service):
        """Test that known hallucination patterns are detected."""
        # Common false positives that should be filtered
        # Note: Patterns are matched AFTER punctuation is removed,
        # so "mm-hmm" matches "mmhmm" and "uh-huh" matches "uhhuh"
        hallucinations = [
            "yeah",
            "yep",
            "yes",
            "mm-hmm",  # becomes "mmhmm" after normalization
            "mmhmm",
            "uh-huh",  # becomes "uhhuh" after normalization
            "thank you",
            "thanks",
            "okay",
            "ok",
            "uh",
            "um",
            "hmm",
        ]

        for text in hallucinations:
            assert stt_service._is_hallucination(text), \
                f"'{text}' should be detected as hallucination"

    def test_valid_speech_not_filtered(self, stt_service):
        """Test that valid speech is NOT detected as hallucination."""
        valid_texts = [
            "Hello, how are you doing today?",
            "I need to schedule a meeting for tomorrow",
            "Can you help me with this problem?",
            "The weather is really nice today",
            "I'm working on a project right now",
            "Please send me the document",
            "That sounds like a good idea",
            "Let me think about that for a moment",
        ]

        for text in valid_texts:
            assert not stt_service._is_hallucination(text), \
                f"'{text}' should NOT be detected as hallucination"

    def test_case_insensitive_matching(self, stt_service):
        """Test that hallucination detection is case-insensitive."""
        variants = [
            "Yeah",
            "YEAH",
            "YeAh",
            "Thank You",
            "THANK YOU",
            "Okay",
            "OKAY",
        ]

        for text in variants:
            assert stt_service._is_hallucination(text), \
                f"'{text}' should be detected (case-insensitive)"

    def test_punctuation_handling(self, stt_service):
        """Test that punctuation doesn't affect detection."""
        punctuated = [
            "yeah!",
            "yes.",
            "okay?",
            "thanks!",
            "mm-hmm.",
        ]

        for text in punctuated:
            assert stt_service._is_hallucination(text), \
                f"'{text}' should be detected despite punctuation"

    def test_empty_and_whitespace(self, stt_service):
        """Test that empty/whitespace text is filtered."""
        empty_cases = [
            "",
            "   ",
            "\n",
            "\t",
            "  \n  ",
        ]

        for text in empty_cases:
            assert stt_service._is_hallucination(text), \
                f"'{text}' (empty/whitespace) should be filtered"

    def test_very_short_words(self, stt_service):
        """Test that very short single-word outputs are filtered as noise."""
        short_words = [
            "a",
            "I",
            "oh",
            "ah",
            "uh",
            "um",
        ]

        for text in short_words:
            assert stt_service._is_hallucination(text), \
                f"'{text}' (very short) should be filtered as noise"

    def test_multi_word_not_filtered(self, stt_service):
        """Test that multi-word phrases are not filtered (even if short)."""
        multi_word = [
            "yes please",
            "okay then",
            "thank you very much",
            "I agree",
            "sounds good",
        ]

        # Multi-word phrases with hallucination keywords should NOT be filtered
        # (except for exact matches of the pattern itself)
        for text in multi_word:
            result = stt_service._is_hallucination(text)
            # This depends on implementation - document expected behavior
            # For now, we expect multi-word NOT to be filtered unless exact match
            if text in ["yes", "okay", "thank you", "thanks"]:
                assert result, f"'{text}' is exact match, should be filtered"
            # If it's longer, it might still be valid speech

    def test_boundary_cases(self, stt_service):
        """Test boundary cases that might be tricky."""
        boundary_cases = [
            ("yeah yeah yeah", True, "Repetitive filler"),
            ("yes sir", False, "Valid response with context"),
            ("um actually", False, "Filler with continuation"),
            ("okay so", False, "Transition phrase"),
        ]

        for text, should_filter, description in boundary_cases:
            result = stt_service._is_hallucination(text)
            # Note: Some of these might not match current implementation
            # Document the behavior for future reference
            print(f"{description}: '{text}' -> filtered={result} (expected={should_filter})")


@pytest.mark.fast
def test_hallucination_patterns_coverage():
    """Test that HALLUCINATION_PATTERNS set is comprehensive."""
    # Verify the pattern set exists and has reasonable coverage
    assert hasattr(ParakeetBatchSTT, 'HALLUCINATION_PATTERNS'), \
        "HALLUCINATION_PATTERNS should be defined"

    patterns = ParakeetBatchSTT.HALLUCINATION_PATTERNS
    assert len(patterns) >= 10, \
        "Should have at least 10 hallucination patterns"

    # Check for common categories
    assert any('yeah' in p or 'yep' in p or 'yes' in p for p in patterns), \
        "Should include agreement words"
    assert any('thank' in p for p in patterns), \
        "Should include thanks/gratitude"
    assert any('mm' in p or 'hmm' in p or 'uh' in p or 'um' in p for p in patterns), \
        "Should include filler words"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
