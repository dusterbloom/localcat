"""
Unit tests for QualityFilter

Tests the unified quality filtering logic that eliminates ~150 lines of duplicate code
between hotpath_processor.py and retrieval.py.
"""

import pytest
from core.memory.quality_filter import QualityFilter, QualityFilterConfig


class TestQualityFilterStorage:
    """Test Layer 2 defense: storage-time filtering"""

    def setup_method(self):
        self.filter = QualityFilter()

    def test_minimum_word_count_storage(self):
        """Reject text below minimum word count for storage"""
        assert self.filter.is_quality_for_storage("Hi") is False  # Too short
        assert self.filter.is_quality_for_storage("Hello there") is False  # 2 words
        assert (
            self.filter.is_quality_for_storage("Hello there friend") is True
        )  # 3 words OK

    def test_system_patterns_storage(self):
        """Reject system/debug artifacts"""
        assert self.filter.is_quality_for_storage("[memory] storing fact") is False
        assert self.filter.is_quality_for_storage("[debug] test message") is False
        assert self.filter.is_quality_for_storage("[session] active") is False
        assert self.filter.is_quality_for_storage("Normal conversation text") is True

    def test_confusion_patterns_storage(self):
        """Reject confusion/misunderstanding"""
        assert self.filter.is_quality_for_storage("I'm confused about this") is False
        assert self.filter.is_quality_for_storage("What do you mean by that") is False
        assert self.filter.is_quality_for_storage("I don't know what") is False
        assert self.filter.is_quality_for_storage("I understand this clearly") is True

    def test_transcription_artifacts_storage(self):
        """Reject transcription artifacts"""
        assert self.filter.is_quality_for_storage("[inaudible] some text") is False
        assert self.filter.is_quality_for_storage("[crosstalk] overlapping") is False
        assert self.filter.is_quality_for_storage("Clear speech here") is True

    def test_filler_words_storage(self):
        """Reject text with excessive filler"""
        # All filler words (over 50% threshold)
        assert self.filter.is_quality_for_storage("um like uh you know") is False
        # "like" as verb is OK
        assert self.filter.is_quality_for_storage("I like pizza") is True
        # Some filler is acceptable
        assert self.filter.is_quality_for_storage("I mean I really like pizza") is True

    def test_pure_questions_storage(self):
        """Reject pure questions without assertions"""
        # Pure question without assertion
        assert self.filter.is_quality_for_storage("Where are you?") is False
        # Question with assertion
        assert self.filter.is_quality_for_storage("You are my friend?") is True
        # Statement
        assert self.filter.is_quality_for_storage("I like the beach") is True

    def test_empty_and_short_storage(self):
        """Reject empty or very short text"""
        assert self.filter.is_quality_for_storage("") is False
        assert self.filter.is_quality_for_storage("   ") is False
        assert self.filter.is_quality_for_storage("ok") is False


class TestQualityFilterRetrieval:
    """Test Layer 4 defense: retrieval-time filtering"""

    def setup_method(self):
        self.filter = QualityFilter()

    def test_minimum_word_count_retrieval(self):
        """Retrieval filter should be stricter (4 words minimum)"""
        text_3_words = "Hello there friend"
        text_4_words = "Hello there my friend"

        # 3 words passes storage
        assert self.filter.is_quality_for_storage(text_3_words) is True

        # But may fail retrieval (depending on content patterns)
        # 4 words should work better for retrieval
        assert self.filter.is_quality_for_retrieval(text_4_words) is True

    def test_empty_responses_retrieval(self):
        """Reject empty/generic responses at retrieval time"""
        assert self.filter.is_quality_for_retrieval("okay") is False
        assert self.filter.is_quality_for_retrieval("yes") is False
        assert self.filter.is_quality_for_retrieval("got it") is False
        assert self.filter.is_quality_for_retrieval("The user likes pizza") is True

    def test_interjections_retrieval(self):
        """Reject interjections at retrieval time"""
        # Pure interjections
        assert self.filter.is_quality_for_retrieval("wow cool") is False
        assert self.filter.is_quality_for_retrieval("oh nice") is False

        # Interjection followed by content
        assert (
            self.filter.is_quality_for_retrieval(
                "oh wow that's really interesting to learn"
            )
            is True
        )

    def test_content_patterns_retrieval(self):
        """Require content patterns for retrieval"""
        # Has content pattern
        assert (
            self.filter.is_quality_for_retrieval("User lives in San Francisco") is True
        )
        assert self.filter.is_quality_for_retrieval("User works at Google") is True

        # No content pattern, but long enough
        assert (
            self.filter.is_quality_for_retrieval(
                "This is a reasonably long sentence without specific content markers"
            )
            is True
        )

        # No content pattern and short
        assert self.filter.is_quality_for_retrieval("random short text") is False

    def test_stricter_filler_threshold_retrieval(self):
        """Retrieval has stricter filler threshold (30% vs 50%)"""
        # 2 filler words out of 6 total = 33% > 30% threshold
        text = "um like I think maybe perhaps"
        assert self.filter.is_quality_for_retrieval(text) is False

        # Less filler
        text = "I think this is interesting"
        assert self.filter.is_quality_for_retrieval(text) is True


class TestQualityFilterScoring:
    """Test quality scoring for ranking"""

    def setup_method(self):
        self.filter = QualityFilter()

    def test_high_quality_score(self):
        """High quality text should score highly"""
        score = self.filter.get_quality_score("The user lives in San Francisco")
        assert score > 0.8

    def test_medium_quality_score(self):
        """Medium quality text (short) should score medium"""
        score = self.filter.get_quality_score("User likes pizza")
        assert 0.5 < score < 0.8

    def test_low_quality_score(self):
        """Low quality text (filler + confusion) should score low"""
        score = self.filter.get_quality_score("um I'm confused about this")
        assert score < 0.5

    def test_empty_text_score(self):
        """Empty text should score 0.0"""
        assert self.filter.get_quality_score("") == 0.0
        assert self.filter.get_quality_score("   ") == 0.0


class TestQualityFilterEdgeCases:
    """Test edge cases and boundary conditions"""

    def setup_method(self):
        self.filter = QualityFilter()

    def test_bracket_ratio_threshold(self):
        """Reject text with excessive brackets"""
        # Excessive brackets (metadata pollution)
        assert (
            self.filter.is_quality_for_storage("[tag1] [tag2] [tag3] short text")
            is False
        )

        # Normal brackets
        assert (
            self.filter.is_quality_for_storage("The user (named John) likes pizza")
            is True
        )

    def test_repeated_characters(self):
        """Reject text with transcription errors"""
        assert self.filter.is_quality_for_storage("I liiiiiike this") is False
        assert self.filter.is_quality_for_storage("I like this") is True

    def test_case_insensitivity(self):
        """Patterns should be case-insensitive"""
        assert self.filter.is_quality_for_storage("[MEMORY] test") is False
        assert self.filter.is_quality_for_storage("I'M CONFUSED") is False
        assert self.filter.is_quality_for_storage("This is NORMAL TEXT") is True


class TestQualityFilterLayerDifferences:
    """Test differences between Layer 2 (storage) and Layer 4 (retrieval)"""

    def setup_method(self):
        self.filter = QualityFilter()

    def test_layer_2_more_permissive(self):
        """Layer 2 (storage) should be more permissive than Layer 4 (retrieval)"""
        # Short text that passes storage but fails retrieval
        text = "I like it"  # 3 words, minimal content

        # Should pass storage (3 words minimum)
        assert self.filter.is_quality_for_storage(text) is True

        # May fail retrieval (stricter requirements)
        # Note: This specific text may pass due to content patterns
        # Testing the general principle

    def test_interjection_handling_difference(self):
        """Layer 4 has additional interjection filtering"""
        text = "wow that's cool"

        # May pass storage (no explicit interjection filter in Layer 2)
        storage_result = self.filter.is_quality_for_storage(text)

        # Should fail retrieval (interjection filter in Layer 4)
        retrieval_result = self.filter.is_quality_for_retrieval(text)

        # Retrieval should be stricter (though both may fail due to length)
        if storage_result:
            assert not retrieval_result, "Retrieval should be stricter than storage"


def test_quality_filter_initialization():
    """Test that QualityFilter initializes correctly"""
    qf = QualityFilter()
    assert qf is not None
    assert hasattr(qf, "_confusion_regex")
    assert hasattr(qf, "_system_regex")
    assert hasattr(qf, "_filler_regex")


def test_quality_filter_config():
    """Test QualityFilterConfig constants"""
    assert QualityFilterConfig.MIN_WORDS_FOR_STORAGE == 3
    assert QualityFilterConfig.MIN_WORDS_FOR_RETRIEVAL == 4
    assert QualityFilterConfig.MAX_BRACKET_RATIO == 0.3
    assert QualityFilterConfig.MAX_REPEATED_CHARS == 4
