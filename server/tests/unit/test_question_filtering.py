"""
Unit tests for prosody-based question filtering.

Tests that questions are:
1. Detected by prosody pitch slope
2. Skipped during extraction (no triples generated)
3. Filtered from conversation retrieval (don't appear in memory bullets)
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# Import the classes we're testing
from core.memory.memory_hotpath import HotMemory
from core.memory.retrieval import Retrieval, Candidate
from core.memory.memory_store import MemoryStore, Paths


@dataclass
class MockProsodyFeatures:
    """Mock prosody features for testing."""
    pitch_mean: float = 250.0
    pitch_std: float = 20.0
    pitch_slope: float = 0.0  # This is what we'll vary for question detection
    intensity_mean: float = 50.0
    intensity_peak: float = 60.0
    speaking_rate: float = 3.0
    pause_count: int = 0
    duration_sec: float = 2.0
    certainty_modifier: float = 0.0


class TestQuestionDetection:
    """Test question detection using prosody."""

    @pytest.fixture
    def hot_memory(self):
        """Create HotMemory instance for testing."""
        store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
        hot = HotMemory(store)
        return hot

    @pytest.fixture
    def retrieval(self):
        """Create Retrieval instance for testing."""
        host = Mock()
        host.entity_index = {}
        host.recency_buffer = []
        host.store = Mock()
        host.current_user_id = "test_user"
        host.current_session_id = "test_session"
        return Retrieval(host)

    def test_prosody_detects_rising_intonation(self, hot_memory):
        """Test that positive pitch slope is detected as a question."""
        # Question: Rising intonation (positive slope)
        question_prosody = MockProsodyFeatures(pitch_slope=35.0)
        assert hot_memory._is_question_from_prosody(question_prosody, "test")

    def test_prosody_detects_falling_intonation(self, hot_memory):
        """Test that negative pitch slope is NOT detected as a question."""
        # Statement: Falling intonation (negative slope)
        statement_prosody = MockProsodyFeatures(pitch_slope=-25.0)
        assert not hot_memory._is_question_from_prosody(statement_prosody, "test")

    def test_prosody_threshold_configurable(self, hot_memory, monkeypatch):
        """Test that question detection threshold is configurable."""
        # Set custom threshold
        monkeypatch.setenv("PROSODY_QUESTION_SLOPE_THRESHOLD", "15")

        # Below threshold (not a question)
        low_prosody = MockProsodyFeatures(pitch_slope=10.0)
        assert not hot_memory._is_question_from_prosody(low_prosody, "test")

        # Above threshold (is a question)
        high_prosody = MockProsodyFeatures(pitch_slope=20.0)
        assert hot_memory._is_question_from_prosody(high_prosody, "test")

    def test_text_fallback_question_mark(self, hot_memory):
        """Test text-based fallback detects question mark."""
        # No prosody, but has question mark
        assert hot_memory._is_question_from_prosody(None, "Is this a question?")

    def test_text_fallback_question_words(self, hot_memory):
        """Test text-based fallback detects question words."""
        questions = [
            "Do you know my color?",
            "What is your name?",
            "Where are you from?",
            "Can you help me?",
            "Why did that happen?",
        ]

        for q in questions:
            assert hot_memory._is_question_from_prosody(None, q), f"Failed to detect: {q}"

    def test_statements_not_detected(self, hot_memory):
        """Test that statements are not detected as questions."""
        statements = [
            "My favorite color is yellow.",
            "I like to read books.",
            "The weather is nice today.",
            "Thank you for your help.",
        ]

        statement_prosody = MockProsodyFeatures(pitch_slope=-20.0)

        for s in statements:
            assert not hot_memory._is_question_from_prosody(statement_prosody, s), f"Incorrectly detected as question: {s}"

    def test_no_extraction_from_questions(self, hot_memory):
        """Test that questions don't produce triples during extraction."""
        # Question with positive pitch slope
        question_prosody = MockProsodyFeatures(pitch_slope=30.0)

        bullets, triples = hot_memory.process_turn(
            text="Do you know my favorite color?",
            session_id="test_session",
            turn_id=1,
            prosody_features=question_prosody
        )

        # Should not extract any triples from the question
        assert len(triples) == 0, f"Question should not produce triples, got: {triples}"

    def test_extraction_from_statements(self, hot_memory):
        """Test that statements DO produce triples during extraction."""
        # Statement with negative pitch slope
        statement_prosody = MockProsodyFeatures(pitch_slope=-25.0)

        bullets, triples = hot_memory.process_turn(
            text="My favorite color is yellow.",
            session_id="test_session",
            turn_id=1,
            prosody_features=statement_prosody
        )

        # Should extract triples from the statement
        assert len(triples) > 0, "Statement should produce triples"

    def test_retrieval_filters_questions(self, retrieval):
        """Test that questions are filtered from retrieval candidates."""
        # Test the text-based question detection in retrieval
        assert retrieval._is_text_question("Do you know my color?")
        assert retrieval._is_text_question("What is your name?")
        assert not retrieval._is_text_question("My favorite color is yellow.")

    def test_question_mark_detection(self, retrieval):
        """Test that question marks are detected."""
        assert retrieval._is_text_question("Is this a question?")
        assert retrieval._is_text_question("Really?")
        assert not retrieval._is_text_question("I said yes.")

    def test_question_starters_detection(self, retrieval):
        """Test that question word starters are detected."""
        question_starters = [
            "do you", "did you", "can you", "could you", "would you",
            "what", "when", "where", "who", "why", "how", "which"
        ]

        for starter in question_starters:
            test_text = f"{starter} test question"
            assert retrieval._is_text_question(test_text), f"Failed to detect: {test_text}"


class TestIntegration:
    """Integration tests for end-to-end question filtering."""

    @pytest.fixture
    def hot_memory(self):
        """Create HotMemory instance for integration testing."""
        store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
        hot = HotMemory(store)
        return hot

    def test_scenario_favorite_color(self, hot_memory):
        """
        Test the actual scenario from the logs:
        1. User states: "My favorite color is yellow" (statement)
        2. User asks: "Do you know my favorite color?" (question)
        3. Verify: Question doesn't create incomplete triples
        """
        # Statement with falling intonation
        statement_prosody = MockProsodyFeatures(pitch_slope=-25.0)
        bullets1, triples1 = hot_memory.process_turn(
            text="My favorite color is yellow",
            session_id="test",
            turn_id=1,
            prosody_features=statement_prosody
        )

        # Should extract fact from statement
        assert len(triples1) > 0, "Should extract triples from statement"
        # Check if we got a complete triple with 'yellow'
        has_color_triple = any('yellow' in str(t).lower() for t in triples1)
        assert has_color_triple, f"Should extract 'yellow' triple, got: {triples1}"

        # Question with rising intonation
        question_prosody = MockProsodyFeatures(pitch_slope=35.0)
        bullets2, triples2 = hot_memory.process_turn(
            text="Do you know my favorite color?",
            session_id="test",
            turn_id=2,
            prosody_features=question_prosody
        )

        # Should NOT extract triples from question
        assert len(triples2) == 0, f"Should not extract triples from question, got: {triples2}"

    def test_no_incomplete_triples(self, hot_memory):
        """Test that incomplete triples like (you, has, favorite_color) are not created."""
        question_prosody = MockProsodyFeatures(pitch_slope=30.0)

        bullets, triples = hot_memory.process_turn(
            text="Do you have a favorite color?",
            session_id="test",
            turn_id=1,
            prosody_features=question_prosody
        )

        # No triples should be extracted
        assert len(triples) == 0, "Question should not produce any triples"

        # Verify no incomplete triples like (you, has, favorite_color)
        for triple in triples:
            if len(triple) >= 3:
                s, r, o = triple[0], triple[1], triple[2]
                # Check this isn't an incomplete triple missing the object value
                assert o and len(o) > 2, f"Incomplete triple detected: {triple}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
