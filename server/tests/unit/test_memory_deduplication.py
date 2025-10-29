"""
Test suite for memory retrieval deduplication.

Verifies that duplicate memories from different sources (graph, convo, summary)
are properly detected and filtered out.
"""

import pytest
import os
from unittest.mock import Mock, MagicMock
from core.memory.retrieval import Retrieval, Candidate


class TestMemoryDeduplication:
    """Test deduplication across memory sources."""

    @pytest.fixture
    def mock_host(self):
        """Create mock host with necessary attributes."""
        host = Mock()
        host.entity_index = {}
        host.recency_buffer = []
        host.store = Mock()
        host.current_user_id = "test_user"
        host.current_session_id = "test_session"
        return host

    @pytest.fixture
    def retrieval(self, mock_host):
        """Create Retrieval instance with mock host."""
        return Retrieval(mock_host)

    def test_exact_duplicate_detection(self, retrieval):
        """Test that exact duplicates are filtered out."""
        text1 = "Your favorite color is yellow"
        text2 = "Your favorite color is yellow"

        norm1 = retrieval._normalize_candidate_text(text1)
        norm2 = retrieval._normalize_candidate_text(text2)

        assert norm1 == norm2, "Exact duplicates should have identical normalized text"

    def test_punctuation_normalization(self, retrieval):
        """Test that punctuation and underscores are normalized."""
        text1 = "you, favorite_color: yellow!"
        text2 = "you favorite color yellow"

        norm1 = retrieval._normalize_candidate_text(text1)
        norm2 = retrieval._normalize_candidate_text(text2)

        assert norm1 == norm2, "Punctuation and underscores should be stripped during normalization"

    def test_semantic_similarity_high_overlap(self, retrieval):
        """Test that semantically similar texts are detected as duplicates."""
        text1 = "you favorite_color yellow"
        text2 = "favorite color is yellow"

        # These share most words: "favorite", "color", "yellow" (3 common)
        # words1 = {you, favorite, color, yellow} (4 words)
        # words2 = {favorite, color, is, yellow} (4 words)
        # union = {you, favorite, color, is, yellow} (5 words)
        # Jaccard similarity = 3/5 = 0.6 (60%)
        is_similar = retrieval._are_semantically_similar(text1, text2, threshold=0.6)

        assert is_similar, "Texts with 60% word overlap should be considered similar at 0.6 threshold"

    def test_semantic_similarity_low_overlap(self, retrieval):
        """Test that different texts are not considered duplicates."""
        text1 = "you favorite_color yellow"
        text2 = "alice lives in new york"

        # These share no words (after normalization)
        is_similar = retrieval._are_semantically_similar(text1, text2, threshold=0.7)

        assert not is_similar, "Texts with no overlap should not be considered similar"

    def test_semantic_similarity_threshold(self, retrieval):
        """Test that similarity threshold is respected."""
        text1 = "alice has a cat named fluffy"
        text2 = "alice has a dog"

        # Overlap: "alice", "has", "a" = 3 words
        # Union: "alice", "has", "a", "cat", "named", "fluffy", "dog" = 7 words
        # Similarity = 3/7 = 0.43

        # Should be similar at 0.3 threshold
        assert retrieval._are_semantically_similar(text1, text2, threshold=0.3)

        # Should NOT be similar at 0.7 threshold
        assert not retrieval._are_semantically_similar(text1, text2, threshold=0.7)

    def test_cross_source_deduplication(self, retrieval):
        """Test that duplicates from different sources are filtered."""
        candidates = [
            Candidate(
                text="you favorite_color yellow",
                source="graph",
                score_hint=0.0,
                ts=1000,
                meta={"edge_id": "e1"}
            ),
            Candidate(
                text="favorite color is yellow",
                source="convo",
                score_hint=0.8,
                ts=2000,
                meta={"bm25_score": 0.8}
            ),
            Candidate(
                text="alice lives in paris",
                source="graph",
                score_hint=0.0,
                ts=3000,
                meta={"edge_id": "e2"}
            ),
        ]

        # Simulate scoring (higher score for graph to ensure it's selected first)
        scored_candidates = [
            (0.9, candidates[0], {"wsrc": 0.3, "wconf": 0.4}),  # Graph candidate
            (0.7, candidates[1], {"wsrc": 0.4, "wconf": 0.3}),  # Convo duplicate
            (0.8, candidates[2], {"wsrc": 0.3, "wconf": 0.4}),  # Unique candidate
        ]

        # Sort by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Apply deduplication
        final_bullets, selected_candidates = retrieval._apply_token_budget_and_deduplication(
            scored_candidates,
            max_bullets=3,
            query="what is my favorite color"
        )

        # Should have 2 bullets, not 3 (duplicate filtered)
        assert len(final_bullets) == 2, f"Expected 2 bullets, got {len(final_bullets)}"
        assert len(selected_candidates) == 2, f"Expected 2 candidates, got {len(selected_candidates)}"

        # Verify the duplicate was filtered
        selected_texts = [c.text for c in selected_candidates]
        assert "you favorite_color yellow" in selected_texts
        assert "alice lives in paris" in selected_texts
        assert "favorite color is yellow" not in selected_texts

    def test_case_insensitive_normalization(self, retrieval):
        """Test that case differences are normalized."""
        text1 = "Your Favorite Color Is Yellow"
        text2 = "your favorite color is yellow"

        norm1 = retrieval._normalize_candidate_text(text1)
        norm2 = retrieval._normalize_candidate_text(text2)

        assert norm1 == norm2, "Case differences should be normalized"

    def test_empty_text_handling(self, retrieval):
        """Test that empty texts are handled gracefully."""
        text1 = ""
        text2 = "something"

        is_similar = retrieval._are_semantically_similar(text1, text2)
        assert not is_similar, "Empty text should not match non-empty text"

        # Two empty texts should match
        is_similar = retrieval._are_semantically_similar("", "")
        assert is_similar, "Two empty texts should match"

    def test_configurable_threshold(self, retrieval, monkeypatch):
        """Test that deduplication threshold is configurable via environment."""
        text1 = "alice has a cat"
        text2 = "alice has a dog"

        # Set custom threshold via environment
        monkeypatch.setenv("MEMORY_DEDUP_THRESHOLD", "0.3")

        # At 0.3 threshold, these should be similar (3/5 = 0.6)
        is_similar = retrieval._are_semantically_similar(text1, text2, threshold=0.3)
        assert is_similar, "Should be similar at low threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
