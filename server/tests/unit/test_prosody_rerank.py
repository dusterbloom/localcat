#!/usr/bin/env python3
"""Unit tests for prosody reranking (wpro component) in retrieval."""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.retrieval import Retrieval, Candidate
from core.memory.memory_store import MemoryStore, Paths


@pytest.fixture
def memory_store():
    """Create a temporary MemoryStore for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        paths = Paths(sqlite_path=db_path, lmdb_dir=None)  # Disable LMDB for simplicity
        store = MemoryStore(paths)
        yield store
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def mock_host(memory_store):
    """Create a mock host with necessary attributes."""
    host = Mock()
    host.store = memory_store
    host.entity_index = {}
    host.recency_buffer = []
    host.current_session_id = "test_session"
    host.current_user_id = "test_user"
    return host


@pytest.fixture
def retrieval(mock_host):
    """Create a Retrieval instance with mock host."""
    return Retrieval(mock_host)


class TestProsodyReranking:
    """Test prosody-aware reranking for convo candidates."""

    def test_wpro_component_for_convo_candidates(self, retrieval, memory_store):
        """Test that wpro component is added for convo candidates."""
        # Store prosody data
        memory_store.set_turn_prosody("test_session", 1, 0.8)  # High certainty
        memory_store.set_turn_prosody("test_session", 2, 0.2)  # Low certainty
        
        # Create convo candidates
        high_certainty_candidate = Candidate(
            text="User said something important",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        low_certainty_candidate = Candidate(
            text="User said something else",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 2}
        )
        
        # Test composite scoring with prosody weight enabled
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.2"}):
            total_score1, components1 = retrieval._composite_score("test query", high_certainty_candidate)
            total_score2, components2 = retrieval._composite_score("test query", low_certainty_candidate)
            
            # Both should have wpro component
            assert "wpro" in components1
            assert "wpro" in components2
            
            # High certainty should have higher wpro score
            assert components1["wpro"] > components2["wpro"]
            
            # Verify wpro values (certainty * weight)
            assert abs(components1["wpro"] - (0.8 * 0.2)) < 0.001
            assert abs(components2["wpro"] - (0.2 * 0.2)) < 0.001

    def test_no_wpro_for_non_convo_sources(self, retrieval):
        """Test that wpro component is not added for non-convo sources."""
        graph_candidate = Candidate(
            text="Alice lives in NYC",
            source="graph",
            score_hint=0.0,
            ts=1000,
            meta={"edge_id": "test_edge"}
        )
        
        summary_candidate = Candidate(
            text="Summary of recent conversations",
            source="summary",
            score_hint=0.5,
            ts=1000,
            meta={}
        )
        
        semantic_candidate = Candidate(
            text="Semantic search result",
            source="semantic",
            score_hint=0.7,
            ts=1000,
            meta={"similarity_score": 0.7}
        )
        
        # Test with prosody weight enabled
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.3"}):
            for candidate in [graph_candidate, summary_candidate, semantic_candidate]:
                total_score, components = retrieval._composite_score("test query", candidate)
                
                # Should not have wpro component
                assert "wpro" not in components or components["wpro"] == 0.0

    def test_wpro_zero_when_prosody_weight_disabled(self, retrieval, memory_store):
        """Test that wpro is zero when MEMORY_WEIGHT_PROSODY is 0.0."""
        # Store prosody data
        memory_store.set_turn_prosody("test_session", 1, 0.9)
        
        convo_candidate = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Test with prosody weight disabled
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.0"}):
            total_score, components = retrieval._composite_score("test query", convo_candidate)
            
            # wpro should be zero
            assert "wpro" in components
            assert components["wpro"] == 0.0

    def test_missing_session_id_or_turn_id_returns_zero(self, retrieval):
        """Test that missing session_id or turn_id returns wpro = 0.0."""
        # Candidate with missing turn_id
        candidate_no_turn_id = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": None}
        )
        
        # Test with prosody weight enabled
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.2"}):
            total_score, components = retrieval._composite_score("test query", candidate_no_turn_id)
            
            # wpro should be zero
            assert "wpro" in components
            assert components["wpro"] == 0.0
        
        # Test with missing session_id in host
        retrieval.host.current_session_id = None
        
        candidate_with_turn_id = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        total_score, components = retrieval._composite_score("test query", candidate_with_turn_id)
        
        # wpro should be zero
        assert "wpro" in components
        assert components["wpro"] == 0.0

    def test_prosody_cache_prevents_repeated_store_calls(self, retrieval, memory_store):
        """Test that prosody cache prevents repeated store hits."""
        # Store prosody data
        memory_store.set_turn_prosody("test_session", 1, 0.7)
        
        convo_candidate = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Spy on store.get_turn_prosody
        original_method = memory_store.get_turn_prosody
        call_count = 0
        
        def counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_method(*args, **kwargs)
        
        memory_store.get_turn_prosody = counting_wrapper
        
        # Test with prosody weight enabled
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.2"}):
            # First call should hit store
            total_score1, components1 = retrieval._composite_score("test query", convo_candidate)
            first_call_count = call_count
            
            # Second call for same candidate should use cache
            total_score2, components2 = retrieval._composite_score("test query", convo_candidate)
            second_call_count = call_count
            
            # Store should only be called once due to caching
            assert first_call_count == 1
            assert second_call_count == 1  # No additional calls
            
            # Results should be identical
            assert components1["wpro"] == components2["wpro"]

    def test_prosody_cache_cleared_per_retrieve(self, retrieval, memory_store):
        """Test that prosody cache is cleared per retrieve call."""
        # Store prosody data
        memory_store.set_turn_prosody("test_session", 1, 0.6)
        
        convo_candidate = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Simulate multiple retrieve calls by manually clearing and calling
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.2"}):
            # First simulate retrieve() - cache should be empty, then populated
            retrieval._prosody_cache.clear()
            total_score1, components1 = retrieval._composite_score("test query", convo_candidate)
            
            # Cache should have the result
            assert len(retrieval._prosody_cache) == 1
            assert ("test_session", 1) in retrieval._prosody_cache
            
            # Simulate another retrieve() - cache should be cleared again
            retrieval._prosody_cache.clear()
            assert len(retrieval._prosody_cache) == 0
            
            # Call again - should repopulate cache
            total_score2, components2 = retrieval._composite_score("test query", convo_candidate)
            
            # Cache should be repopulated
            assert len(retrieval._prosody_cache) == 1

    def test_store_exception_handling(self, retrieval):
        """Test that store exceptions are handled gracefully."""
        convo_candidate = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Mock store to raise exception
        def failing_get_turn_prosody(*args, **kwargs):
            raise Exception("Database error")
        
        retrieval.host.store.get_turn_prosody = failing_get_turn_prosody
        
        # Test with prosody weight enabled
        with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": "0.2"}):
            # Should not raise exception
            total_score, components = retrieval._composite_score("test query", convo_candidate)
            
            # wpro should be zero due to exception
            assert "wpro" in components
            assert components["wpro"] == 0.0

    def test_default_prosody_weight(self, retrieval):
        """Test that default prosody weight is 0.0."""
        convo_candidate = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Test with default environment (no MEMORY_WEIGHT_PROSODY set)
        total_score, components = retrieval._composite_score("test query", convo_candidate)
        
        # wpro should be zero by default
        assert "wpro" in components
        assert components["wpro"] == 0.0

    def test_prosody_weight_parsing(self, retrieval, memory_store):
        """Test that prosody weight is parsed correctly from environment."""
        memory_store.set_turn_prosody("test_session", 1, 0.8)
        
        convo_candidate = Candidate(
            text="User said something",
            source="convo",
            score_hint=0.5,
            ts=1000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Test various weight values
        test_cases = [
            ("0.5", 0.8 * 0.5),
            ("1.0", 0.8 * 1.0),
            ("0", 0.0),
            ("0.0", 0.0),
        ]
        
        for env_value, expected_wpro in test_cases:
            with patch.dict(os.environ, {"MEMORY_WEIGHT_PROSODY": env_value}):
                total_score, components = retrieval._composite_score("test query", convo_candidate)
                
                assert "wpro" in components
                assert abs(components["wpro"] - expected_wpro) < 0.001, f"Failed for env_value={env_value}"
