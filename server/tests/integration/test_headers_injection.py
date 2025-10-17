#!/usr/bin/env python3
"""Integration tests for headers-first injection with auto-expand threshold."""

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


class TestHeadersInjection:
    """Test headers-first injection functionality."""

    def test_headers_mode_outputs_compact_headers(self, retrieval):
        """Test that headers mode outputs compact headers, not full bullets."""
        # Create test candidates
        graph_candidate = Candidate(
            text="Alice is named Smith",
            source="graph",
            score_hint=0.0,
            ts=1000,
            meta={"edge_id": "edge1", "weight": 0.95, "pos": 2, "neg": 0}
        )
        
        convo_candidate = Candidate(
            text="Bob mentioned he went to the store yesterday to buy groceries",
            source="convo",
            score_hint=0.5,
            ts=2000,
            meta={"bm25_score": 0.5, "eid": "conversation", "turn_id": 1}
        )
        
        # Test header formatting
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers"}):
            # Format bullets using headers mode
            bullets, selected = retrieval._apply_token_budget_and_deduplication(
                [(0.8, graph_candidate, {"wconf": 0.34, "wrec": 0.2, "wuse": 0.05, "wsrc": 0.3})],
                max_bullets=3,
                query="test"
            )
            
            # Should output compact header, not full bullet
            assert len(bullets) == 1
            bullet = bullets[0]
            
            # Should be header format, not legacy bullet format
            assert bullet.startswith("• name: Alice [conf=")
            assert "[graph]" not in bullet
            assert "Alice is named Smith" not in bullet  # Full text not in header
            
            # Should contain confidence scalar
            assert "conf=" in bullet
            assert "rec=" in bullet
            assert "use=" in bullet

    def test_low_score_candidate_auto_expands(self, retrieval):
        """Test that low-scoring candidates expand to full text."""
        # Create candidate with low score (below default threshold 0.65)
        low_score_candidate = Candidate(
            text="This is a very long conversation text that should be truncated in normal mode but shown in full when auto-expanded due to low score",
            source="convo",
            score_hint=0.1,  # Low score
            ts=2000,
            meta={"bm25_score": 0.1, "eid": "conversation", "turn_id": 1}
        )
        
        # Test with low score and headers mode
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers", "MEMORY_HEADER_EXPAND_THRESHOLD": "0.65"}):
            bullets, selected = retrieval._apply_token_budget_and_deduplication(
                [(0.5, low_score_candidate, {"wconf": 0.1, "wrec": 0.1, "wsrc": 0.2})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            bullet = bullets[0]
            
            # Should contain both header and full text (expanded)
            assert "convo:" in bullet
            assert "->" in bullet
            assert low_score_candidate.text in bullet  # Full text should be present

    def test_high_score_candidate_does_not_expand(self, retrieval):
        """Test that high-scoring candidates remain compact."""
        # Create candidate with high score (above threshold)
        high_score_candidate = Candidate(
            text="Short text",
            source="graph",
            score_hint=0.0,
            ts=1000,
            meta={"edge_id": "edge1", "weight": 0.9, "pos": 3, "neg": 0}
        )
        
        # Test with high score and headers mode
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers", "MEMORY_HEADER_EXPAND_THRESHOLD": "0.5"}):
            bullets, selected = retrieval._apply_token_budget_and_deduplication(
                [(0.8, high_score_candidate, {"wconf": 0.35, "wrec": 0.2, "wuse": 0.1, "wsrc": 0.3})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            bullet = bullets[0]
            
            # Should be compact header only (no expansion)
            assert "->" not in bullet
            assert bullet.startswith("• fact:")
            assert "conf=" in bullet

    def test_headers_mode_with_convo_prosody(self, retrieval, memory_store):
        """Test that convo headers include prosody when available."""
        # Store prosody data
        memory_store.set_turn_prosody("test_session", 1, 0.75)
        
        convo_candidate = Candidate(
            text="User said something with high confidence",
            source="convo",
            score_hint=0.6,
            ts=2000,
            meta={"bm25_score": 0.6, "eid": "conversation", "turn_id": 1}
        )
        
        # Test with prosody weight enabled
        with patch.dict(os.environ, {
            "MEMORY_INJECTION_MODE": "headers",
            "MEMORY_WEIGHT_PROSODY": "0.2"
        }):
            bullets, selected = retrieval._apply_token_budget_and_deduplication(
                [(0.7, convo_candidate, {"wconf": 0.2, "wrec": 0.15, "wpro": 0.15, "wsrc": 0.2})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            bullet = bullets[0]
            
            # Should include prosody in header
            assert "pro=" in bullet
            assert "convo:" in bullet
            assert "conf=" in bullet
            assert "rec=" in bullet

    def test_legacy_mode_unchanged(self, retrieval):
        """Test that legacy mode outputs unchanged bullet format."""
        candidate = Candidate(
            text="Alice lives in NYC",
            source="graph",
            score_hint=0.0,
            ts=1000,
            meta={"edge_id": "edge1"}
        )
        
        # Test with legacy mode (default)
        bullets, selected = retrieval._apply_token_budget_and_deduplication(
            [(0.5, candidate, {"wconf": 0.35, "wrec": 0.2, "wsrc": 0.3})],
            max_bullets=3,
            query="test"
        )
        
        assert len(bullets) == 1
        bullet = bullets[0]
        
        # Should be legacy bullet format
        assert bullet.startswith("• [graph] Alice lives in NYC")
        assert "conf=" not in bullet  # No scalar info in legacy mode

    def test_headers_mode_graph_name_parsing(self, retrieval):
        """Test that graph headers correctly parse name relations."""
        name_candidate = Candidate(
            text="John is named Doe",
            source="graph",
            score_hint=0.0,
            ts=1000,
            meta={"edge_id": "edge1", "weight": 0.95, "pos": 2, "neg": 0}
        )
        
        has_candidate = Candidate(
            text="Mary has a cat",
            source="graph",
            score_hint=0.0,
            ts=1000,
            meta={"edge_id": "edge2", "weight": 0.8, "pos": 1, "neg": 0}
        )
        
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers"}):
            # Test name parsing
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                [(0.8, name_candidate, {"wconf": 0.35, "wrec": 0.2, "wuse": 0.1, "wsrc": 0.3})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            assert bullets[0].startswith("• name: John [conf=")
            
            # Test has parsing
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                [(0.7, has_candidate, {"wconf": 0.3, "wrec": 0.2, "wuse": 0.05, "wsrc": 0.3})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            assert bullets[0].startswith("• has: Mary [conf=")

    def test_custom_expand_threshold(self, retrieval):
        """Test that custom expand threshold works correctly."""
        # Create candidate with medium score
        candidate = Candidate(
            text="Medium scoring candidate",
            source="convo",
            score_hint=0.4,
            ts=2000,
            meta={"bm25_score": 0.4, "eid": "conversation", "turn_id": 1}
        )
        
        # Test with high threshold (should expand)
        with patch.dict(os.environ, {
            "MEMORY_INJECTION_MODE": "headers",
            "MEMORY_HEADER_EXPAND_THRESHOLD": "0.8"
        }):
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                [(0.6, candidate, {"wconf": 0.2, "wrec": 0.2, "wsrc": 0.2})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            assert "->" in bullets[0]  # Should expand
        
        # Test with low threshold (should not expand)
        with patch.dict(os.environ, {
            "MEMORY_INJECTION_MODE": "headers",
            "MEMORY_HEADER_EXPAND_THRESHOLD": "0.3"
        }):
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                [(0.6, candidate, {"wconf": 0.2, "wrec": 0.2, "wsrc": 0.2})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            assert "->" not in bullets[0]  # Should not expand

    def test_semantic_headers_with_similarity(self, retrieval):
        """Test that semantic headers include similarity scores."""
        semantic_candidate = Candidate(
            text="Related to hobbies and interests",
            source="semantic",
            score_hint=0.7,
            ts=1000,
            meta={"similarity_score": 0.7}
        )
        
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers"}):
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                [(0.5, semantic_candidate, {"wconf": 0.2, "wrec": 0.1, "wsrc": 0.2})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            bullet = bullets[0]
            
            # Should include similarity and confidence
            assert "semantic:" in bullet
            assert "sim=0.70" in bullet or "sim=0.7" in bullet
            assert "conf=" in bullet

    def test_summary_headers_formatting(self, retrieval):
        """Test that summary headers are formatted correctly."""
        summary_candidate = Candidate(
            text="Summary of recent conversations about work and projects",
            source="summary",
            score_hint=0.5,
            ts=1000,
            meta={}
        )
        
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers"}):
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                [(0.4, summary_candidate, {"wconf": 0.1, "wrec": 0.15, "wsrc": 0.15})],
                max_bullets=3,
                query="test"
            )
            
            assert len(bullets) == 1
            bullet = bullets[0]
            
            # Should include summary label and scalars
            assert "summary:" in bullet
            assert "conf=" in bullet
            assert "rec=" in bullet
            assert "sim=" not in bullet  # No similarity in summary

    def test_headers_token_budget_respected(self, retrieval):
        """Test that headers mode respects token budget."""
        # Create multiple candidates
        candidates = [
            Candidate(
                text=f"Candidate {i}",
                source="graph",
                score_hint=0.0,
                ts=1000 + i * 100,
                meta={"edge_id": f"edge{i}"}
            )
            for i in range(5)
        ]
        
        scored_candidates = [
            (0.8 - i * 0.1, candidate, {"wconf": 0.3, "wrec": 0.2, "wsrc": 0.3})
            for i, candidate in enumerate(candidates)
        ]
        
        with patch.dict(os.environ, {"MEMORY_INJECTION_MODE": "headers"}):
            # Test with restrictive token budget
            bullets, _ = retrieval._apply_token_budget_and_deduplication(
                scored_candidates,
                max_bullets=10,  # High bullet limit
                query="test"
            )
            
            # Should still respect token budget (headers are shorter, so more may fit)
            assert len(bullets) > 0
            for bullet in bullets:
                assert bullet.startswith("• fact:")
                assert "[conf=" in bullet
