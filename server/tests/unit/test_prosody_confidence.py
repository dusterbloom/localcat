#!/usr/bin/env python3
"""Unit tests for prosody-aware confidence fallback to stored prosody data."""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.confidence_strategy import ProsodyAwareConfidence, Edge, Context
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
def prosody_strategy():
    """Create a ProsodyAwareConfidence strategy without fusion."""
    # Mock the fusion to be unavailable for testing fallback
    strategy = ProsodyAwareConfidence()
    strategy.fusion = None  # Disable fusion to test simple adjustment
    return strategy


class TestProsodyConfidenceFallback:
    """Test prosody-aware confidence fallback to stored prosody."""

    def test_low_stored_certainty_reduces_confidence(self, memory_store, prosody_strategy):
        """Test that low stored certainty reduces confidence below baseline."""
        session_id = "test_session"
        turn_id = 1
        
        # Store low prosody certainty
        memory_store.set_turn_prosody(session_id, turn_id, 0.2)  # Low certainty
        
        # Create test edge and context
        edge = Edge(src="Alice", rel="name", dst="Smith", pos=1, neg=0, updated_at=1000, id="test_edge")
        context = Context(
            store=memory_store,
            text="Alice Smith",
            session_id=session_id,
            turn_id=turn_id,
            prosody_features=None  # No inline prosody, should trigger fallback
        )
        
        # Get confidence with stored prosody
        confidence = prosody_strategy.score(edge, context)
        
        # Get baseline confidence for comparison
        baseline_strategy = prosody_strategy.baseline
        baseline_confidence = baseline_strategy.score(edge, context)
        
        # Confidence with low stored certainty should be lower than baseline
        assert confidence < baseline_confidence
        assert confidence > 0.0  # Should still be positive

    def test_high_stored_certainty_increases_confidence(self, memory_store, prosody_strategy):
        """Test that high stored certainty increases confidence above baseline."""
        session_id = "test_session"
        turn_id = 2
        
        # Store high prosody certainty
        memory_store.set_turn_prosody(session_id, turn_id, 0.9)  # High certainty
        
        # Create test edge and context
        edge = Edge(src="Bob", rel="likes", dst="hiking", pos=1, neg=0, updated_at=1000, id="test_edge2")
        context = Context(
            store=memory_store,
            text="Bob likes hiking",
            session_id=session_id,
            turn_id=turn_id,
            prosody_features=None  # No inline prosody, should trigger fallback
        )
        
        # Get confidence with stored prosody
        confidence = prosody_strategy.score(edge, context)
        
        # Get baseline confidence for comparison
        baseline_strategy = prosody_strategy.baseline
        baseline_confidence = baseline_strategy.score(edge, context)
        
        # Confidence with high stored certainty should be higher than baseline
        assert confidence > baseline_confidence
        assert confidence <= 1.0  # Should not exceed 1.0

    def test_missing_store_meta_falls_back_to_baseline(self, prosody_strategy):
        """Test that missing store/meta falls back to baseline without error."""
        # Create context without store
        edge = Edge(src="Charlie", rel="lives_in", dst="NYC", pos=1, neg=0, updated_at=1000, id="test_edge3")
        context = Context(
            store=None,  # No store
            text="Charlie lives in NYC",
            session_id="missing_session",
            turn_id=999,
            prosody_features=None
        )
        
        # Should fall back to baseline without error
        confidence = prosody_strategy.score(edge, context)
        baseline_confidence = prosody_strategy.baseline.score(edge, context)
        
        assert confidence == baseline_confidence

    def test_missing_session_or_turn_falls_back_to_baseline(self, memory_store, prosody_strategy):
        """Test that missing session_id or turn_id falls back to baseline."""
        edge = Edge(src="Diana", rel="works_at", dst="Acme", pos=1, neg=0, updated_at=1000, id="test_edge4")
        
        # Test missing session_id
        context1 = Context(
            store=memory_store,
            text="Diana works at Acme",
            session_id=None,  # Missing session
            turn_id=1,
            prosody_features=None
        )
        
        confidence1 = prosody_strategy.score(edge, context1)
        baseline_confidence = prosody_strategy.baseline.score(edge, context1)
        assert confidence1 == baseline_confidence
        
        # Test missing turn_id
        context2 = Context(
            store=memory_store,
            text="Diana works at Acme",
            session_id="test_session",
            turn_id=None,  # Missing turn
            prosody_features=None
        )
        
        confidence2 = prosody_strategy.score(edge, context2)
        assert confidence2 == baseline_confidence

    def test_neutral_certainty_no_adjustment(self, memory_store, prosody_strategy):
        """Test that neutral certainty (0.5) doesn't adjust baseline."""
        session_id = "test_session"
        turn_id = 5
        
        # Store neutral prosody certainty
        memory_store.set_turn_prosody(session_id, turn_id, 0.5)  # Neutral
        
        # Create test edge and context
        edge = Edge(src="Eve", rel="has", dst="cat", pos=1, neg=0, updated_at=1000, id="test_edge5")
        context = Context(
            store=memory_store,
            text="Eve has a cat",
            session_id=session_id,
            turn_id=turn_id,
            prosody_features=None
        )
        
        # Get confidence with stored prosody
        confidence = prosody_strategy.score(edge, context)
        baseline_confidence = prosody_strategy.baseline.score(edge, context)
        
        # Neutral certainty should not adjust baseline
        assert confidence == baseline_confidence

    def test_inline_prosody_takes_precedence(self, memory_store, prosody_strategy):
        """Test that inline prosody features take precedence over stored prosody."""
        session_id = "test_session"
        turn_id = 6
        
        # Store high prosody certainty
        memory_store.set_turn_prosody(session_id, turn_id, 0.9)
        
        # Create mock inline prosody with low certainty
        mock_prosody = Mock()
        mock_prosody.certainty_modifier = -0.2  # Low certainty modifier
        
        # Create test edge and context with inline prosody
        edge = Edge(src="Frank", rel="is", dst="tall", pos=1, neg=0, updated_at=1000, id="test_edge6")
        context = Context(
            store=memory_store,
            text="Frank is tall",
            session_id=session_id,
            turn_id=turn_id,
            prosody_features=mock_prosody  # Inline prosody provided
        )
        
        # Should use inline prosody, not stored prosody (fallback not triggered)
        # Since fusion is None in our fixture, it will fall back to baseline
        confidence = prosody_strategy.score(edge, context)
        baseline_confidence = prosody_strategy.baseline.score(edge, context)
        
        # Should equal baseline (fusion disabled, inline prosody ignored)
        assert confidence == baseline_confidence

    def test_confidence_bounds(self, memory_store, prosody_strategy):
        """Test that confidence stays within [0, 1] bounds."""
        session_id = "test_session"
        turn_id = 7
        
        # Test very low certainty
        memory_store.set_turn_prosody(session_id, turn_id, 0.0)
        
        edge = Edge(src="Grace", rel="has", dst="dog", pos=1, neg=0, updated_at=1000, id="test_edge7")
        context = Context(
            store=memory_store,
            text="Grace has a dog",
            session_id=session_id,
            turn_id=turn_id,
            prosody_features=None
        )
        
        confidence_low = prosody_strategy.score(edge, context)
        assert 0.0 <= confidence_low <= 1.0
        
        # Test very high certainty
        memory_store.set_turn_prosody(session_id, turn_id, 1.0)
        confidence_high = prosody_strategy.score(edge, context)
        assert 0.0 <= confidence_high <= 1.0

    def test_synthetic_prosody_features_creation(self, memory_store):
        """Test the creation of synthetic prosody features from stored certainty."""
        strategy = ProsodyAwareConfidence()
        strategy.fusion = None  # Disable fusion
        
        session_id = "test_session"
        turn_id = 8
        
        # Store high certainty
        memory_store.set_turn_prosody(session_id, turn_id, 0.8)
        
        edge = Edge(src="Henry", rel="likes", dst="music", pos=1, neg=0, updated_at=1000, id="test_edge8")
        context = Context(
            store=memory_store,
            text="Henry likes music",
            session_id=session_id,
            turn_id=turn_id,
            prosody_features=None
        )
        
        # This should trigger synthetic prosody creation
        confidence = strategy._score_with_stored_prosody(edge, context)
        
        # Should be a valid confidence score
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # High certainty should boost baseline

    def test_certainty_modifier_calculation(self, memory_store):
        """Test that certainty modifier is calculated correctly."""
        strategy = ProsodyAwareConfidence()
        strategy.fusion = None
        
        edge = Edge(src="Iris", rel="is", dst="happy", pos=1, neg=0, updated_at=1000, id="test_edge9")
        
        # Test low certainty maps to negative modifier
        memory_store.set_turn_prosody("session", 1, 0.1)  # Should map to -0.3
        context = Context(store=memory_store, session_id="session", turn_id=1)
        confidence1 = strategy._score_with_stored_prosody(edge, context)
        
        # Test high certainty maps to positive modifier
        memory_store.set_turn_prosody("session", 2, 0.9)  # Should map to +0.3
        context = Context(store=memory_store, session_id="session", turn_id=2)
        confidence2 = strategy._score_with_stored_prosody(edge, context)
        
        # Test neutral certainty maps to zero modifier
        memory_store.set_turn_prosody("session", 3, 0.5)  # Should map to 0.0
        context = Context(store=memory_store, session_id="session", turn_id=3)
        confidence3 = strategy._score_with_stored_prosody(edge, context)
        
        baseline = strategy.baseline.score(edge, context)
        
        # High confidence should be > baseline, low should be < baseline, neutral == baseline
        assert confidence2 > baseline
        assert confidence1 < baseline
        assert confidence3 == baseline
