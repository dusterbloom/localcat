#!/usr/bin/env python3
"""Unit tests for confidence strategies"""
import pytest
import sys
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.confidence_strategy import (
    RelationTypeConfidence,
    UsageBasedConfidence,
    Edge,
    Context,
    create_confidence_strategy
)
from core.memory.memory_store import MemoryStore, Paths


@pytest.fixture
def store():
    """Create in-memory store for testing"""
    return MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))


# --- RelationTypeConfidence Tests ---

def test_relation_type_baseline_name():
    """Test baseline confidence for 'name' relation"""
    strategy = RelationTypeConfidence()
    edge = Edge(src="I", rel="name", dst="Alice", pos=0, neg=0, updated_at=1000, id="test-id")
    context = Context(store=None)

    conf = strategy.score(edge, context)
    assert conf == 0.95


def test_relation_type_baseline_verb():
    """Test baseline confidence for verb relations"""
    strategy = RelationTypeConfidence()
    edge = Edge(src="I", rel="v:work_at", dst="Google", pos=0, neg=0, updated_at=1000, id="test-id")
    context = Context(store=None)

    conf = strategy.score(edge, context)
    assert conf == 0.85


def test_relation_type_baseline_other():
    """Test baseline confidence for other relations"""
    strategy = RelationTypeConfidence()
    edge = Edge(src="Alice", rel="likes", dst="coffee", pos=0, neg=0, updated_at=1000, id="test-id")
    context = Context(store=None)

    conf = strategy.score(edge, context)
    assert conf == 0.9


# --- UsageBasedConfidence Tests ---

def test_usage_based_no_reinforcement(store):
    """Test usage-based confidence with no reinforcement"""
    import time
    strategy = UsageBasedConfidence()
    now_ts = int(time.time() * 1000)  # Current time in milliseconds
    edge = Edge(src="I", rel="name", dst="Bob", pos=0, neg=0, updated_at=now_ts, id="test-id")
    context = Context(store=store)

    # Should match baseline (no reinforcement/recency/source modifiers)
    conf = strategy.score(edge, context)
    assert conf == pytest.approx(0.95, abs=0.01)


def test_usage_based_reinforcement_boost(store):
    """Test reinforcement boosts confidence"""
    import time
    strategy = UsageBasedConfidence()
    now_ts = int(time.time() * 1000)  # Current time

    # No reinforcement
    edge1 = Edge(src="I", rel="name", dst="Charlie", pos=0, neg=0, updated_at=now_ts, id="test-id")
    conf1 = strategy.score(edge1, Context(store=store))

    # With 1 reinforcement
    edge2 = Edge(src="I", rel="name", dst="Charlie", pos=1, neg=0, updated_at=now_ts, id="test-id")
    conf2 = strategy.score(edge2, Context(store=store))

    # With 3 reinforcements (max boost)
    edge3 = Edge(src="I", rel="name", dst="Charlie", pos=3, neg=0, updated_at=now_ts, id="test-id")
    conf3 = strategy.score(edge3, Context(store=store))

    assert conf2 > conf1  # Reinforcement boosts confidence
    assert conf3 > conf2 or conf3 == 1.0  # More reinforcements = more boost (may clamp at 1.0)
    # Expected is 0.95 * 1.15 = 1.0925, which gets clamped to 1.0
    assert conf3 == pytest.approx(1.0, abs=0.01)  # Clamped to max


def test_usage_based_negation_penalty(store):
    """Test negation reduces confidence"""
    import time
    strategy = UsageBasedConfidence()
    now_ts = int(time.time() * 1000)

    # No negation
    edge1 = Edge(src="I", rel="v:live_in", dst="NYC", pos=0, neg=0, updated_at=now_ts, id="test-id")
    conf1 = strategy.score(edge1, Context(store=store))

    # With 1 negation
    edge2 = Edge(src="I", rel="v:live_in", dst="NYC", pos=0, neg=1, updated_at=now_ts, id="test-id")
    conf2 = strategy.score(edge2, Context(store=store))

    # With 3 negations (max penalty)
    edge3 = Edge(src="I", rel="v:live_in", dst="NYC", pos=0, neg=3, updated_at=now_ts, id="test-id")
    conf3 = strategy.score(edge3, Context(store=store))

    assert conf2 < conf1  # Negation reduces confidence
    assert conf3 < conf2  # More negations = more penalty
    assert conf3 == pytest.approx(0.85 * 0.4, abs=0.01)  # Max 60% penalty


def test_usage_based_recency_decay(store):
    """Test that old facts decay"""
    import time
    strategy = UsageBasedConfidence()
    now = int(time.time() * 1000)

    # Recent fact (today)
    edge1 = Edge(src="I", rel="name", dst="David", pos=0, neg=0, updated_at=now, id="test-id")
    conf1 = strategy.score(edge1, Context(store=store))

    # 40 days old
    days_40_ms = 40 * 24 * 60 * 60 * 1000
    edge2 = Edge(src="I", rel="name", dst="David", pos=0, neg=0, updated_at=now - days_40_ms, id="test-id")
    conf2 = strategy.score(edge2, Context(store=store))

    # 100 days old
    days_100_ms = 100 * 24 * 60 * 60 * 1000
    edge3 = Edge(src="I", rel="name", dst="David", pos=0, neg=0, updated_at=now - days_100_ms, id="test-id")
    conf3 = strategy.score(edge3, Context(store=store))

    assert conf1 > conf2  # 40 days old decays
    assert conf2 > conf3  # 100 days old decays more
    assert conf2 == pytest.approx(0.95 * 0.9, abs=0.01)  # 10% penalty after 30 days
    assert conf3 == pytest.approx(0.95 * 0.8, abs=0.01)  # 20% penalty after 90 days


def test_usage_based_source_count_boost(store):
    """Test that multiple sources boost confidence"""
    import time
    strategy = UsageBasedConfidence()
    now_ts = int(time.time() * 1000)

    # Store edge with multiple sources
    edge_id = store.edge_id("I", "name", "Emma")
    store.observe_edge("I", "name", "Emma", 0.95, now_ts)

    # Add 3 conversation sources
    for i in range(3):
        tid = store.enqueue_turn(f"I'm Emma turn {i}", "session-1", i, now_ts + i)
        store.enqueue_edge_source(edge_id, tid, now_ts + i)

    store.flush_if_needed(max_ops=1)

    # Score confidence
    edge = Edge(src="I", rel="name", dst="Emma", pos=0, neg=0, updated_at=now_ts, id=edge_id)
    conf = strategy.score(edge, Context(store=store))

    # Should have 10% boost for 3+ sources (0.95 * 1.1 = 1.045, clamped to 1.0)
    assert conf == pytest.approx(1.0, abs=0.01)


def test_usage_based_combined_signals(store):
    """Test combination of all signals"""
    import time
    strategy = UsageBasedConfidence()
    now_ts = int(time.time() * 1000)

    # Edge with reinforcement, recent, and multiple sources
    edge_id = store.edge_id("I", "name", "Frank")
    store.observe_edge("I", "name", "Frank", 0.95, now_ts)

    # Add sources
    for i in range(2):
        tid = store.enqueue_turn(f"I'm Frank", "session-1", i, now_ts)
        store.enqueue_edge_source(edge_id, tid, now_ts)

    store.flush_if_needed(max_ops=1)

    # Score with reinforcement
    edge = Edge(src="I", rel="name", dst="Frank", pos=2, neg=0, updated_at=now_ts, id=edge_id)
    conf = strategy.score(edge, Context(store=store))

    # Expected: 0.95 * (1 + 0.05*2) [reinf] * 1.0 [recency] * 1.05 [sources] = 1.09725
    # Clamped to 1.0
    assert conf == pytest.approx(1.0, abs=0.01)


def test_usage_based_clamp_to_range(store):
    """Test that confidence is clamped to [0.0, 1.0]"""
    import time
    strategy = UsageBasedConfidence()
    now_ts = int(time.time() * 1000)

    # Edge with extreme reinforcement (should clamp to 1.0)
    edge1 = Edge(src="I", rel="name", dst="Grace", pos=10, neg=0, updated_at=now_ts, id="test-id")
    conf1 = strategy.score(edge1, Context(store=store))
    assert conf1 <= 1.0

    # Edge with extreme negation (should not go below 0.0)
    edge2 = Edge(src="I", rel="name", dst="Henry", pos=0, neg=10, updated_at=now_ts, id="test-id")
    conf2 = strategy.score(edge2, Context(store=store))
    assert conf2 >= 0.0


# --- Factory Tests ---

def test_create_strategy_relation_type():
    """Test factory creates RelationTypeConfidence"""
    strategy = create_confidence_strategy("relation_type")
    assert isinstance(strategy, RelationTypeConfidence)


def test_create_strategy_usage_based():
    """Test factory creates UsageBasedConfidence"""
    strategy = create_confidence_strategy("usage_based")
    assert isinstance(strategy, UsageBasedConfidence)


def test_create_strategy_unknown_raises():
    """Test factory raises on unknown strategy"""
    with pytest.raises(ValueError, match="Unknown confidence strategy"):
        create_confidence_strategy("invalid")


# --- Integration with HotMemory ---

def test_hotmemory_uses_injected_strategy(store):
    """Test that HotMemory uses the injected confidence strategy"""
    from core.memory.memory_hotpath import HotMemory

    # Create HotMemory with UsageBasedConfidence
    strategy = UsageBasedConfidence()
    hot_memory = HotMemory(store, confidence_strategy=strategy)

    # Verify strategy is set
    assert hot_memory.confidence is strategy


def test_hotmemory_defaults_to_relation_type(store):
    """Test that HotMemory defaults to RelationTypeConfidence"""
    from core.memory.memory_hotpath import HotMemory

    # Create HotMemory without strategy
    hot_memory = HotMemory(store)

    # Should default to RelationTypeConfidence
    assert isinstance(hot_memory.confidence, RelationTypeConfidence)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])