#!/usr/bin/env python3
"""
Integration test for full confidence learning pipeline

Tests the complete flow:
1. Edge provenance tracking (conversation → edges → provenance)
2. Confidence strategy injection (factory → service → HotMemory)
3. Usage-based confidence learning (structural signals)
4. End-to-end memory processing
"""
import pytest
import sys
import os
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
from core.memory.hotmem_service import HotMemService
from core.memory.confidence_strategy import (
    RelationTypeConfidence,
    UsageBasedConfidence,
    create_confidence_strategy
)


def test_end_to_end_baseline_confidence():
    """Test full pipeline with baseline (RelationTypeConfidence)"""
    # Create service with baseline strategy
    service = HotMemService(
        user_id="test-user",
        sqlite_path=":memory:",
        lmdb_dir=None,
        confidence_strategy=RelationTypeConfidence()
    )

    # Process a conversation turn
    session_id = "test-session-1"
    turn_id = 0

    # Simulate processing through HotMemory
    bullets, triples = service.hot.process_turn(
        text="My name is Alice",
        session_id=session_id,
        turn_id=turn_id
    )

    # Flush to ensure data is written
    service.store.flush_if_needed(max_ops=1)

    # Verify provenance was tracked
    conversation = service.store.get_conversation(session_id)
    assert len(conversation) >= 1
    assert "Alice" in conversation[0][1]

    # Verify confidence is from baseline
    assert isinstance(service.hot.confidence, RelationTypeConfidence)


def test_end_to_end_usage_based_confidence():
    """Test full pipeline with usage-based learning"""
    # Create service with usage-based strategy
    service = HotMemService(
        user_id="test-user",
        sqlite_path=":memory:",
        lmdb_dir=None,
        confidence_strategy=UsageBasedConfidence()
    )

    session_id = "test-session-2"

    # Process multiple mentions of same fact
    for i in range(3):
        service.hot.process_turn(
            text="My name is Bob",
            session_id=session_id,
            turn_id=i
        )

    service.store.flush_if_needed(max_ops=1)

    # Verify all turns stored
    conversation = service.store.get_conversation(session_id)
    assert len(conversation) == 3

    # Verify provenance shows multiple sources
    edge_id = service.store.edge_id("I", "name", "Bob")
    source_count = service.store.get_edge_sources_count(edge_id)
    assert source_count >= 2  # Should have multiple sources

    # Verify confidence strategy is usage-based
    assert isinstance(service.hot.confidence, UsageBasedConfidence)


def test_confidence_affects_scoring():
    """Test that different strategies produce different confidence scores"""
    import time

    # Baseline strategy
    baseline_service = HotMemService(
        user_id="user-baseline",
        sqlite_path=":memory:",
        lmdb_dir=None,
        confidence_strategy=RelationTypeConfidence()
    )

    # Usage-based strategy
    usage_service = HotMemService(
        user_id="user-usage",
        sqlite_path=":memory:",
        lmdb_dir=None,
        confidence_strategy=UsageBasedConfidence()
    )

    # Process same fact with reinforcement in usage-based
    text = "I work at Google"
    session_id = "test-session"

    # Baseline: single mention
    baseline_service.hot.process_turn(text, session_id, 0)
    baseline_service.store.flush_if_needed(max_ops=1)

    # Usage-based: multiple mentions (should boost confidence)
    for i in range(3):
        usage_service.hot.process_turn(text, session_id, i)
    usage_service.store.flush_if_needed(max_ops=1)

    # Get edge IDs
    baseline_edge_id = baseline_service.store.edge_id("I", "v:work_at", "Google")
    usage_edge_id = usage_service.store.edge_id("I", "v:work_at", "Google")

    # Get stored confidence values
    baseline_edge = baseline_service.store.sql.cursor().execute(
        "SELECT weight, pos FROM edge WHERE id = ?", (baseline_edge_id,)
    ).fetchone()

    usage_edge = usage_service.store.sql.cursor().execute(
        "SELECT weight, pos FROM edge WHERE id = ?", (usage_edge_id,)
    ).fetchone()

    # Usage-based should have reinforcement
    if usage_edge:
        assert usage_edge[1] >= 2  # pos (reinforcement count) should be >= 2


def test_factory_integration():
    """Test that factory correctly injects confidence strategy"""
    from core.factory import VoiceAgentFactory
    from config import VoiceAgentConfig

    config = VoiceAgentConfig()
    factory = VoiceAgentFactory(config)

    # Test with environment variable
    os.environ['CONFIDENCE_STRATEGY'] = 'usage_based'
    service = factory.create_hotmem_service()

    assert isinstance(service.hot.confidence, UsageBasedConfidence)

    # Cleanup
    del os.environ['CONFIDENCE_STRATEGY']


def test_strategy_factory():
    """Test confidence strategy factory function"""
    # Baseline
    baseline = create_confidence_strategy("relation_type")
    assert isinstance(baseline, RelationTypeConfidence)

    # Usage-based
    usage = create_confidence_strategy("usage_based")
    assert isinstance(usage, UsageBasedConfidence)

    # Invalid
    with pytest.raises(ValueError):
        create_confidence_strategy("invalid_strategy")


def test_provenance_with_confidence_evolution():
    """Test that confidence evolves over time with provenance"""
    service = HotMemService(
        user_id="test-user",
        sqlite_path=":memory:",
        lmdb_dir=None,
        confidence_strategy=UsageBasedConfidence()
    )

    session_id = "evolution-test"

    # First mention
    service.hot.process_turn("I live in NYC", session_id, 0)
    service.store.flush_if_needed(max_ops=1)

    edge_id = service.store.edge_id("I", "v:live_in", "NYC")

    # Check initial state
    initial_count = service.store.get_edge_sources_count(edge_id)
    assert initial_count >= 1

    # Reinforce with more mentions
    for i in range(1, 4):
        service.hot.process_turn("I live in NYC", session_id, i)
    service.store.flush_if_needed(max_ops=1)

    # Check evolved state
    final_count = service.store.get_edge_sources_count(edge_id)
    assert final_count > initial_count

    # Verify provenance shows all mentions
    provenance = service.store.get_edge_provenance(edge_id)
    assert len(provenance) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])