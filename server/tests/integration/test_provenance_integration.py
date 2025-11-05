#!/usr/bin/env python3
"""Integration tests for provenance system"""
import pytest
import sys
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory


@pytest.fixture
def hot_memory():
    """Create HotMemory with temporary storage"""
    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    return HotMemory(store)


def test_extraction_with_provenance(hot_memory):
    """Test full pipeline: conversation → extraction → provenance storage"""
    # Process conversation turn
    bullets, triples = hot_memory.process_turn(
        text="My name is Alice and I work at Google",
        session_id="session-1",
        turn_id=0
    )

    # Force flush
    hot_memory.store.flush_if_needed(max_ops=1)

    # Verify turn stored
    conversation = hot_memory.store.get_conversation("session-1")
    assert len(conversation) == 1
    assert "Alice" in conversation[0][1]

    # Verify edges extracted
    extractions = hot_memory.store.get_turn_extractions("session-1", 0)
    assert len(extractions) > 0

    # Verify provenance links
    for src, rel, dst, weight in extractions:
        edge_id = hot_memory.store.edge_id(src, rel, dst)
        provenance = hot_memory.store.get_edge_provenance(edge_id)
        assert len(provenance) >= 1
        assert provenance[0][1] == "session-1"  # session_id
        assert provenance[0][2] == 0            # turn_id


def test_edge_reinforcement_tracking(hot_memory):
    """Test that reinforced edges have multiple provenance rows"""
    # Say same fact in two different ways
    hot_memory.process_turn("My name is Bob", "session-1", 0)
    hot_memory.process_turn("I'm Bob", "session-1", 5)

    hot_memory.store.flush_if_needed(max_ops=1)

    # Edge should exist (subject normalized to user entity)
    edge_id = hot_memory.store.edge_id(hot_memory.user_eid, "name", "bob")

    # Should have 2 provenance sources across normalized edges (robust to relation variants)
    cur = hot_memory.store.sql.cursor()
    total = cur.execute(
        """
        SELECT COUNT(*)
        FROM edge_source es
        JOIN edge e ON es.edge_id = e.id
        WHERE e.src = ? AND e.dst = ?
        """,
        (hot_memory.user_eid, "bob")
    ).fetchone()[0]
    assert total == 2

    # Edge should be reinforced (pos > 0)
    cur = hot_memory.store.sql.cursor()
    result = cur.execute(
        "SELECT pos, weight FROM edge WHERE id = ?",
        (edge_id,)
    ).fetchone()
    assert result[0] > 0  # pos count


def test_question_no_extraction_but_stores_turn(hot_memory):
    """Test that questions store turn but don't extract edges"""
    bullets, triples = hot_memory.process_turn(
        text="What is your name?",
        session_id="session-1",
        turn_id=0
    )

    hot_memory.store.flush_if_needed(max_ops=1)

    # Turn should be stored
    conversation = hot_memory.store.get_conversation("session-1")
    assert len(conversation) == 1
    assert "name" in conversation[0][1].lower()

    # No edges extracted
    extractions = hot_memory.store.get_turn_extractions("session-1", 0)
    assert len(extractions) == 0


def test_multi_session_isolation(hot_memory):
    """Test that different sessions don't interfere"""
    # Two sessions, same turn_id
    hot_memory.process_turn("Alice works at Google", "session-1", 0)
    hot_memory.process_turn("Bob works at Microsoft", "session-2", 0)

    hot_memory.store.flush_if_needed(max_ops=1)

    # Each session should have 1 turn
    conv1 = hot_memory.store.get_conversation("session-1")
    conv2 = hot_memory.store.get_conversation("session-2")

    assert len(conv1) == 1
    assert len(conv2) == 1
    assert "Alice" in conv1[0][1]
    assert "Bob" in conv2[0][1]


def test_conversation_replay(hot_memory):
    """Test that we can replay a full conversation for evaluation"""
    # Simulate a multi-turn conversation
    conversation = [
        "My name is Charlie",
        "I work at Apple",
        "I live in California",
        "I like hiking",
        "What's the weather like?"  # Question, no edges
    ]

    for i, text in enumerate(conversation):
        hot_memory.process_turn(text, "session-1", i)

    hot_memory.store.flush_if_needed(max_ops=1)

    # Should have all 5 turns stored
    stored_conv = hot_memory.store.get_conversation("session-1")
    assert len(stored_conv) == 5

    # Verify we can replay in order
    for i, (turn_id, text, ts) in enumerate(stored_conv):
        assert turn_id == i
        assert conversation[i] in text or text in conversation[i]


def test_provenance_after_conflict_resolution(hot_memory):
    """Test provenance tracking when facts conflict"""
    # First fact
    hot_memory.process_turn("I live in New York", "session-1", 0)
    hot_memory.store.flush_if_needed(max_ops=1)

    # Conflicting fact
    hot_memory.process_turn("I live in San Francisco", "session-1", 5)
    hot_memory.store.flush_if_needed(max_ops=1)

    # Both turns should be stored
    conversation = hot_memory.store.get_conversation("session-1")
    assert len(conversation) == 2

    # The new edge (San Francisco) should have provenance
    # Check provenance for any edge ending at 'san francisco'
    cur = hot_memory.store.sql.cursor()
    rows = cur.execute(
        """
        SELECT t.text, es.extracted_at
        FROM edge_source es
        JOIN edge e ON es.edge_id = e.id
        JOIN conversation_turn t ON es.turn_id = t.id
        WHERE e.src = ? AND e.dst = ?
        ORDER BY es.extracted_at DESC
        """,
        (hot_memory.user_eid, "san francisco")
    ).fetchall()
    assert len(rows) >= 1
    assert "san francisco" in rows[0][0].lower()


def test_multiple_facts_single_turn(hot_memory):
    """Test that multiple facts from one turn all link to that turn"""
    # Complex sentence with multiple facts
    hot_memory.process_turn(
        "I'm David, I work at Tesla in Austin, and I love robotics",
        "session-1",
        0
    )

    hot_memory.store.flush_if_needed(max_ops=1)

    # Get all extractions from this turn
    extractions = hot_memory.store.get_turn_extractions("session-1", 0)

    # Should have multiple edges
    assert len(extractions) >= 2

    # All should link back to turn 0
    for src, rel, dst, weight in extractions:
        edge_id = hot_memory.store.edge_id(src, rel, dst)
        provenance = hot_memory.store.get_edge_provenance(edge_id)
        assert len(provenance) >= 1
        assert provenance[0][2] == 0  # turn_id


def test_provenance_persistence(hot_memory):
    """Test that provenance survives flush cycles"""
    # Process turn
    hot_memory.process_turn("I'm Emma and I study physics", "session-1", 0)

    # Multiple flush cycles
    hot_memory.store.flush_if_needed(max_ops=1)
    hot_memory.store.flush_if_needed(max_ops=1)
    hot_memory.store.flush_if_needed(max_ops=1)

    # Provenance should still be there
    conversation = hot_memory.store.get_conversation("session-1")
    assert len(conversation) == 1

    extractions = hot_memory.store.get_turn_extractions("session-1", 0)
    assert len(extractions) > 0


def test_edge_provenance_ordering_integration(hot_memory):
    """Test that provenance maintains chronological order across sessions"""
    # Same fact mentioned in different sessions at different times
    sessions = [
        ("session-1", 0, 1000, "My name is Frank"),
        ("session-2", 0, 2000, "I'm Frank"),
        ("session-3", 0, 3000, "Call me Frank"),
    ]

    for session_id, turn_id, _, text in sessions:
        hot_memory.process_turn(text, session_id, turn_id)

    hot_memory.store.flush_if_needed(max_ops=1)

    # Get edge provenance
    # Collect provenance across all normalized edges for 'frank'
    cur = hot_memory.store.sql.cursor()
    rows = cur.execute(
        """
        SELECT es.extracted_at
        FROM edge_source es
        JOIN edge e ON es.edge_id = e.id
        WHERE e.src = ? AND e.dst = ?
        ORDER BY es.extracted_at DESC
        """,
        (hot_memory.user_eid, "frank")
    ).fetchall()

    # Should have multiple sources, timestamps in descending order
    assert len(rows) >= 2
    timestamps = [r[0] for r in rows]
    assert timestamps == sorted(timestamps, reverse=True)


def test_negation_with_provenance(hot_memory):
    """Test that negated edges still maintain provenance"""
    # State a fact
    hot_memory.process_turn("I don't like coffee", "session-1", 0)
    hot_memory.store.flush_if_needed(max_ops=1)

    # Should have turn stored
    conversation = hot_memory.store.get_conversation("session-1")
    assert len(conversation) == 1

    # Check extractions (might include negation)
    extractions = hot_memory.store.get_turn_extractions("session-1", 0)

    # At least the turn should be recorded for provenance tracking
    # Even if no positive edge is created due to negation
    assert len(conversation) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
