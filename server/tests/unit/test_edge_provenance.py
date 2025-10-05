#!/usr/bin/env python3
"""Unit tests for edge provenance system"""
import pytest
import sys
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.memory_store import MemoryStore, Paths


@pytest.fixture
def store():
    """Create temporary in-memory store"""
    return MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))


def test_conversation_turn_storage(store):
    """Test storing and retrieving conversation turns"""
    # Store turn
    tid = store.enqueue_turn("Hello world", "session-1", 0, 1000)
    store.flush_if_needed(max_ops=1)

    # Verify stored
    cur = store.sql.cursor()
    result = cur.execute(
        "SELECT text, session_id, turn_id FROM conversation_turn WHERE id = ?",
        (tid,)
    ).fetchone()

    assert result is not None
    assert result[0] == "Hello world"
    assert result[1] == "session-1"
    assert result[2] == 0


def test_turn_idempotency(store):
    """Test that same turn isn't duplicated"""
    tid1 = store.enqueue_turn("Test", "session-1", 5, 1000)
    tid2 = store.enqueue_turn("Test", "session-1", 5, 1001)  # Different timestamp

    assert tid1 == tid2  # Same hash

    store.flush_if_needed(max_ops=1)

    # Only one row
    cur = store.sql.cursor()
    count = cur.execute("SELECT COUNT(*) FROM conversation_turn").fetchone()[0]
    assert count == 1


def test_edge_source_linking(store):
    """Test linking edges to conversation turns"""
    # Store turn and edge
    tid = store.enqueue_turn("Alice works at Google", "session-1", 0, 1000)
    store.observe_edge("Alice", "works_at", "Google", 0.9, 1000)
    edge_id = store.edge_id("Alice", "works_at", "Google")
    store.enqueue_edge_source(edge_id, tid, 1000)

    store.flush_if_needed(max_ops=1)

    # Verify link
    cur = store.sql.cursor()
    result = cur.execute(
        "SELECT turn_id FROM edge_source WHERE edge_id = ?",
        (edge_id,)
    ).fetchone()

    assert result is not None
    assert result[0] == tid


def test_multiple_sources_for_edge(store):
    """Test same edge extracted from multiple conversations"""
    # Create two turns with same fact
    tid1 = store.enqueue_turn("My name is Bob", "session-1", 0, 1000)
    tid2 = store.enqueue_turn("I'm Bob", "session-1", 5, 2000)

    # Same edge from both
    store.observe_edge("I", "name", "Bob", 0.95, 1000)
    edge_id = store.edge_id("I", "name", "Bob")
    store.enqueue_edge_source(edge_id, tid1, 1000)
    store.enqueue_edge_source(edge_id, tid2, 2000)

    store.flush_if_needed(max_ops=1)

    # Should have 2 sources
    count = store.get_edge_sources_count(edge_id)
    assert count == 2

    # Provenance should show both
    provenance = store.get_edge_provenance(edge_id)
    assert len(provenance) == 2
    texts = [p[0] for p in provenance]
    assert "My name is Bob" in texts
    assert "I'm Bob" in texts


def test_get_turn_extractions(store):
    """Test retrieving all edges from a conversation turn"""
    # Store turn with multiple facts
    tid = store.enqueue_turn("Alice works at Google in California", "session-1", 0, 1000)

    # Extract multiple edges
    edges = [
        ("Alice", "works_at", "Google", 0.9),
        ("Google", "located_in", "California", 0.85),
    ]

    for s, r, d, conf in edges:
        store.observe_edge(s, r, d, conf, 1000)
        edge_id = store.edge_id(s, r, d)
        store.enqueue_edge_source(edge_id, tid, 1000)

    store.flush_if_needed(max_ops=1)

    # Get extractions
    extractions = store.get_turn_extractions("session-1", 0)
    assert len(extractions) == 2

    # Verify content
    triples = [(e[0], e[1], e[2]) for e in extractions]
    assert ("Alice", "works_at", "Google") in triples
    assert ("Google", "located_in", "California") in triples


def test_get_conversation(store):
    """Test retrieving full conversation by session"""
    # Store conversation with 5 turns
    for i in range(5):
        store.enqueue_turn(f"Turn {i} text", "session-1", i, 1000 + i)

    store.flush_if_needed(max_ops=1)

    # Get conversation
    conversation = store.get_conversation("session-1")
    assert len(conversation) == 5

    # Verify order
    for i, (turn_id, text, ts) in enumerate(conversation):
        assert turn_id == i
        assert text == f"Turn {i} text"


def test_text_truncation(store):
    """Test that long text is truncated to 2000 chars"""
    long_text = "x" * 5000
    tid = store.enqueue_turn(long_text, "session-1", 0, 1000)
    store.flush_if_needed(max_ops=1)

    # Verify truncated
    cur = store.sql.cursor()
    result = cur.execute(
        "SELECT text FROM conversation_turn WHERE id = ?",
        (tid,)
    ).fetchone()

    assert len(result[0]) == 2000


def test_foreign_key_cascade(store):
    """Test that deleting turn cascades to edge_source"""
    # Store turn and link
    tid = store.enqueue_turn("Test", "session-1", 0, 1000)
    store.observe_edge("A", "r", "B", 0.9, 1000)
    edge_id = store.edge_id("A", "r", "B")
    store.enqueue_edge_source(edge_id, tid, 1000)
    store.flush_if_needed(max_ops=1)

    # Delete turn
    cur = store.sql.cursor()
    cur.execute("DELETE FROM conversation_turn WHERE id = ?", (tid,))
    store.sql.commit()

    # edge_source should be gone (cascade)
    count = cur.execute(
        "SELECT COUNT(*) FROM edge_source WHERE turn_id = ?",
        (tid,)
    ).fetchone()[0]
    assert count == 0


def test_provenance_ordering(store):
    """Test that provenance returns most recent first"""
    # Create 3 turns with same fact at different times
    edge_id = store.edge_id("I", "name", "Charlie")
    store.observe_edge("I", "name", "Charlie", 0.95, 1000)

    for i, ts in enumerate([1000, 2000, 3000]):
        tid = store.enqueue_turn(f"I'm Charlie {i}", "session-1", i, ts)
        store.enqueue_edge_source(edge_id, tid, ts)

    store.flush_if_needed(max_ops=1)

    # Get provenance
    provenance = store.get_edge_provenance(edge_id)
    assert len(provenance) == 3

    # Should be ordered by extracted_at DESC (most recent first)
    timestamps = [p[3] for p in provenance]
    assert timestamps == [3000, 2000, 1000]


def test_empty_provenance(store):
    """Test querying provenance for edge with no sources"""
    # Create edge without provenance link
    store.observe_edge("X", "rel", "Y", 0.9, 1000)
    edge_id = store.edge_id("X", "rel", "Y")
    store.flush_if_needed(max_ops=1)

    # Should return empty list
    provenance = store.get_edge_provenance(edge_id)
    assert provenance == []

    # Count should be 0
    count = store.get_edge_sources_count(edge_id)
    assert count == 0


def test_session_isolation(store):
    """Test that different sessions don't interfere"""
    # Two sessions, same turn_id
    tid1 = store.enqueue_turn("Session 1 text", "session-1", 0, 1000)
    tid2 = store.enqueue_turn("Session 2 text", "session-2", 0, 1000)

    assert tid1 != tid2  # Different hashes

    store.flush_if_needed(max_ops=1)

    # Each session should have 1 turn
    conv1 = store.get_conversation("session-1")
    conv2 = store.get_conversation("session-2")

    assert len(conv1) == 1
    assert len(conv2) == 1
    assert conv1[0][1] == "Session 1 text"
    assert conv2[0][1] == "Session 2 text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])