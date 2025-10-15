#!/usr/bin/env python3
"""Unit tests for turn prosody metadata storage in MemoryStore.

Tests the set_turn_prosody and get_turn_prosody methods with various edge cases.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


class TestTurnProsodyStorage:
    """Test turn prosody metadata storage and retrieval."""

    def test_set_and_get_basic_prosody(self, memory_store):
        """Test basic storage and retrieval of prosody certainty."""
        session_id = "test_session"
        turn_id = 1
        certainty = 0.85
        
        # Store prosody
        memory_store.set_turn_prosody(session_id, turn_id, certainty)
        
        # Retrieve prosody
        retrieved_certainty, retrieved_meta = memory_store.get_turn_prosody(session_id, turn_id)
        
        assert retrieved_certainty == certainty
        assert retrieved_meta == {}

    def test_set_and_get_prosody_with_metadata(self, memory_store):
        """Test storage and retrieval of prosody with metadata."""
        session_id = "test_session"
        turn_id = 2
        certainty = 0.72
        meta = {"pitch_mean": 150.5, "energy": 0.8, "duration": 2.3}
        
        # Store prosody with metadata
        memory_store.set_turn_prosody(session_id, turn_id, certainty, meta)
        
        # Retrieve prosody
        retrieved_certainty, retrieved_meta = memory_store.get_turn_prosody(session_id, turn_id)
        
        assert retrieved_certainty == certainty
        assert retrieved_meta == meta

    def test_get_missing_prosody_returns_defaults(self, memory_store):
        """Test that missing prosody data returns defaults."""
        session_id = "nonexistent_session"
        turn_id = 999
        
        # Retrieve missing prosody
        certainty, meta = memory_store.get_turn_prosody(session_id, turn_id)
        
        assert certainty == 0.5  # Default baseline
        assert meta == {}       # Default empty meta

    def test_update_existing_prosody(self, memory_store):
        """Test updating existing prosody data."""
        session_id = "test_session"
        turn_id = 3
        
        # Store initial prosody
        memory_store.set_turn_prosody(session_id, turn_id, 0.6, {"initial": True})
        
        # Update with new values
        memory_store.set_turn_prosody(session_id, turn_id, 0.9, {"updated": True})
        
        # Retrieve updated prosody
        certainty, meta = memory_store.get_turn_prosody(session_id, turn_id)
        
        assert certainty == 0.9
        assert meta == {"updated": True}

    def test_certainty_clamping(self, memory_store):
        """Test that certainty values are clamped to [0,1] range."""
        session_id = "test_session"
        turn_id = 4
        
        # Store out-of-range values
        memory_store.set_turn_prosody(session_id, turn_id, -0.5)  # Below 0
        certainty, _ = memory_store.get_turn_prosody(session_id, turn_id)
        assert certainty == 0.0
        
        memory_store.set_turn_prosody(session_id, turn_id, 1.5)  # Above 1
        certainty, _ = memory_store.get_turn_prosody(session_id, turn_id)
        assert certainty == 1.0

    def test_multiple_sessions_and_turns(self, memory_store):
        """Test storage across multiple sessions and turns."""
        # Store prosody for different sessions/turns
        memory_store.set_turn_prosody("session1", 1, 0.1)
        memory_store.set_turn_prosody("session1", 2, 0.2)
        memory_store.set_turn_prosody("session2", 1, 0.8)
        
        # Verify retrieval
        assert memory_store.get_turn_prosody("session1", 1)[0] == 0.1
        assert memory_store.get_turn_prosody("session1", 2)[0] == 0.2
        assert memory_store.get_turn_prosody("session2", 1)[0] == 0.8
        assert memory_store.get_turn_prosody("session1", 3)[0] == 0.5  # Default

    def test_invalid_certainty_value_in_db(self, memory_store):
        """Test handling of invalid certainty values in database."""
        session_id = "test_session"
        turn_id = 5
        
        # Manually insert invalid data to test robustness
        cur = memory_store.sql.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO turn_meta(session_id, turn_id, key, value)
            VALUES(?, ?, 'prosody_certainty', ?)
        """, (session_id, turn_id, "invalid_number"))
        memory_store.sql.commit()
        
        # Should return default on invalid data
        certainty, meta = memory_store.get_turn_prosody(session_id, turn_id)
        assert certainty == 0.5  # Default fallback
        assert meta == {}

    def test_invalid_metadata_json_in_db(self, memory_store):
        """Test handling of invalid JSON metadata in database."""
        session_id = "test_session"
        turn_id = 6
        
        # Store valid certainty first
        memory_store.set_turn_prosody(session_id, turn_id, 0.75)
        
        # Manually insert invalid JSON
        cur = memory_store.sql.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO turn_meta(session_id, turn_id, key, value)
            VALUES(?, ?, 'prosody_meta', ?)
        """, (session_id, turn_id, "invalid json {"))
        memory_store.sql.commit()
        
        # Should return default empty meta on invalid JSON
        certainty, meta = memory_store.get_turn_prosody(session_id, turn_id)
        assert certainty == 0.75
        assert meta == {}

    def test_non_dict_metadata_in_db(self, memory_store):
        """Test handling of non-dict metadata in database."""
        session_id = "test_session"
        turn_id = 7
        
        # Store valid certainty first
        memory_store.set_turn_prosody(session_id, turn_id, 0.75)
        
        # Manually insert non-dict JSON
        cur = memory_store.sql.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO turn_meta(session_id, turn_id, key, value)
            VALUES(?, ?, 'prosody_meta', ?)
        """, (session_id, turn_id, '"not a dict"'))
        memory_store.sql.commit()
        
        # Should return default empty meta for non-dict
        certainty, meta = memory_store.get_turn_prosody(session_id, turn_id)
        assert certainty == 0.75
        assert meta == {}

    def test_set_metadata_only(self, memory_store):
        """Test setting metadata without certainty (should not happen in practice but test robustness)."""
        session_id = "test_session"
        turn_id = 8
        meta = {"test": "value"}
        
        # Manually insert metadata only
        cur = memory_store.sql.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO turn_meta(session_id, turn_id, key, value)
            VALUES(?, ?, 'prosody_meta', ?)
        """, (session_id, turn_id, '{"test": "value"}'))
        memory_store.sql.commit()
        
        # Should return default certainty with metadata
        certainty, retrieved_meta = memory_store.get_turn_prosody(session_id, turn_id)
        assert certainty == 0.5  # Default
        assert retrieved_meta == meta

    def test_certainty_precision_formatting(self, memory_store):
        """Test that certainty is stored with 3 decimal places."""
        session_id = "test_session"
        turn_id = 9
        certainty = 0.1234567  # Many decimal places
        
        memory_store.set_turn_prosody(session_id, turn_id, certainty)
        
        # Check raw stored value
        cur = memory_store.sql.cursor()
        result = cur.execute("""
            SELECT value FROM turn_meta WHERE session_id = ? AND turn_id = ? AND key = 'prosody_certainty'
        """, (session_id, turn_id)).fetchone()
        
        assert result[0] == "0.123"  # Should be formatted to 3 decimal places
        
        # Retrieval should still work
        retrieved_certainty, _ = memory_store.get_turn_prosody(session_id, turn_id)
        assert retrieved_certainty == 0.123
