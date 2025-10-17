"""Integration test for final selection updating edge usage."""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock
from core.memory.retrieval import Retrieval
from core.memory.memory_store import MemoryStore, Paths


class TestFinalSelectionUpdatesUsage:
    """Test that final selection properly updates edge usage."""
    
    def test_final_selection_updates_edge_usage(self):
        """Run a retrieval cycle that selects a known graph bullet; verify edge_usage updated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test database
            db_path = os.path.join(temp_dir, "test_memory.db")
            paths = Paths(sqlite_path=db_path, lmdb_dir=None)
            store = MemoryStore(paths)
            
            # Setup mock host
            host = Mock()
            host.store = store
            host.current_user_id = "test_user"
            host.current_session_id = "test_session"
            host.entity_index = {}
            host.recency_buffer = []
            
            # Create test edge in database
            edge_id = "user_works_at_company"
            current_time = int(time.time() * 1000)
            
            # Insert edge directly into database
            store.sql.execute("""
                INSERT INTO edge (id, src, rel, dst, weight, pos, neg, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (edge_id, "user", "works_at", "Company", 0.8, 5, 1, 1, current_time))
            store.sql.commit()
            
            # Setup entity index
            host.entity_index["user"] = [("user", "works_at", "Company")]
            
            # Mock provenance to allow access
            def mock_get_edges_provenance_batch(edge_ids):
                return {edge_id: [("test text", "test_session", 1, current_time)]}
            
            host.store.get_edges_provenance_batch = mock_get_edges_provenance_batch
            host.store.are_sessions_owned_by_user_batch = Mock(return_value={"test_session"})
            
            # Test retrieval
            retrieval = Retrieval(host)
            
            # Initial usage should be 0
            access_count, last_accessed = store.get_edge_usage(edge_id)
            assert access_count == 0
            assert last_accessed == 0
            
            # Run retrieval that should select the graph bullet
            bullets = retrieval.retrieve("user works", [], 1)
            
            # Should have selected the graph bullet
            assert len(bullets) > 0
            assert "[graph]" in bullets[0]
            
            # Usage should be updated
            access_count, last_accessed = store.get_edge_usage(edge_id)
            assert access_count == 1
            assert last_accessed > 0
            
    def test_multiple_selections_increment_usage(self):
        """Test that multiple selections properly increment usage count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test database
            db_path = os.path.join(temp_dir, "test_memory.db")
            paths = Paths(sqlite_path=db_path, lmdb_dir=None)
            store = MemoryStore(paths)
            
            # Setup mock host
            host = Mock()
            host.store = store
            host.current_user_id = "test_user"
            host.current_session_id = "test_session"
            host.entity_index = {}
            host.recency_buffer = []
            
            # Create test edge
            edge_id = "user_lives_in_city"
            current_time = int(time.time() * 1000)
            
            store.sql.execute("""
                INSERT INTO edge (id, src, rel, dst, weight, pos, neg, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (edge_id, "user", "lives_in", "City", 0.7, 3, 0, 1, current_time))
            store.sql.commit()
            
            host.entity_index["user"] = [("user", "lives_in", "City")]
            
            # Mock provenance
            def mock_get_edges_provenance_batch(edge_ids):
                return {edge_id: [("test text", "test_session", 1, current_time)]}
            
            host.store.get_edges_provenance_batch = mock_get_edges_provenance_batch
            host.store.are_sessions_owned_by_user_batch = Mock(return_value={"test_session"})
            
            retrieval = Retrieval(host)
            
            # Run retrieval multiple times
            for i in range(3):
                bullets = retrieval.retrieve("where user lives", [], 1)
                assert len(bullets) > 0
                
                # Check usage count increments
                access_count, _ = store.get_edge_usage(edge_id)
                assert access_count == i + 1
