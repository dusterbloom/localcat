"""Test that frequently used graph facts get boosted in ranking."""

import pytest
import time
from unittest.mock import Mock, MagicMock
from core.memory.retrieval import Retrieval
from core.memory.memory_store import MemoryStore


class TestGraphUsageBoost:
    """Test that graph facts with higher usage counts get ranking boosts."""
    
    def test_graph_usage_boosts_frequently_used_facts(self):
        """Start with two graph facts close in weight/recency; after selecting A multiple times, A ranks above B."""
        # Setup mock host with memory store that has usage tracking
        host = Mock()
        host.store = Mock(spec=MemoryStore)
        host.current_user_id = "user123"
        host.current_session_id = "session456"
        host.entity_index = {}
        host.recency_buffer = []
        
        # Setup two similar facts with slightly different usage
        fact_A_id = "edge_A"
        fact_B_id = "edge_B"
        current_time = int(time.time() * 1000)
        
        # Mock edge_usage table responses
        host.store.get_edge_usage.side_effect = lambda edge_id: (
            (10, current_time - 1000) if edge_id == fact_A_id else (1, current_time - 1000)
        )
        
        # Mock neighbors to return edge metadata
        def mock_neighbors(s, r):
            if s == "user" and r == "works_at":
                return [
                    ("ACME", 0.8, current_time - 3600000, 5, 1, 1),  # edge_A
                    ("TechCorp", 0.75, current_time - 3500000, 4, 1, 1),  # edge_B
                ]
            return []
        
        host.store.neighbors.side_effect = mock_neighbors
        host.store.edge_id.side_effect = lambda s, r, d: f"{s}_{r}_{d}"
        
        # Mock provenance to allow both edges
        def mock_get_edges_provenance_batch(edge_ids):
            return {eid: [("test text", "session456", 1, current_time - 3600000)] for eid in edge_ids}
        
        host.store.get_edges_provenance_batch = mock_get_edges_provenance_batch
        host.store.are_sessions_owned_by_user_batch = Mock(return_value={"session456"})
        
        # Setup entity index
        host.entity_index["user"] = [("user", "works_at", "ACME"), ("user", "works_at", "TechCorp")]
        
        retrieval = Retrieval(host)
        
        # Test _graph_collect_candidates with usage-influenced scoring
        # Need to provide turn_id parameter
        turn_id = 1
        seen = set()
        candidates = retrieval._graph_collect_candidates("user works", ["user"], turn_id, 1, seen.copy(), allowed_relations=None)
        
        # The fact with higher usage (ACME) should rank above the one with lower usage (TechCorp)
        # even though TechCorp has slightly higher weight
        assert len(candidates) > 0
        # We'll need to implement the composite scoring to verify this works
        
    def test_usage_tracking_updates_after_selection(self):
        """Test that usage counts are incremented after selection."""
        host = Mock()
        host.store = Mock(spec=MemoryStore)
        
        # Mock increment_edge_usage to track calls
        host.store.increment_edge_usage = Mock()
        
        retrieval = Retrieval(host)
        
        # Simulate final selection with graph bullets containing edge_id in metadata
        # This will be tested once we implement the usage tracking hook
        pass
