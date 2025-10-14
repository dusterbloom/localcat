"""
Integration tests for retrieval fusion with semantic source support.

Tests the integration of the optional semantic source with the existing retrieval
system, ensuring proper fallback behavior when semantic is disabled.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import the modules we need to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.memory.retrieval import Retrieval, Candidate


class TestRetrievalFusionWithSemanticEnabled:
    """Test retrieval fusion when semantic source is enabled."""
    
    @pytest.fixture
    def mock_host(self):
        """Create a mock host for testing."""
        host = Mock()
        host.entity_index = {}
        host.recency_buffer = []
        host.store = Mock()
        host.store.get_edge_usage = Mock(return_value=(0, 0))
        return host
    
    @pytest.fixture
    def retrieval_with_semantic(self, mock_host):
        """Create a Retrieval instance with semantic enabled."""
        with patch.dict(os.environ, {
            'MEMORY_SOURCES': 'graph,convo,summary,semantic',
            'MEMORY_WEIGHT_SEMANTIC': '0.2'
        }):
            retrieval = Retrieval(mock_host)
            return retrieval
    
    def test_semantic_source_enabled(self, retrieval_with_semantic):
        """Test that semantic source is properly enabled."""
        enabled_sources = ["graph", "convo", "summary", "semantic"]
        source_priority = retrieval_with_semantic._get_source_priority("test query")
        
        # Semantic should be included in priority
        assert "semantic" in source_priority, "Semantic source should be in priority list"
    
    def test_semantic_collect_candidates(self, retrieval_with_semantic):
        """Test semantic candidate collection."""
        # Mock semantic search results
        with patch.object(retrieval_with_semantic.host.store, 'semantic_search') as mock_search:
            mock_search.return_value = [
                ("User enjoys hiking and outdoor activities", 0.85, int(time.time() * 1000)),
                ("User likes reading technical books", 0.75, int(time.time() * 1000))
            ]
            
            candidates = retrieval_with_semantic._semantic_collect_candidates(
                query="what are my hobbies",
                max_bullets=3,
                seen=set()
            )
            
            assert len(candidates) == 2, "Should collect 2 semantic candidates"
            assert all(c.source == "semantic" for c in candidates), "All candidates should be semantic"
            assert all(c.meta.get("semantic_score") > 0 for c in candidates), "Should have semantic scores"
    
    def test_semantic_in_composite_scoring(self, retrieval_with_semantic):
        """Test that semantic candidates are properly scored."""
        semantic_candidate = Candidate(
            text="User enjoys hiking",
            source="semantic",
            score_hint=0.8,
            ts=int(time.time() * 1000),
            meta={"semantic_score": 0.85}
        )
        
        query = "what are my hobbies"
        score, components = retrieval_with_semantic._composite_score(query, semantic_candidate)
        
        # Check that semantic source weight is applied
        assert "wsrc" in components, "Should have source weight component"
        assert components["wsrc"] > 0, "Semantic source should have positive weight"
        
        # Check that semantic score is used for confidence
        assert "wconf" in components, "Should have confidence component"
        assert components["wconf"] > 0, "Should have positive confidence"
    
    def test_semantic_priority_in_query_routing(self, retrieval_with_semantic):
        """Test semantic source priority for different query types."""
        # Semantic queries should prioritize semantic source
        semantic_queries = [
            "what do you know about my personality",
            "tell me about my interests",
            "what are my preferences",
            "describe my character traits"
        ]
        
        for query in semantic_queries:
            priority = retrieval_with_semantic._get_source_priority(query)
            assert "semantic" in priority, f"Semantic should be in priority for query: {query}"
            # Semantic should be high priority for semantic queries
            semantic_rank = priority.index("semantic")
            assert semantic_rank < len(priority) - 1, "Semantic should not be lowest priority"


class TestRetrievalFusionWithoutSemanticIdenticalToBaseline:
    """Test that behavior is identical to baseline when semantic is disabled."""
    
    @pytest.fixture
    def mock_host(self):
        """Create a mock host for testing."""
        host = Mock()
        host.entity_index = {}
        host.recency_buffer = []
        host.store = Mock()
        host.store.get_edge_usage = Mock(return_value=(0, 0))
        return host
    
    @pytest.fixture
    def retrieval_baseline(self, mock_host):
        """Create a baseline Retrieval instance (semantic disabled)."""
        with patch.dict(os.environ, {
            'MEMORY_SOURCES': 'graph,convo,summary',  # No semantic
            'MEMORY_WEIGHT_GRAPH': '0.3',
            'MEMORY_WEIGHT_CONVO': '0.4',
            'MEMORY_WEIGHT_SUMMARY': '0.3'
        }, clear=True):
            retrieval = Retrieval(mock_host)
            return retrieval
    
    @pytest.fixture
    def retrieval_with_semantic(self, mock_host):
        """Create a Retrieval instance with semantic (for comparison)."""
        with patch.dict(os.environ, {
            'MEMORY_SOURCES': 'graph,convo,summary,semantic',
            'MEMORY_WEIGHT_GRAPH': '0.3',
            'MEMORY_WEIGHT_CONVO': '0.4', 
            'MEMORY_WEIGHT_SUMMARY': '0.3',
            'MEMORY_WEIGHT_SEMANTIC': '0.0'  # Zero weight to disable effect
        }):
            retrieval = Retrieval(mock_host)
            return retrieval
    
    def test_semantic_disabled_no_candidates(self, retrieval_baseline):
        """Test that semantic source is ignored when disabled."""
        enabled_sources = ["graph", "convo", "summary"]
        source_priority = retrieval_baseline._get_source_priority("test query")
        
        # Semantic should not be in priority list
        assert "semantic" not in source_priority, "Semantic should not be in priority when disabled"
    
    def test_identical_source_priority_without_semantic(self, retrieval_baseline):
        """Test that source priority is identical without semantic."""
        queries = [
            "where do I live",
            "what's my job", 
            "tell me about my family",
            "what are my hobbies"
        ]
        
        for query in queries:
            priority = retrieval_baseline._get_source_priority(query)
            
            # Should only contain traditional sources
            expected_sources = {"graph", "convo", "summary"}
            actual_sources = set(priority)
            
            assert actual_sources == expected_sources, \
                f"Query '{query}' should only have traditional sources: {actual_sources}"
    
    def test_budget_allocation_without_semantic(self, retrieval_baseline):
        """Test that budget allocation ignores semantic when disabled."""
        enabled_sources = ["graph", "convo", "summary"]
        budget = retrieval_baseline._allocate_budget(max_bullets=3, enabled_sources=enabled_sources)
        
        # Semantic should not be in budget
        assert "semantic" not in budget, "Semantic should not be in budget allocation"
        assert sum(budget.values()) <= 3, "Total budget should not exceed max_bullets"
    
    @patch('core.memory.retrieval.os.getenv')
    def test_environment_variable_fallback(self, mock_getenv, mock_host):
        """Test that environment variables have proper fallbacks."""
        # Mock getenv to return None for semantic settings
        def mock_env_side_effect(key, default=None):
            if key == "MEMORY_SOURCES":
                return "graph,convo,summary"  # No semantic
            elif key.startswith("MEMORY_WEIGHT_"):
                if key == "MEMORY_WEIGHT_SEMANTIC":
                    return None  # Should use default
            return default
        
        mock_getenv.side_effect = mock_env_side_effect
        
        retrieval = Retrieval(mock_host)
        
        # Should still work with fallback values
        assert hasattr(retrieval, 'source_weights')
        assert retrieval.source_weights.get("MEMORY_WEIGHT_SEMANTIC", 0.1) == 0.1  # Default fallback


class TestSemanticFallbackBehavior:
    """Test semantic source fallback behavior when sidecar is unavailable."""
    
    @pytest.fixture
    def mock_host(self):
        """Create a mock host for testing."""
        host = Mock()
        host.entity_index = {}
        host.recency_buffer = []
        host.store = Mock()
        host.store.get_edge_usage = Mock(return_value=(0, 0))
        return host
    
    @pytest.fixture
    def retrieval_with_semantic(self, mock_host):
        """Create a Retrieval instance with semantic enabled."""
        with patch.dict(os.environ, {
            'MEMORY_SOURCES': 'graph,convo,summary,semantic'
        }):
            retrieval = Retrieval(mock_host)
            return retrieval
    
    def test_semantic_collect_fallback_on_error(self, retrieval_with_semantic):
        """Test fallback behavior when semantic search fails."""
        # Mock semantic search to raise exception
        with patch.object(retrieval_with_semantic.host.store, 'semantic_search') as mock_search:
            mock_search.side_effect = Exception("Semantic service unavailable")
            
            candidates = retrieval_with_semantic._semantic_collect_candidates(
                query="test query",
                max_bullets=3,
                seen=set()
            )
            
            # Should return empty list on error
            assert candidates == [], "Should return empty list on semantic search error"
    
    def test_semantic_weight_zero_disables_effect(self, retrieval_with_semantic):
        """Test that zero semantic weight effectively disables semantic."""
        # Override semantic weight to zero
        retrieval_with_semantic.source_weights["MEMORY_WEIGHT_SEMANTIC"] = 0.0
        
        semantic_candidate = Candidate(
            text="Semantic result",
            source="semantic",
            score_hint=0.8,
            ts=int(time.time() * 1000),
            meta={"semantic_score": 0.9}
        )
        
        query = "test query"
        score, components = retrieval_with_semantic._composite_score(query, semantic_candidate)
        
        # Source weight should be zero
        assert components["wsrc"] == 0.0, "Zero semantic weight should disable source component"
    
    def test_graceful_degradation_without_semantic_service(self, retrieval_with_semantic):
        """Test graceful degradation when semantic service is missing."""
        # Mock store without semantic_search method
        retrieval_with_semantic.host.store = Mock()
        delattr(retrieval_with_semantic.host.store, 'semantic_search')
        
        candidates = retrieval_with_semantic._semantic_collect_candidates(
            query="test query",
            max_bullets=3,
            seen=set()
        )
        
        # Should handle missing method gracefully
        assert isinstance(candidates, list), "Should return list even when semantic service missing"
        assert len(candidates) == 0, "Should return empty list when semantic service missing"


if __name__ == "__main__":
    pytest.main([__file__])
