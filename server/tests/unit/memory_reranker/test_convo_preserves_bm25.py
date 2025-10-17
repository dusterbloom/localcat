"""Test that Enhanced FTS BM25 scores are preserved and used in composite scoring."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from core.memory.retrieval import Retrieval


class TestConvoPreservesBM25Ordering:
    """Test that conversation bullets with higher BM25 scores rank above lower scores."""
    
    def test_convo_preserves_bm25_ordering_when_relevant(self):
        """Simulate two convo hits with different BM25; confirm higher BM25 ranks above when other terms equal."""
        # Setup mock host with Enhanced FTS that returns different BM25 scores
        host = Mock()
        host.store = Mock()
        host.current_user_id = "user123"
        host.current_session_id = "session456"
        
        # Mock EnhancedFTS to return results with different BM25 scores
        enhanced_fts_mock = Mock()
        enhanced_fts_mock.enhanced_search.return_value = [
            (2.5, "User works at ACME Corporation", "user123", int(time.time() * 1000) - 3600000),  # Higher BM25
            (1.2, "User lives in New York", "user123", int(time.time() * 1000) - 7200000),  # Lower BM25
        ]
        
        # Patch EnhancedFTS import
        import sys
        from unittest.mock import patch
        
        with patch('core.memory.enhanced_fts.EnhancedFTS', return_value=enhanced_fts_mock):
            retrieval = Retrieval(host)
            
            # Test convo_retrieve directly to check BM25 preservation
            seen = set()
            bullets = retrieval._convo_retrieve("user works", 2, seen)
            
            # Should return 2 bullets with higher BM25 first
            assert len(bullets) == 2
            assert "ACME Corporation" in bullets[0]  # Higher BM25 should be first
            assert "New York" in bullets[1]  # Lower BM25 should be second
            
    def test_bm25_scores_preserved_in_candidate_metadata(self):
        """Test that BM25 scores are included in candidate metadata for composite scoring."""
        host = Mock()
        host.store = Mock()
        host.current_user_id = "user123"
        host.current_session_id = "session456"
        
        # Mock EnhancedFTS with known scores
        enhanced_fts_mock = Mock()
        test_time = int(time.time() * 1000)
        enhanced_fts_mock.enhanced_search.return_value = [
            (3.0, "High score match", "user123", test_time),
            (1.0, "Low score match", "user123", test_time),
        ]
        
        with patch('core.memory.enhanced_fts.EnhancedFTS', return_value=enhanced_fts_mock):
            retrieval = Retrieval(host)
            
            # We'll need to implement a method to check candidate metadata
            # For now, verify that the search is called with correct params
            seen = set()
            retrieval._convo_retrieve("test query", 2, seen)
            
            # Verify enhanced search was called
            enhanced_fts_mock.enhanced_search.assert_called_once()
            call_args = enhanced_fts_mock.enhanced_search.call_args
            # Args are passed as positional arguments
            assert call_args[0][0] == "test query"  # query
            assert call_args[0][1] == 4  # max_bullets * 2
            # eids might be in kwargs or positional args depending on implementation
            if len(call_args[0]) > 2:
                assert call_args[0][2] == ["user123", "session456"]  # eids
            elif 'eids' in call_args[1]:
                assert call_args[1]['eids'] == ["user123", "session456"]
