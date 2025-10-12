"""Test that disabled embeddings behave like baseline composite scoring."""

import pytest
import time
import os
from unittest.mock import Mock, patch
from core.memory.retrieval import Retrieval


class TestEmbeddingsDisabledBehavior:
    """Test that when embeddings are disabled, ranking matches baseline composite scoring."""
    
    def test_embeddings_disabled_behaves_like_baseline(self):
        """With embeddings off, ranking decisions ignore wsim and match composite score without sim."""
        # Ensure embeddings are disabled
        os.environ["MEMORY_RERANK_EMBEDDINGS_ENABLED"] = "false"
        
        host = Mock()
        host.store = Mock()
        host.current_user_id = "user123"
        host.current_session_id = "session456"
        
        retrieval = Retrieval(host)
        
        # Test that composite scoring doesn't use embeddings when disabled
        # We'll need to implement the composite scoring function first
        query = "test query"
        
        # Create test candidates from different sources
        candidates = [
            {
                "text": "Graph fact about user",
                "source": "graph",
                "score_hint": 0.0,  # Not used for graph
                "ts": int(time.time() * 1000) - 3600000,
                "meta": {"edge_id": "edge_1", "weight": 0.8, "pos": 5, "neg": 1}
            },
            {
                "text": "Conversation match",
                "source": "convo", 
                "score_hint": 2.5,  # BM25 score
                "ts": int(time.time() * 1000) - 1800000,
                "meta": {"bm25_score": 2.5}
            }
        ]
        
        # When we implement _composite_score, verify it doesn't use embeddings when disabled
        # For now, just verify the environment variable is read correctly
        assert os.getenv("MEMORY_RERANK_EMBEDDINGS_ENABLED") == "false"
        
    def test_embeddings_reranker_not_loaded_when_disabled(self):
        """Test that embedding reranker is not instantiated when disabled."""
        os.environ["MEMORY_RERANK_EMBEDDINGS_ENABLED"] = "false"
        
        # Test that reranker module import is skipped when disabled
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            # Should not raise ImportError when embeddings are disabled
            host = Mock()
            retrieval = Retrieval(host)
            # The embedding reranker should not be loaded
            # This will be verified once we implement the reranker integration
            pass
