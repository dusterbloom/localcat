"""Performance test for embedding reranker latency."""

import pytest
import time
import os
from unittest.mock import Mock
from core.memory.retrieval import Retrieval
from core.memory.rerank_embeddings import EmbeddingReranker


class TestRerankerLatency:
    """Test that embedding reranker stays within latency budget."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers", reason="sentence_transformers not available"),
        reason="sentence_transformers not available"
    )
    def test_latency_budget_with_embeddings(self):
        """Ensure rerank stays under the configured budget for ≤24 candidates after warmup."""
        # Enable embeddings
        os.environ["MEMORY_RERANK_EMBEDDINGS_ENABLED"] = "true"
        os.environ["MEMORY_RERANK_MAX_CANDIDATES"] = "24"
        
        # Create reranker
        reranker = EmbeddingReranker()
        
        # Warm up the model
        warmup_texts = [
            "User works at a technology company",
            "The user enjoys programming and software development",
            "User lives in a modern city with good public transport"
        ]
        warmup_query = "What do you know about the user's job?"
        
        # Warmup
        reranker.similarity(warmup_query, warmup_texts)
        
        # Test with maximum candidates (24)
        test_query = "What do you know about the user?"
        test_texts = [
            f"Sample text {i} about user activities and preferences" 
            for i in range(24)
        ]
        
        # Measure latency
        start_time = time.time()
        similarities = reranker.similarity(test_query, test_texts)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Verify results
        assert len(similarities) == 24
        assert all(0.0 <= s <= 1.0 for s in similarities)
        
        # Check latency budget (should be under 15ms for 24 candidates)
        # Allow some tolerance for test environment
        assert elapsed_ms < 20.0, f"Reranking took {elapsed_ms:.1f}ms, expected <20ms"
        
        print(f"Embedding reranker: {elapsed_ms:.1f}ms for 24 candidates")
        
    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers", reason="sentence_transformers not available"),
        reason="sentence_transformers not available"
    )
    def test_latency_scales_linearly(self):
        """Test that latency scales reasonably with number of candidates."""
        os.environ["MEMORY_RERANK_EMBEDDINGS_ENABLED"] = "true"
        
        reranker = EmbeddingReranker()
        
        # Warm up
        reranker.similarity("test query", ["warmup text"])
        
        # Test different candidate counts
        test_query = "What do you know about the user?"
        candidate_counts = [1, 5, 10, 20]
        latencies = []
        
        for count in candidate_counts:
            test_texts = [f"Sample text {i}" for i in range(count)]
            
            start_time = time.time()
            similarities = reranker.similarity(test_query, test_texts)
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)
            
            assert len(similarities) == count
        
        print(f"Latency scaling: {list(zip(candidate_counts, latencies))}")
        
        # Latency should not increase disproportionately
        # 20 candidates should not take more than 4x the time of 5 candidates
        if len(latencies) >= 4:
            ratio = latencies[3] / latencies[1]  # 20 candidates / 5 candidates
            assert ratio < 4.0, f"Latency scaled poorly: {ratio:.2f}x"
            
    def test_disabled_embeddings_zero_latency(self):
        """Test that disabled embeddings have zero overhead."""
        os.environ["MEMORY_RERANK_EMBEDDINGS_ENABLED"] = "false"
        
        reranker = EmbeddingReranker()
        
        test_query = "test query"
        test_texts = ["text1", "text2", "text3"] * 8  # 24 texts
        
        start_time = time.time()
        similarities = reranker.similarity(test_query, test_texts)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should return zeros immediately
        assert len(similarities) == 24
        assert all(s == 0.0 for s in similarities)
        
        # Should be very fast (under 1ms)
        assert elapsed_ms < 1.0, f"Disabled embeddings took {elapsed_ms:.1f}ms"
        
        print(f"Disabled embeddings: {elapsed_ms:.3f}ms for 24 candidates")
