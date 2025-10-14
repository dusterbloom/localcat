"""
Unit tests for memory retrieval hardening features.

Tests the enhanced composite scoring, token budget enforcement, cross-source deduplication,
and strengthened greeting/intent gating implemented in the retrieval hardening spec.
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


class TestCompositeScoringDeterministicOrdering:
    """Test that composite scoring produces deterministic ordering."""
    
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
    def retrieval(self, mock_host):
        """Create a Retrieval instance for testing."""
        with patch.dict(os.environ, {
            'MEMORY_RERANK_WEIGHTS': '{"wsrc": 0.2, "wconf": 0.3, "wrec": 0.25, "wuse": 0.1, "wsim": 0.1, "wdiv": 0.05}',
            'MEMORY_WEIGHT_GRAPH': '0.4',
            'MEMORY_WEIGHT_CONVO': '0.3', 
            'MEMORY_WEIGHT_SUMMARY': '0.2',
            'MEMORY_WEIGHT_SEMANTIC': '0.1'
        }):
            retrieval = Retrieval(mock_host)
            return retrieval
    
    def test_deterministic_ordering_same_inputs(self, retrieval):
        """Test that same inputs produce same ordering multiple times."""
        query = "What do you know about where I live?"
        
        candidates = [
            Candidate(
                text="User lives in New York City",
                source="graph", 
                score_hint=0.8,
                ts=int(time.time() * 1000) - 86400000,  # 1 day ago
                meta={"weight": 0.9, "pos": 3, "neg": 0, "edge_id": "edge1"}
            ),
            Candidate(
                text="I mentioned living in NYC yesterday",
                source="convo",
                score_hint=1.2,  # BM25 score
                ts=int(time.time() * 1000) - 3600000,  # 1 hour ago
                meta={"bm25_score": 1.2}
            ),
            Candidate(
                text="User resides in New York",
                source="summary", 
                score_hint=0.5,
                ts=int(time.time() * 1000) - 172800000,  # 2 days ago
                meta={}
            )
        ]
        
        # Score candidates twice
        results1 = []
        results2 = []
        
        for candidate in candidates:
            score1, _ = retrieval._composite_score(query, candidate, other_candidates=candidates)
            score2, _ = retrieval._composite_score(query, candidate, other_candidates=candidates)
            results1.append(score1)
            results2.append(score2)
        
        # Should be identical
        assert results1 == results2, "Composite scoring should be deterministic"
        
        # Order should be consistent
        indexed_results1 = list(zip(results1, candidates))
        indexed_results2 = list(zip(results2, candidates))
        
        indexed_results1.sort(key=lambda x: x[0], reverse=True)
        indexed_results2.sort(key=lambda x: x[0], reverse=True)
        
        # Extract candidate texts in order
        order1 = [candidate.text for _, candidate in indexed_results1]
        order2 = [candidate.text for _, candidate in indexed_results2]
        
        assert order1 == order2, "Candidate ordering should be deterministic"
    
    def test_source_weights_configuration(self, retrieval):
        """Test that source weights are properly configured from environment."""
        # Check that source weights were loaded
        assert hasattr(retrieval, 'source_weights')
        assert retrieval.source_weights["MEMORY_WEIGHT_GRAPH"] == 0.4
        assert retrieval.source_weights["MEMORY_WEIGHT_CONVO"] == 0.3
        assert retrieval.source_weights["MEMORY_WEIGHT_SUMMARY"] == 0.2
        assert retrieval.source_weights["MEMORY_WEIGHT_SEMANTIC"] == 0.1
    
    def test_diversity_penalty_calculation(self, retrieval):
        """Test diversity penalty calculation for similar candidates."""
        similar_candidates = [
            Candidate(
                text="User lives in New York City",
                source="convo",
                score_hint=0.8,
                ts=int(time.time() * 1000),
                meta={}
            ),
            Candidate(
                text="User resides in New York City", 
                source="convo",
                score_hint=0.7,
                ts=int(time.time() * 1000),
                meta={}
            ),
            Candidate(
                text="User works at Google",
                source="convo", 
                score_hint=0.6,
                ts=int(time.time() * 1000),
                meta={}
            )
        ]
        
        # Test penalty for similar candidate (first two)
        penalty1 = retrieval._calculate_diversity_penalty(similar_candidates[0], similar_candidates)
        penalty2 = retrieval._calculate_diversity_penalty(similar_candidates[1], similar_candidates)
        penalty3 = retrieval._calculate_diversity_penalty(similar_candidates[2], similar_candidates)
        
        # First two should have higher penalty (they're similar)
        assert penalty1 > 0, "Similar candidate should have diversity penalty"
        assert penalty2 > 0, "Similar candidate should have diversity penalty"
        # Third should have lower penalty (different content)
        assert penalty3 < penalty1, "Different candidate should have lower penalty"
        
    def test_text_normalization_for_diversity(self, retrieval):
        """Test text normalization for diversity comparison."""
        text1 = "User lives in New York City!"
        text2 = "user lives in new york city"
        text3 = "User works at Google"
        
        norm1 = retrieval._normalize_for_diversity(text1)
        norm2 = retrieval._normalize_for_diversity(text2)
        norm3 = retrieval._normalize_for_diversity(text3)
        
        assert norm1 == norm2, "Texts with same content should normalize identically"
        assert norm1 != norm3, "Different texts should normalize differently"


class TestTokenBudgetAndBulletCapEnforced:
    """Test token budget enforcement and bullet capping."""
    
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
    def retrieval(self, mock_host):
        """Create a Retrieval instance for testing."""
        with patch.dict(os.environ, {
            'MEMORY_TOKEN_BUDGET': '200',
            'MEMORY_MAX_BULLETS': '2'
        }):
            retrieval = Retrieval(mock_host)
            return retrieval
    
    def test_token_budget_enforcement(self, retrieval):
        """Test that token budget is never exceeded."""
        # Create candidates that would exceed budget
        long_text = "This is a very long memory bullet that contains many words and should consume a significant portion of the token budget when included in the final output for testing purposes."
        
        candidates = [
            (1.0, Candidate(
                text=long_text,
                source="graph",
                score_hint=0.8,
                ts=int(time.time() * 1000),
                meta={"weight": 0.9}
            ), {}),
            (0.9, Candidate(
                text=long_text,
                source="convo", 
                score_hint=0.7,
                ts=int(time.time() * 1000),
                meta={"bm25_score": 0.7}
            ), {}),
            (0.8, Candidate(
                text=long_text,
                source="summary",
                score_hint=0.6,
                ts=int(time.time() * 1000),
                meta={}
            ), {})
        ]
        
        bullets, selected = retrieval._apply_token_budget_and_deduplication(
            candidates, max_bullets=5, query="test query"
        )
        
        # Should be limited by token budget, not max_bullets
        assert len(bullets) <= 2, "Should be limited by token budget"
        assert len(selected) <= 2, "Selected candidates should be limited"
        
        # Estimate tokens (rough heuristic: 4 chars = 1 token)
        total_chars = sum(len(bullet) for bullet in bullets)
        estimated_tokens = total_chars / 4
        
        assert estimated_tokens <= 200, f"Estimated tokens {estimated_tokens} should not exceed budget 200"
    
    def test_bullet_cap_enforcement(self, retrieval):
        """Test that bullet cap is never exceeded."""
        short_candidates = [
            (1.0, Candidate(
                text="Short bullet 1",
                source="graph",
                score_hint=0.8,
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.9, Candidate(
                text="Short bullet 2",
                source="convo",
                score_hint=0.7, 
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.8, Candidate(
                text="Short bullet 3",
                source="summary",
                score_hint=0.6,
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.7, Candidate(
                text="Short bullet 4",
                source="graph",
                score_hint=0.5,
                ts=int(time.time() * 1000),
                meta={}
            ), {})
        ]
        
        bullets, selected = retrieval._apply_token_budget_and_deduplication(
            short_candidates, max_bullets=5, query="test query"
        )
        
        # Should be limited by bullet cap (2)
        assert len(bullets) <= 2, f"Bullet count {len(bullets)} should not exceed cap 2"
        assert len(selected) <= 2, f"Selected count {len(selected)} should not exceed cap 2"


class TestCrossSourceDeduplication:
    """Test cross-source deduplication functionality."""
    
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
    def retrieval(self, mock_host):
        """Create a Retrieval instance for testing."""
        retrieval = Retrieval(mock_host)
        return retrieval
    
    def test_cross_source_deduplication(self, retrieval):
        """Test that duplicate content across sources is deduplicated."""
        candidates = [
            (1.0, Candidate(
                text="User lives in New York City",
                source="graph",
                score_hint=0.8,
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.9, Candidate(
                text="user lives in new york city",  # Same content, different case
                source="convo",
                score_hint=0.7,
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.8, Candidate(
                text="User works at Google",  # Different content
                source="summary",
                score_hint=0.6,
                ts=int(time.time() * 1000),
                meta={}
            ), {})
        ]
        
        bullets, selected = retrieval._apply_token_budget_and_deduplication(
            candidates, max_bullets=5, query="test query"
        )
        
        # Should have only 2 unique bullets (NYC and Google)
        assert len(bullets) == 2, f"Should have 2 unique bullets, got {len(bullets)}"
        
        # Check that deduplication worked
        bullet_texts = [bullet.lower() for bullet in bullets]
        nyc_bullets = [b for b in bullet_texts if "new york" in b]
        google_bullets = [b for b in bullet_texts if "google" in b]
        
        assert len(nyc_bullets) == 1, "Should have only one NYC bullet after deduplication"
        assert len(google_bullets) == 1, "Should have one Google bullet"
    
    def test_source_tags_preserved(self, retrieval):
        """Test that source tags are preserved in final bullets."""
        candidates = [
            (1.0, Candidate(
                text="User lives in New York",
                source="graph",
                score_hint=0.8,
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.9, Candidate(
                text="User works at Google",
                source="convo",
                score_hint=0.7,
                ts=int(time.time() * 1000),
                meta={}
            ), {}),
            (0.8, Candidate(
                text="User likes programming",
                source="summary",
                score_hint=0.6,
                ts=int(time.time() * 1000),
                meta={}
            ), {})
        ]
        
        bullets, selected = retrieval._apply_token_budget_and_deduplication(
            candidates, max_bullets=5, query="test query"
        )
        
        # All bullets should have source tags
        for bullet in bullets:
            assert "[graph]" in bullet or "[convo]" in bullet or "[summary]" in bullet, \
                f"Bullet should have source tag: {bullet}"


class TestGreetingIntentGatingSuppressesInjection:
    """Test strengthened greeting and intent gating."""
    
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
    def retrieval(self, mock_host):
        """Create a Retrieval instance for testing."""
        retrieval = Retrieval(mock_host)
        return retrieval
    
    def test_greeting_suppression(self, retrieval):
        """Test that pure greetings suppress memory injection."""
        with patch.object(retrieval, '_get_source_priority', return_value=['graph', 'convo', 'summary']):
            with patch.object(retrieval, '_allocate_budget', return_value={'graph': 2}):
                with patch.object(retrieval, '_graph_collect_candidates', return_value=[]):
                    
                    # Test various greetings
                    greetings = [
                        "hello",
                        "hi there", 
                        "hey",
                        "good morning",
                        "howdy",
                        "greetings",
                        "how are you",
                        "what's up",
                        "nice to meet you"
                    ]
                    
                    for greeting in greetings:
                        result = retrieval.retrieve(
                            query=greeting,
                            entities=[],
                            turn_id=1,
                            max_bullets=3
                        )
                        
                        assert result == [], f"Greeting '{greeting}' should suppress memory injection"
    
    def test_name_inquiry_allowed(self, retrieval):
        """Test that name inquiries during greetings are allowed."""
        with patch.object(retrieval, '_get_source_priority', return_value=['graph', 'convo', 'summary']):
            with patch.object(retrieval, '_allocate_budget', return_value={'graph': 2}):
                with patch.object(retrieval, '_graph_collect_candidates') as mock_collect:
                    # Mock returning a name-related candidate
                    mock_candidate = Candidate(
                        text="User's name is John",
                        source="graph",
                        score_hint=0.8,
                        ts=int(time.time() * 1000),
                        meta={"weight": 0.9, "pos": 2, "neg": 0, "edge_id": "name_edge"}
                    )
                    mock_collect.return_value = [mock_candidate]
                    
                    # Test name inquiries during greetings
                    name_queries = [
                        "hello what's your name",
                        "hi who are you", 
                        "good morning what is your name",
                        "hey what are you called"
                    ]
                    
                    for query in name_queries:
                        with patch.object(retrieval, '_apply_token_budget_and_deduplication') as mock_apply:
                            mock_apply.return_value = (["• [graph] User's name is John"], [mock_candidate])
                            
                            result = retrieval.retrieve(
                                query=query,
                                entities=["name"],
                                turn_id=1,
                                max_bullets=3
                            )
                            
                            assert len(result) > 0, f"Name inquiry '{query}' should allow memory injection"
    
    def test_regular_queries_unaffected(self, retrieval):
        """Test that regular queries are unaffected by greeting gating."""
        with patch.object(retrieval, '_get_source_priority', return_value=['graph', 'convo', 'summary']):
            with patch.object(retrieval, '_allocate_budget', return_value={'graph': 2}):
                with patch.object(retrieval, '_graph_collect_candidates') as mock_collect:
                    mock_candidate = Candidate(
                        text="User lives in New York",
                        source="graph", 
                        score_hint=0.8,
                        ts=int(time.time() * 1000),
                        meta={"weight": 0.9, "pos": 2, "neg": 0}
                    )
                    mock_collect.return_value = [mock_candidate]
                    
                    with patch.object(retrieval, '_apply_token_budget_and_deduplication') as mock_apply:
                        mock_apply.return_value = (["• [graph] User lives in New York"], [mock_candidate])
                        
                        result = retrieval.retrieve(
                            query="where do I live",
                            entities=["live"],
                            turn_id=1,
                            max_bullets=3
                        )
                        
                        assert len(result) > 0, "Regular query should work normally"


if __name__ == "__main__":
    pytest.main([__file__])
