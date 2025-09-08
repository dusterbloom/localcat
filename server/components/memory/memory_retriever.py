"""
Memory Retriever Module

Handles all query processing and retrieval operations for the HotMem system.
Provides multiple retrieval strategies and result ranking.

Author: SOLID Refactoring
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict, deque
import math
import heapq

from loguru import logger
from components.memory.memory_store import MemoryStore

class MemoryRetriever:
    """
    Handles all retrieval operations for the HotMem system.
    
    Responsibilities:
    - Query processing and tokenization
    - Multiple retrieval strategies (exact, semantic, fuzzy)
    - Result ranking and scoring
    - Performance optimization and caching
    """
    
    def __init__(self, store: MemoryStore, config: Optional[Dict[str, Any]] = None):
        self.store = store
        self.config = config or {}
        
        # Retrieval configuration
        self.max_results = self.config.get('max_results', 10)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.semantic_threshold = self.config.get('semantic_threshold', 0.5)
        
        # Performance tracking
        self.query_stats = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Simple cache for frequent queries
        self.query_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
    def retrieve(self, query: str, session_id: str = None, 
                strategies: List[str] = None) -> List[Tuple[float, str, str, str]]:
        """
        Retrieve relevant facts based on the query.
        
        Args:
            query: Query string
            session_id: Optional session ID for context
            strategies: List of retrieval strategies to use
            
        Returns:
            List of (score, subject, relation, object) tuples
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(query, strategies)
        if cache_key in self.query_cache:
            cached_result, cache_time = self.query_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                self.cache_hits += 1
                self.query_stats['cache_hits'] += 1
                return cached_result
        
        self.cache_misses += 1
        
        # Process query
        query_tokens = self._tokenize_query(query)
        
        # Apply retrieval strategies
        results = []
        strategies = strategies or ['exact', 'semantic', 'fuzzy']
        
        for strategy in strategies:
            strategy_results = self._apply_strategy(strategy, query_tokens, query)
            results.extend(strategy_results)
        
        # Rank and filter results
        ranked_results = self._rank_results(results, query_tokens)
        
        # Cache result
        self.query_cache[cache_key] = (ranked_results, time.time())
        
        # Update stats
        query_time = int((time.time() - start_time) * 1000)
        self.query_stats['query_time_ms'] += query_time
        self.query_stats['queries_processed'] += 1
        
        logger.debug(f"Retrieved {len(ranked_results)} results in {query_time}ms")
        
        return ranked_results[:self.max_results]
    
    def _tokenize_query(self, query: str) -> Set[str]:
        """Tokenize and normalize query."""
        tokens = set()
        
        # Simple tokenization
        for word in query.lower().split():
            word = word.strip('.,!?;:()[]{}"\'')
            if word:
                tokens.add(word)
        
        # Add query synonym expansion
        if any(w in tokens for w in {'drive', 'drives', 'driving'}):
            tokens.add('has')
        if any(w in tokens for w in {'teach', 'teaches', 'teaching'}):
            tokens.add('teach_at')
        if 'work' in tokens or 'works' in tokens:
            tokens.add('works_at')
        
        return tokens
    
    def _apply_strategy(self, strategy: str, query_tokens: Set[str], 
                       original_query: str) -> List[Tuple[float, str, str, str]]:
        """Apply a specific retrieval strategy."""
        if strategy == 'exact':
            return self._exact_match_retrieval(query_tokens)
        elif strategy == 'semantic':
            return self._semantic_retrieval(query_tokens, original_query)
        elif strategy == 'fuzzy':
            return self._fuzzy_match_retrieval(query_tokens)
        else:
            logger.warning(f"Unknown retrieval strategy: {strategy}")
            return []
    
    def _exact_match_retrieval(self, query_tokens: Set[str]) -> List[Tuple[float, str, str, str]]:
        """Exact word matching retrieval."""
        results = []
        
        try:
            # Get all edges from store
            edges = self.store.get_all_edges()
            
            for subject, relation, object_, confidence, timestamp in edges:
                score = self._calculate_exact_match_score(query_tokens, subject, relation, object_)
                if score >= self.min_confidence:
                    results.append((score, subject, relation, object_))
        
        except Exception as e:
            logger.error(f"Exact match retrieval failed: {e}")
        
        return results
    
    def _semantic_retrieval(self, query_tokens: Set[str], 
                          original_query: str) -> List[Tuple[float, str, str, str]]:
        """Semantic similarity retrieval."""
        results = []
        
        try:
            # This would integrate with semantic search capabilities
            # For now, use a simple word overlap approach
            edges = self.store.get_all_edges()
            
            for subject, relation, object_, confidence, timestamp in edges:
                semantic_score = self._calculate_semantic_score(query_tokens, subject, relation, object_)
                if semantic_score >= self.semantic_threshold:
                    results.append((semantic_score, subject, relation, object_))
        
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
        
        return results
    
    def _fuzzy_match_retrieval(self, query_tokens: Set[str]) -> List[Tuple[float, str, str, str]]:
        """Fuzzy matching retrieval."""
        results = []
        
        try:
            edges = self.store.get_all_edges()
            
            for subject, relation, object_, confidence, timestamp in edges:
                fuzzy_score = self._calculate_fuzzy_score(query_tokens, subject, relation, object_)
                if fuzzy_score >= self.min_confidence:
                    results.append((fuzzy_score, subject, relation, object_))
        
        except Exception as e:
            logger.error(f"Fuzzy match retrieval failed: {e}")
        
        return results
    
    def _calculate_exact_match_score(self, query_tokens: Set[str], 
                                   subject: str, relation: str, object_: str) -> float:
        """Calculate exact match score."""
        subject_tokens = set(subject.lower().split())
        relation_tokens = set(relation.lower().split())
        object_tokens = set(object_.lower().split())
        
        all_tokens = subject_tokens | relation_tokens | object_tokens
        intersection = query_tokens & all_tokens
        
        if not intersection:
            return 0.0
        
        # Jaccard similarity
        return len(intersection) / len(query_tokens | all_tokens)
    
    def _calculate_semantic_score(self, query_tokens: Set[str],
                                subject: str, relation: str, object_: str) -> float:
        """Calculate semantic similarity score."""
        # This would integrate with proper semantic similarity
        # For now, use a simple approach with relation synonyms
        relation_lower = relation.lower()
        
        # Boost score for semantically similar relations
        relation_synonyms = {
            'work': ['works_at', 'employed_by'],
            'teach': ['teach_at', 'educates'],
            'drive': ['has', 'owns'],
            'live': ['lives_in', 'resides_in']
        }
        
        score = self._calculate_exact_match_score(query_tokens, subject, relation, object_)
        
        # Check for relation synonyms
        for base_word, synonyms in relation_synonyms.items():
            if base_word in query_tokens and relation_lower in synonyms:
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _calculate_fuzzy_score(self, query_tokens: Set[str],
                             subject: str, relation: str, object_: str) -> float:
        """Calculate fuzzy match score."""
        # Simple fuzzy matching using substring containment
        subject_lower = subject.lower()
        relation_lower = relation.lower()
        object_lower = object_.lower()
        
        score = 0.0
        for token in query_tokens:
            if (token in subject_lower or 
                token in relation_lower or 
                token in object_lower):
                score += 0.3
        
        return min(score, 1.0)
    
    def _rank_results(self, results: List[Tuple[float, str, str, str]], 
                     query_tokens: Set[str]) -> List[Tuple[float, str, str, str]]:
        """Rank and deduplicate results."""
        # Remove duplicates
        seen = set()
        unique_results = []
        
        for score, subject, relation, object_ in results:
            key = (subject.lower(), relation.lower(), object_.lower())
            if key not in seen:
                seen.add(key)
                unique_results.append((score, subject, relation, object_))
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x[0], reverse=True)
        
        return unique_results
    
    def _get_cache_key(self, query: str, strategies: List[str]) -> str:
        """Generate cache key for query."""
        strategies_str = ','.join(sorted(strategies or []))
        return f"{query.lower()}:{strategies_str}"
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = dict(self.query_stats)
        stats['cache_hits'] = self.cache_hits
        stats['cache_misses'] = self.cache_misses
        stats['cache_size'] = len(self.query_cache)
        
        if self.cache_hits + self.cache_misses > 0:
            stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses)
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Query cache cleared")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies."""
        return ['exact', 'semantic', 'fuzzy']
    
    def explain_retrieval(self, query: str, results: List[Tuple[float, str, str, str]]) -> Dict[str, Any]:
        """
        Provide explanation for retrieval results.
        
        Args:
            query: Original query
            results: Retrieved results
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'query': query,
            'query_tokens': list(self._tokenize_query(query)),
            'results_count': len(results),
            'strategies_used': self.get_available_strategies(),
            'score_distribution': [r[0] for r in results],
            'top_results': results[:3]
        }
        
        return explanation