"""
Unified Extraction Orchestrator

Coordinates multiple extraction strategies and provides unified interface.
Handles strategy selection, result aggregation, and performance optimization.

Author: SOLID Refactoring
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import threading
from loguru import logger

from components.memory.memory_interfaces import IMemoryExtractor, ExtractionResult
from components.extraction.extraction_strategies import (
    ExtractionStrategyBase, create_strategy, get_available_strategies,
    EXTRACTION_STRATEGIES
)

class UnifiedExtractionOrchestrator(IMemoryExtractor):
    """
    Unified orchestrator for multiple extraction strategies.
    
    Responsibilities:
    - Strategy management and coordination
    - Result aggregation and deduplication
    - Performance optimization and caching
    - A/B testing and strategy selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Strategy configuration
        self.enabled_strategies = self.config.get('enabled_strategies', 
                                              ['hotmem', 'ud', 'hybrid', 'lightweight'])
        self.strategy_configs = self.config.get('strategy_configs', {})
        self.fallback_strategies = self.config.get('fallback_strategies', ['pattern'])
        
        # Performance configuration
        self.max_extraction_time_ms = self.config.get('max_extraction_time_ms', 5000)
        self.max_total_triples = self.config.get('max_total_triples', 100)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # A/B testing configuration
        self.ab_testing = self.config.get('ab_testing', False)
        self.ab_test_groups = self.config.get('ab_test_groups', {})
        
        # Initialize strategies
        self.strategies = {}
        self.strategy_stats = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0,
            'total_triples': 0,
            'errors': 0
        })
        
        # Caching
        self.extraction_cache = {}
        self.cache_lock = threading.RLock()
        
        # Load strategies
        self._load_strategies()
        
        logger.info(f"UnifiedExtractionOrchestrator initialized with {len(self.strategies)} strategies")
    
    def _load_strategies(self) -> None:
        """Load all enabled extraction strategies."""
        for strategy_name in self.enabled_strategies:
            strategy_config = self.strategy_configs.get(strategy_name, {})
            strategy_config['enabled'] = True
            
            try:
                strategy = create_strategy(strategy_name, strategy_config)
                if strategy and strategy.is_available():
                    self.strategies[strategy_name] = strategy
                    logger.info(f"Loaded strategy: {strategy_name}")
                else:
                    logger.warning(f"Strategy {strategy_name} not available")
            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_name}: {e}")
        
        # Load fallback strategies
        for strategy_name in self.fallback_strategies:
            if strategy_name not in self.strategies:
                try:
                    strategy_config = self.strategy_configs.get(strategy_name, {})
                    strategy_config['enabled'] = True
                    strategy = create_strategy(strategy_name, strategy_config)
                    if strategy and strategy.is_available():
                        self.strategies[strategy_name] = strategy
                        logger.info(f"Loaded fallback strategy: {strategy_name}")
                except Exception as e:
                    logger.error(f"Failed to load fallback strategy {strategy_name}: {e}")
    
    def extract_triples(self, text: str, lang: str = "en") -> ExtractionResult:
        """
        Extract entities and triples using multiple strategies.
        
        Args:
            text: Input text to process
            lang: Language code
            
        Returns:
            ExtractionResult containing entities and triples
        """
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching:
            cached_result = self._get_from_cache(text, lang)
            if cached_result:
                return cached_result
        
        # Select strategies for this extraction
        strategies_to_use = self._select_strategies(text, lang)
        
        # Extract using selected strategies
        all_triples = []
        all_entities = set()
        strategy_results = {}
        
        extraction_deadline = start_time + (self.max_extraction_time_ms / 1000)
        
        for strategy_name, strategy in strategies_to_use.items():
            if time.time() > extraction_deadline:
                logger.warning(f"Extraction timeout reached, skipping strategy: {strategy_name}")
                break
            
            try:
                strategy_start = time.time()
                triples = strategy.extract(text, lang)
                strategy_time = int((time.time() - strategy_start) * 1000)
                
                # Update statistics
                self.strategy_stats[strategy_name]['calls'] += 1
                self.strategy_stats[strategy_name]['total_time'] += strategy_time
                self.strategy_stats[strategy_name]['total_triples'] += len(triples)
                
                # Store results
                strategy_results[strategy_name] = {
                    'triples': triples,
                    'time_ms': strategy_time,
                    'success': True
                }
                
                all_triples.extend(triples)
                
                # Extract entities from triples
                for subject, relation, object_ in triples:
                    all_entities.add(subject)
                    all_entities.add(object_)
                
                logger.debug(f"Strategy {strategy_name} extracted {len(triples)} triples in {strategy_time}ms")
                
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {e}")
                self.strategy_stats[strategy_name]['errors'] += 1
                strategy_results[strategy_name] = {
                    'triples': [],
                    'time_ms': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Process and filter results
        final_triples = self._process_results(all_triples)
        final_entities = list(all_entities)
        
        # Limit total triples
        if len(final_triples) > self.max_total_triples:
            final_triples = final_triples[:self.max_total_triples]
        
        # Create result
        extraction_time = int((time.time() - start_time) * 1000)
        result = ExtractionResult(
            entities=final_entities,
            triples=final_triples,
            extraction_time_ms=extraction_time,
            metadata={
                'strategies_used': list(strategies_to_use.keys()),
                'strategy_results': strategy_results,
                'total_strategies': len(strategies_to_use),
                'ab_test_group': self._get_ab_test_group(text)
            }
        )
        
        # Cache result
        if self.enable_caching:
            self._add_to_cache(text, lang, result)
        
        return result
    
    def _select_strategies(self, text: str, lang: str) -> Dict[str, ExtractionStrategyBase]:
        """Select strategies to use for extraction."""
        strategies = {}
        
        # A/B testing mode
        if self.ab_testing:
            test_group = self._get_ab_test_group(text)
            group_strategies = self.ab_test_groups.get(test_group, self.enabled_strategies)
            
            for strategy_name in group_strategies:
                if strategy_name in self.strategies:
                    strategies[strategy_name] = self.strategies[strategy_name]
        
        # Normal mode - use all enabled strategies
        else:
            strategies = self.strategies.copy()
        
        # Prioritize strategies by configuration
        prioritized = {}
        for strategy_name in self.enabled_strategies:
            if strategy_name in strategies:
                prioritized[strategy_name] = strategies[strategy_name]
        
        return prioritized
    
    def _process_results(self, all_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Process and deduplicate extraction results."""
        if not all_triples:
            return []
        
        # Simple deduplication
        seen = set()
        unique_triples = []
        
        for triple in all_triples:
            # Normalize triple for comparison
            normalized = (triple[0].lower(), triple[1].lower(), triple[2].lower())
            
            if normalized not in seen:
                seen.add(normalized)
                unique_triples.append(triple)
        
        # Sort by some quality heuristic (could be enhanced)
        unique_triples.sort(key=lambda t: (
            len(t[0]) + len(t[1]) + len(t[2]),  # Prefer longer, more specific triples
            t[0] != 'you',  # Prefer non-"you" subjects
        ), reverse=True)
        
        return unique_triples
    
    def _get_from_cache(self, text: str, lang: str) -> Optional[ExtractionResult]:
        """Get extraction result from cache."""
        with self.cache_lock:
            cache_key = f"{text}:{lang}"
            if cache_key in self.extraction_cache:
                result, timestamp = self.extraction_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return result
                else:
                    # Remove expired cache entry
                    del self.extraction_cache[cache_key]
        return None
    
    def _add_to_cache(self, text: str, lang: str, result: ExtractionResult) -> None:
        """Add extraction result to cache."""
        with self.cache_lock:
            cache_key = f"{text}:{lang}"
            
            # Simple cache eviction - remove oldest entries if cache is too large
            if len(self.extraction_cache) > 1000:
                oldest_key = min(self.extraction_cache.keys(), 
                               key=lambda k: self.extraction_cache[k][1])
                del self.extraction_cache[oldest_key]
            
            self.extraction_cache[cache_key] = (result, time.time())
    
    def _get_ab_test_group(self, text: str) -> str:
        """Get A/B test group for the given text."""
        if not self.ab_testing:
            return 'control'
        
        # Simple hash-based group assignment
        hash_value = hash(text) % 100
        
        if hash_value < 50:
            return 'control'
        elif hash_value < 75:
            return 'test_a'
        else:
            return 'test_b'
    
    def get_extraction_strategies(self) -> List[str]:
        """Get list of available extraction strategies."""
        return list(self.strategies.keys())
    
    def is_strategy_available(self, strategy: str) -> bool:
        """Check if a specific extraction strategy is available."""
        return strategy in self.strategies and self.strategies[strategy].is_available()
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        stats = {
            'strategies': {},
            'cache': {
                'enabled': self.enable_caching,
                'size': len(self.extraction_cache),
                'ttl_seconds': self.cache_ttl
            },
            'ab_testing': {
                'enabled': self.ab_testing,
                'groups': self.ab_test_groups
            }
        }
        
        # Strategy statistics
        for strategy_name, strategy in self.strategies.items():
            stats['strategies'][strategy_name] = {
                'config': strategy.get_strategy_config(),
                'stats': dict(self.strategy_stats[strategy_name]),
                'available': strategy.is_available()
            }
        
        return stats
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            if strategy_name not in self.enabled_strategies:
                self.enabled_strategies.append(strategy_name)
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            if strategy_name in self.enabled_strategies:
                self.enabled_strategies.remove(strategy_name)
            return True
        return False
    
    def reload_strategy(self, strategy_name: str) -> bool:
        """Reload a strategy."""
        if strategy_name in self.strategies:
            try:
                strategy_config = self.strategy_configs.get(strategy_name, {})
                new_strategy = create_strategy(strategy_name, strategy_config)
                if new_strategy and new_strategy.is_available():
                    self.strategies[strategy_name] = new_strategy
                    logger.info(f"Reloaded strategy: {strategy_name}")
                    return True
            except Exception as e:
                logger.error(f"Failed to reload strategy {strategy_name}: {e}")
        return False
    
    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        with self.cache_lock:
            self.extraction_cache.clear()
        logger.info("Extraction cache cleared")
    
    def configure_ab_test(self, test_config: Dict[str, Any]) -> None:
        """Configure A/B testing."""
        self.ab_testing = test_config.get('enabled', self.ab_testing)
        self.ab_test_groups = test_config.get('groups', self.ab_test_groups)
        logger.info(f"A/B testing configured: enabled={self.ab_testing}, groups={len(self.ab_test_groups)}")
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies."""
        performance = {}
        
        for strategy_name, stats in self.strategy_stats.items():
            if stats['calls'] > 0:
                performance[strategy_name] = {
                    'calls': stats['calls'],
                    'avg_time_ms': stats['total_time'] / stats['calls'],
                    'avg_triples': stats['total_triples'] / stats['calls'],
                    'error_rate': stats['errors'] / stats['calls'],
                    'efficiency': stats['total_triples'] / max(stats['total_time'], 1)  # triples per millisecond
                }
        
        return performance