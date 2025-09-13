"""
EntityResolver: Advanced Entity Resolution and Deduplication Service
=====================================================================

SOTA entity resolution using dedupe library for intelligent matching
and deduplication of similar entities in knowledge graphs.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

try:
    import dedupe
    DEDUPE_AVAILABLE = True
except Exception as e:
    DEDUPE_AVAILABLE = False
    logger.warning(f"[EntityResolver] dedupe not available: {e}")

try:
    import rapidfuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception as e:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning(f"[EntityResolver] rapidfuzz not available: {e}")


@dataclass
class EntityResolutionResult:
    """Result of entity resolution"""
    resolved_entities: Dict[str, str]  # original -> resolved
    resolution_stats: Dict[str, Any]
    processing_time_ms: float


class EntityResolver:
    """
    Advanced entity resolution using dedupe library.
    Handles fuzzy matching, canonicalization, and deduplication.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize entity resolver with configuration"""
        force_rapidfuzz = config.get('force_rapidfuzz', False)
        self.enabled = (config.get('entity_resolution_enabled', False) and 
                       (DEDUPE_AVAILABLE or (force_rapidfuzz and RAPIDFUZZ_AVAILABLE)))
        self.threshold = config.get('entity_resolution_threshold', 0.85)
        self.sample_size = config.get('entity_resolution_sample_size', 1000)
        self.use_rapidfuzz = config.get('use_rapidfuzz_fallback', True) and RAPIDFUZZ_AVAILABLE
        self.force_rapidfuzz = force_rapidfuzz
        
        # Performance settings
        self.max_entities_for_dedupe = 1000
        self.min_entity_length = 2
        
        # Cache for resolved entities
        self._resolution_cache = {}
        
        # dedupe objects (lazy loaded)
        self._deduper = None
        self._trained = False
        
        logger.info(f"[EntityResolver] Initialized with enabled={'✓' if self.enabled else '✗'}, threshold={self.threshold}")
    
    def resolve_entities(self, entities: List[str], text: str = "") -> EntityResolutionResult:
        """
        Main entry point for entity resolution
        """
        start = time.perf_counter()
        
        try:
            if not entities or not self.enabled:
                return EntityResolutionResult(
                    {e: e for e in entities}, 
                    {'method': 'none'}, 
                    0.0
                )
            
            # Performance guard
            if len(entities) > self.max_entities_for_dedupe:
                logger.debug(f"[EntityResolver] Too many entities ({len(entities)}), using rapidfuzz fallback")
                return self._resolve_with_rapidfuzz(entities, text)
            
            # Check cache first
            cache_key = str(sorted(entities))
            if cache_key in self._resolution_cache:
                cached_result = self._resolution_cache[cache_key]
                processing_time = (time.perf_counter() - start) * 1000
                return EntityResolutionResult(
                    cached_result,
                    {'method': 'cached'},
                    processing_time
                )
            
            # Filter entities
            filtered_entities = [e for e in entities if len(e.strip()) >= self.min_entity_length]
            
            if len(filtered_entities) < 2:
                return EntityResolutionResult(
                    {e: e for e in entities},
                    {'method': 'too_few_entities'},
                    (time.perf_counter() - start) * 1000
                )
            
            # Use dedupe if available and not forced to use rapidfuzz
            if DEDUPE_AVAILABLE and not self.force_rapidfuzz:
                try:
                    result = self._resolve_with_dedupe(filtered_entities, text)
                    
                    # Cache result
                    self._resolution_cache[cache_key] = result.resolved_entities
                    
                    # Manage cache size
                    if len(self._resolution_cache) > 500:
                        oldest_keys = list(self._resolution_cache.keys())[:100]
                        for key in oldest_keys:
                            del self._resolution_cache[key]
                    
                    return result
                    
                except Exception as e:
                    logger.debug(f"[EntityResolver] dedupe failed: {e}")
                    # Fallback to rapidfuzz
                    return self._resolve_with_rapidfuzz(entities, text)
            else:
                # Use rapidfuzz directly
                return self._resolve_with_rapidfuzz(entities, text)
                
        except Exception as e:
            logger.error(f"[EntityResolver] Entity resolution failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return EntityResolutionResult(
                {e: e for e in entities},
                {'error': str(e)},
                processing_time
            )
    
    def _resolve_with_dedupe(self, entities: List[str], text: str) -> EntityResolutionResult:
        """Resolve entities using dedupe library"""
        start = time.perf_counter()
        
        # Lazy load deduper
        if self._deduper is None:
            self._deduper = self._create_deduper()
        
        if not self._deduper:
            # Fallback to rapidfuzz
            return self._resolve_with_rapidfuzz(entities, text)
        
        try:
            # Prepare data for dedupe
            data = {i: {'name': entity} for i, entity in enumerate(entities)}
            
            # Train if not trained
            if not self._trained:
                self._train_deduper(data)
            
            # Cluster entities
            clustered = self._deduper.partition(data, threshold=self.threshold)
            
            # Build resolution map
            resolved_map = {}
            for cluster_id, entity_ids in clustered:
                if not entity_ids:
                    continue
                
                # Use most frequent entity as canonical
                entity_texts = [entities[eid] for eid in entity_ids]
                canonical_entity = max(set(entity_texts), key=entity_texts.count)
                
                for eid in entity_ids:
                    resolved_map[entities[eid]] = canonical_entity
            
            # Handle entities not in clusters
            for entity in entities:
                if entity not in resolved_map:
                    resolved_map[entity] = entity
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'dedupe',
                'original_count': len(entities),
                'resolved_count': len(set(resolved_map.values())),
                'clusters': len([c for c in clustered if c[1]]),
                'trained': self._trained
            }
            
            return EntityResolutionResult(resolved_map, stats, processing_time)
            
        except Exception as e:
            logger.debug(f"[EntityResolver] dedupe processing failed: {e}")
            # Fallback to rapidfuzz
            return self._resolve_with_rapidfuzz(entities, text)
    
    def _resolve_with_rapidfuzz(self, entities: List[str], text: str) -> EntityResolutionResult:
        """Resolve entities using rapidfuzz as fallback"""
        start = time.perf_counter()
        
        if not self.use_rapidfuzz or not RAPIDFUZZ_AVAILABLE:
            # Simple identity mapping
            processing_time = (time.perf_counter() - start) * 1000
            return EntityResolutionResult(
                {e: e for e in entities},
                {'method': 'identity'},
                processing_time
            )
        
        try:
            resolved_map = {}
            processed = set()
            
            for i, entity1 in enumerate(entities):
                if entity1 in processed:
                    continue
                
                # Find similar entities
                similar_entities = [entity1]
                for j, entity2 in enumerate(entities):
                    if i != j and entity2 not in processed:
                        similarity = rapidfuzz.fuzz.token_set_ratio(entity1.lower(), entity2.lower()) / 100.0
                        if similarity >= self.threshold:
                            similar_entities.append(entity2)
                            processed.add(entity2)
                
                # Use most frequent as canonical
                canonical = max(set(similar_entities), key=similar_entities.count)
                for entity in similar_entities:
                    resolved_map[entity] = canonical
                
                processed.add(entity1)
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'rapidfuzz',
                'original_count': len(entities),
                'resolved_count': len(set(resolved_map.values())),
                'threshold': self.threshold
            }
            
            return EntityResolutionResult(resolved_map, stats, processing_time)
            
        except Exception as e:
            logger.debug(f"[EntityResolver] rapidfuzz failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return EntityResolutionResult(
                {e: e for e in entities},
                {'method': 'identity', 'error': str(e)},
                processing_time
            )
    
    def _create_deduper(self):
        """Create dedupe object"""
        try:
            # Define fields for comparison
            fields = [
                {
                    'field': 'name',
                    'type': 'String',
                    'has missing': False
                }
            ]
            
            deduper = dedupe.Dedupe(fields)
            logger.info("[EntityResolver] Created dedupe object")
            return deduper
            
        except Exception as e:
            logger.error(f"[EntityResolver] Failed to create dedupe object: {e}")
            return None
    
    def _train_deduper(self, data: Dict[int, Dict[str, str]]):
        """Train dedupe model"""
        try:
            # Prepare training data
            if len(data) > self.sample_size:
                # Sample data for training
                import random
                sample_keys = random.sample(list(data.keys()), self.sample_size)
                sample_data = {k: data[k] for k in sample_keys}
            else:
                sample_data = data
            
            # Train deduper
            self._deduper.prepare_training(sample_data)
            self._dedupe.train()
            
            self._trained = True
            logger.info(f"[EntityResolver] Trained dedupe model on {len(sample_data)} samples")
            
        except Exception as e:
            logger.error(f"[EntityResolver] Failed to train dedupe: {e}")
            self._trained = False
    
    def _dedupe_train(self, data: Dict[int, Dict[str, str]]):
        """Simple training without active learning"""
        try:
            # Use existing training data or create sample
            if len(data) >= 10:
                # Create some training examples
                training_pairs = []
                
                # Create some positive examples (similar entities)
                entities = list(data.values())
                for i in range(min(50, len(entities) - 1)):
                    if i + 1 < len(entities):
                        training_pairs.append((entities[i], entities[i + 1], 1))
                
                # Create some negative examples (dissimilar entities)
                for i in range(0, min(25, len(entities) - 5), 5):
                    if i + 5 < len(entities):
                        training_pairs.append((entities[i], entities[i + 5], 0))
                
                if training_pairs:
                    self._deduper.mark_pairs(training_pairs)
            
            # Train with existing data
            self._deduper.train(index_predicates=False)
            self._trained = True
            
        except Exception as e:
            logger.error(f"[EntityResolver] Training failed: {e}")
            self._trained = False
    
    def clear_cache(self):
        """Clear resolution cache"""
        self._resolution_cache.clear()
        logger.debug("[EntityResolver] Resolution cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        return {
            'enabled': self.enabled,
            'dedupe_available': DEDUPE_AVAILABLE,
            'rapidfuzz_available': RAPIDFUZZ_AVAILABLE,
            'trained': self._trained,
            'cache_size': len(self._resolution_cache),
            'threshold': self.threshold
        }


logger.info("🎯 EntityResolver initialized - advanced entity resolution with dedupe")
logger.info("📊 Features: Fuzzy matching, canonicalization, intelligent deduplication")