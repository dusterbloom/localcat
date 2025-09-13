"""
CoreferenceResolver: Dedicated Coreference Resolution Service
=============================================================

Extracted from HotMemoryFacade - now focused solely on:
- Neural coreference resolution using FCoref
- Rule-based coreference resolution
- Pronoun resolution and entity linking
- Performance optimization for coreference processing
"""

import os
import time
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

try:
    from fastcoref import FCoref
    FASTCOREF_AVAILABLE = True
except Exception as e:
    FCoref = None
    FASTCOREF_AVAILABLE = False
    logger.warning(f"[CoreferenceResolver] fastcoref not available: {e}")


@dataclass
class CoreferenceResult:
    """Result of coreference resolution"""
    resolved_triples: List[Tuple[str, str, str]]
    resolution_stats: Dict[str, Any]
    processing_time_ms: float


class CoreferenceResolver:
    """
    Dedicated service for resolving coreferences in extracted triples.
    Handles both neural and rule-based approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize coreference resolver with configuration"""
        self.use_coref = config.get('use_coref', False) and FASTCOREF_AVAILABLE and FCoref is not None
        self.coref_max_entities = config.get('coref_max_entities', 24)
        self.device = config.get('coref_device', 'cpu')
        
        # Neural coreference model (lazy loaded)
        self._coref_model = None
        self._coref_cache = {}  # Cache resolved text
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
        # Performance guardrails
        self.max_triples_for_coref = 24
        self.max_entities_for_coref = 24
        
        logger.info(f"[CoreferenceResolver] Initialized with neural={'✓' if self.use_coref else '✗'}, max_entities={self.coref_max_entities}")
    
    def resolve_coreferences(self, triples: List[Tuple[str, str, str]], doc, text: str = "") -> CoreferenceResult:
        """
        Main entry point for coreference resolution
        """
        start = time.perf_counter()
        
        try:
            if not triples:
                return CoreferenceResult(triples, {'method': 'none'}, 0.0)
            
            # Skip coreference for too many triples (performance guard)
            if len(triples) > self.max_triples_for_coref:
                logger.debug(f"[CoreferenceResolver] Skipping coreference for {len(triples)} triples (too many)")
                return CoreferenceResult(triples, {'method': 'skipped', 'reason': 'too_many_triples'}, 0.0)
            
            # Try neural coreference first if enabled
            if self.use_coref and self._should_use_neural(doc, len(triples)):
                try:
                    result = self._apply_neural_coreference(triples, doc)
                    if result.resolved_triples != triples:
                        logger.debug(f"[CoreferenceResolver] Neural coreference resolved {len(result.resolved_triples)} triples")
                        return result
                except Exception as e:
                    logger.debug(f"[CoreferenceResolver] Neural coreference failed: {e}")
            
            # Fallback to rule-based coreference
            try:
                result = self._apply_rule_based_coreference(triples, doc, text)
                logger.debug(f"[CoreferenceResolver] Rule-based coreference resolved {len(result.resolved_triples)} triples")
                return result
            except Exception as e:
                logger.warning(f"[CoreferenceResolver] Rule-based coreference failed: {e}")
            
            # Return original triples if both methods fail
            processing_time = (time.perf_counter() - start) * 1000
            return CoreferenceResult(triples, {'method': 'failed'}, processing_time)
            
        except Exception as e:
            logger.error(f"[CoreferenceResolver] Coreference resolution failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return CoreferenceResult(triples, {'error': str(e)}, processing_time)
    
    def _should_use_neural(self, doc, triple_count: int) -> bool:
        """Determine if neural coreference should be used"""
        if not self.use_coref or not doc:
            return False
        
        # Performance checks
        if triple_count > self.max_triples_for_coref:
            return False
        
        # Check document size
        try:
            doc_entities = len(getattr(doc, 'ents', []))
            if doc_entities > self.coref_max_entities:
                return False
        except Exception:
            pass
        
        return True
    
    def _apply_neural_coreference(self, triples: List[Tuple[str, str, str]], doc) -> CoreferenceResult:
        """Apply neural coreference resolution using FCoref"""
        start = time.perf_counter()
        
        # Lazy load model
        if self._coref_model is None:
            self._coref_model = self._load_neural_model()
        
        if not self._coref_model:
            processing_time = (time.perf_counter() - start) * 1000
            return CoreferenceResult(triples, {'method': 'neural', 'status': 'model_load_failed'}, processing_time)
        
        try:
            # Check cache first
            cache_key = str(triples) + str(hash(str(doc)))
            if cache_key in self._coref_cache:
                cached_result = self._coref_cache[cache_key]
                processing_time = (time.perf_counter() - start) * 1000
                return CoreferenceResult(cached_result, {'method': 'neural', 'cached': True}, processing_time)
            
            # Apply neural coreference
            resolved_triples = self._coref_model.resolve_coreferences(triples, doc)
            
            # Cache result
            self._coref_cache[cache_key] = resolved_triples
            
            # Manage cache size
            if len(self._coref_cache) > 1000:
                oldest_keys = list(self._coref_cache.keys())[:200]
                for key in oldest_keys:
                    del self._coref_cache[key]
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'neural',
                'original_count': len(triples),
                'resolved_count': len(resolved_triples),
                'cache_size': len(self._coref_cache)
            }
            
            return CoreferenceResult(resolved_triples, stats, processing_time)
            
        except Exception as e:
            logger.debug(f"[CoreferenceResolver] Neural coreference processing failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return CoreferenceResult(triples, {'method': 'neural', 'error': str(e)}, processing_time)
    
    def _apply_rule_based_coreference(self, triples: List[Tuple[str, str, str]], doc, text: str = "") -> CoreferenceResult:
        """Apply rule-based coreference resolution"""
        start = time.perf_counter()
        
        try:
            resolved_triples = []
            
            # Common pronoun mappings
            pronoun_map = {
                'i': 'you',
                'me': 'you', 
                'my': 'your',
                'mine': 'your',
                'am': 'are',
                'was': 'were'
            }
            
            # Entity context from document
            doc_entities = {}
            try:
                for ent in getattr(doc, 'ents', []):
                    doc_entities[ent.text.lower()] = ent.text
            except Exception:
                pass
            
            # Apply rules to each triple
            for s, r, d in triples:
                # Subject resolution
                resolved_s = self._resolve_entity(s, pronoun_map, doc_entities, text)
                # Object resolution  
                resolved_d = self._resolve_entity(d, pronoun_map, doc_entities, text)
                
                resolved_triples.append((resolved_s, r, resolved_d))
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'rule_based',
                'original_count': len(triples),
                'resolved_count': len(resolved_triples),
                'rules_applied': sum(1 for orig, new in zip(triples, resolved_triples) if orig != new)
            }
            
            return CoreferenceResult(resolved_triples, stats, processing_time)
            
        except Exception as e:
            logger.debug(f"[CoreferenceResolver] Rule-based coreference failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return CoreferenceResult(triples, {'method': 'rule_based', 'error': str(e)}, processing_time)
    
    def _resolve_entity(self, entity: str, pronoun_map: Dict[str, str], doc_entities: Dict[str, str], text: str) -> str:
        """Resolve a single entity using rules"""
        if not entity:
            return entity
        
        entity_lower = entity.lower()
        
        # Direct pronoun mapping
        if entity_lower in pronoun_map:
            return pronoun_map[entity_lower]
        
        # Check document entities
        if entity_lower in doc_entities:
            return doc_entities[entity_lower]
        
        # Contextual resolution based on text
        if text and 'you' in text.lower():
            # First-person pronouns in user context should map to 'you'
            first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself', 'am', 'was'}
            if entity_lower in first_person_pronouns:
                return 'you'
        
        return entity
    
    def _load_neural_model(self):
        """Load neural coreference model"""
        try:
            model = FCoref(device=self.device)
            logger.info("[CoreferenceResolver] Neural coref model loaded")
            return model
        except Exception as e:
            logger.warning(f"[CoreferenceResolver] Failed to load neural coref model: {e}")
            return None
    
    def prewarm(self):
        """Prewarm neural model if enabled"""
        if self.use_coref and self._coref_model is None:
            logger.info("[CoreferenceResolver] Prewarming neural coreference model...")
            self._coref_model = self._load_neural_model()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get coreference performance metrics"""
        metrics = dict(self.metrics)
        metrics.update({
            'cache_size': len(self._coref_cache),
            'neural_enabled': self.use_coref,
            'model_loaded': self._coref_model is not None
        })
        return metrics
    
    def clear_cache(self):
        """Clear coreference cache"""
        self._coref_cache.clear()
        logger.debug("[CoreferenceResolver] Coreference cache cleared")


logger.info("🎯 CoreferenceResolver initialized - dedicated coreference service")
logger.info("📊 Features: Neural coreference, rule-based fallback, performance optimization")