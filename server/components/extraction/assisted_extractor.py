"""
AssistedExtractor: Dedicated LLM-Assisted Extraction Service
==============================================================

Extracted from HotMemoryFacade - now focused solely on:
- LLM-assisted relation classification
- JSON-based structured extraction
- Classifier-based extraction
- Performance optimization and caching
"""

import os
import time
import json
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

try:
    import requests
except Exception:
    requests = None


@dataclass
class AssistedExtractionResult:
    """Result of assisted extraction"""
    triples: List[Tuple[str, str, str]]
    extraction_stats: Dict[str, Any]
    processing_time_ms: float
    method_used: str


class AssistedExtractor:
    """
    Dedicated service for LLM-assisted relation extraction.
    Handles multiple extraction strategies with performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize assisted extractor with configuration"""
        self.enabled = config.get('assisted_enabled', False)
        self.model_config = config.get('assisted_model')
        
        # Performance settings
        self.timeout_ms = getattr(self.model_config, 'timeout_ms', 120)
        self.max_triples = getattr(self.model_config, 'max_triples', 3)
        
        # Performance tracking
        self.metrics = defaultdict(list)
        self._assisted_calls = 0
        self._assisted_success = 0
        
        # Cache for classifier results
        self._classifier_cache = {}
        self._cache_max_size = config.get('cache_size', 1000)
        
        logger.info(f"[AssistedExtractor] Initialized with model={self.model_config.name if self.model_config else 'none'}, enabled={'✓' if self.enabled else '✗'}")
    
    def should_assist(self, text: str, triples: List[Tuple[str, str, str]], doc) -> bool:
        """Determine if assisted extraction should be triggered"""
        if not self.enabled or not triples:
            return False
        
        # Trigger conditions for assistance
        trigger_conditions = [
            # Very few triples extracted
            len(triples) <= 2,
            # High uncertainty in relations
            any(r in {'is', 'are', 'was', 'were', 'has', 'have', 'do', 'does'} for _, r, _ in triples),
            # Complex sentences
            len(text.split()) > 20,
            # Question marks in text (might indicate complex relationships)
            '?' in text,
        ]
        
        return any(trigger_conditions)
    
    def extract_assisted(self, text: str, entities: List[str], base_triples: List[Tuple[str, str, str]], 
                        session_id: Optional[str] = None) -> AssistedExtractionResult:
        """
        Main entry point for assisted extraction
        """
        start = time.perf_counter()
        self._assisted_calls += 1
        
        try:
            # Choose extraction method based on availability
            if self._should_use_classifier():
                result = self._extract_with_classifier(text, entities, base_triples)
            elif self._should_use_json():
                result = self._extract_with_json(text, entities, base_triples)
            else:
                result = self._extract_with_fallback(text, entities, base_triples)
            
            # Track success
            if result.triples:
                self._assisted_success += 1
            
            processing_time = (time.perf_counter() - start) * 1000
            result.processing_time_ms = processing_time
            
            # Track metrics
            self.metrics['assisted_ms'].append(processing_time)
            self.metrics['assisted_triples'].append(len(result.triples))
            
            logger.info(f"[AssistedExtractor] {result.method_used} (ms={processing_time:.0f}, triples={len(result.triples)})")
            
            return result
            
        except Exception as e:
            logger.debug(f"[AssistedExtractor] Error: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return AssistedExtractionResult([], {'error': str(e)}, processing_time, 'failed')
    
    def _should_use_classifier(self) -> bool:
        """Check if classifier-based extraction should be used"""
        # Conditions for classifier approach
        return (
            self.enabled and
            self.model_config and
            requests is not None and
            len(self._classifier_cache) < self._cache_max_size
        )
    
    def _should_use_json(self) -> bool:
        """Check if JSON-based extraction should be used"""
        # Conditions for JSON approach
        return (
            self.enabled and
            self.model_config and
            requests is not None and
            self.max_triples > 1
        )
    
    def _extract_with_classifier(self, text: str, entities: List[str], base_triples: List[Tuple[str, str, str]]) -> AssistedExtractionResult:
        """Extract using classifier-based approach"""
        start = time.perf_counter()
        
        # Generate all possible entity pairs for classification
        entity_pairs = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                entity_pairs.append((e1, e2))
        
        # Limit to prevent too many calls
        entity_pairs = entity_pairs[:self.max_triples * 2]
        
        extracted_triples = []
        classification_results = []
        
        for s, d in entity_pairs:
            # Check cache first
            cache_key = f"{s}|{d}|{hash(text[:100])}"
            if cache_key in self._classifier_cache:
                classification_results.append(self._classifier_cache[cache_key])
                continue
            
            # Classify relation
            try:
                relation = self._classify_relation(s, d, text)
                classification_results.append((s, relation, d))
                
                # Cache result
                self._classifier_cache[cache_key] = (s, relation, d)
                
                # Manage cache size
                if len(self._classifier_cache) > self._cache_max_size:
                    oldest_keys = list(self._classifier_cache.keys())[:self._cache_max_size // 4]
                    for key in oldest_keys:
                        del self._classifier_cache[key]
                        
            except Exception as e:
                logger.debug(f"[AssistedExtractor] Classification failed for ({s}, {d}): {e}")
        
        # Filter high-confidence results
        for s, r, d in classification_results:
            if r and r not in {'unknown', 'none', ''}:
                extracted_triples.append((s, r, d))
        
        processing_time = (time.perf_counter() - start) * 1000
        stats = {
            'method': 'classifier',
            'pairs_classified': len(entity_pairs),
            'triples_extracted': len(extracted_triples),
            'cache_size': len(self._classifier_cache)
        }
        
        return AssistedExtractionResult(extracted_triples, stats, processing_time, 'classifier')
    
    def _extract_with_json(self, text: str, entities: List[str], base_triples: List[Tuple[str, str, str]]) -> AssistedExtractionResult:
        """Extract using JSON-based structured extraction"""
        start = time.perf_counter()
        
        # Create prompt for JSON extraction
        prompt = self._create_json_prompt(text, entities)
        
        try:
            # Call LLM
            response = self._call_llm_json(prompt)
            
            # Parse JSON response
            extracted_triples = self._parse_json_response(response)
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'json',
                'entities_in_prompt': len(entities),
                'triples_extracted': len(extracted_triples),
                'response_length': len(response) if response else 0
            }
            
            return AssistedExtractionResult(extracted_triples, stats, processing_time, 'json')
            
        except Exception as e:
            logger.debug(f"[AssistedExtractor] JSON extraction failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return AssistedExtractionResult([], {'error': str(e)}, processing_time, 'json_failed')
    
    def _extract_with_fallback(self, text: str, entities: List[str], base_triples: List[Tuple[str, str, str]]) -> AssistedExtractionResult:
        """Fallback extraction using simple heuristics"""
        start = time.perf_counter()
        
        # Simple pattern-based extraction as fallback
        extracted_triples = []
        
        # Look for common patterns in text
        patterns = [
            (r'(\w+)\s+(is|are|was|were)\s+(\w+)', lambda m: (m.group(1), m.group(2), m.group(3))),
            (r'(\w+)\s+(has|have)\s+(\w+)', lambda m: (m.group(1), m.group(2), m.group(3))),
        ]
        
        import re
        for pattern, extractor in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    triple = extractor(match)
                    if len(triple) == 3:
                        extracted_triples.append(triple)
                except Exception:
                    pass
        
        processing_time = (time.perf_counter() - start) * 1000
        stats = {
            'method': 'fallback',
            'patterns_found': len(extracted_triples),
            'base_triples_count': len(base_triples)
        }
        
        return AssistedExtractionResult(extracted_triples, stats, processing_time, 'fallback')
    
    def _classify_relation(self, s: str, d: str, context: str) -> str:
        """Classify relation between subject and object"""
        if not requests or not self.model_config:
            return 'unknown'
        
        # Create classification prompt
        prompt = f"""Classify the relationship between "{s}" and "{d}" in this context: "{context}"

Common relations: works_at, lives_in, married_to, has, is, friend_of, parent_of, located_in

Return only the relation name:"""

        try:
            response = requests.post(
                f"{self.model_config.base_url}/chat/completions",
                headers={'Content-Type': 'application/json'},
                json={
                    'model': self.model_config.name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 20,
                    'temperature': 0.1
                },
                timeout=self.model_config.timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip().lower()
                # Clean up response
                content = content.replace('.', '').replace(',', '').replace('"', '')
                return content if content else 'unknown'
            
        except Exception as e:
            logger.debug(f"[AssistedExtractor] LLM call failed: {e}")
        
        return 'unknown'
    
    def _create_json_prompt(self, text: str, entities: List[str]) -> str:
        """Create prompt for JSON-based extraction"""
        entities_str = '", "'.join(entities)
        
        return f"""Extract relationships from this text: "{text}"

Available entities: "{entities_str}"

Return a JSON array with format: [{{"subject": "entity1", "relation": "relationship", "object": "entity2"}}]

Focus on these relation types: works_at, lives_in, married_to, has, is, friend_of, parent_of, located_in

Response:"""
    
    def _call_llm_json(self, prompt: str) -> str:
        """Call LLM for JSON extraction"""
        if not requests or not self.model_config:
            return ""
        
        try:
            response = requests.post(
                f"{self.model_config.base_url}/chat/completions",
                headers={'Content-Type': 'application/json'},
                json={
                    'model': self.model_config.name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 200,
                    'temperature': 0.1
                },
                timeout=self.model_config.timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.debug(f"[AssistedExtractor] JSON LLM call failed: {e}")
        
        return ""
    
    def _parse_json_response(self, response: str) -> List[Tuple[str, str, str]]:
        """Parse JSON response into triples"""
        triples = []
        
        try:
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                for item in data:
                    if isinstance(item, dict):
                        s = item.get('subject', '').strip()
                        r = item.get('relation', '').strip()
                        d = item.get('object', '').strip()
                        if s and r and d:
                            triples.append((s, r, d))
        
        except Exception as e:
            logger.debug(f"[AssistedExtractor] JSON parsing failed: {e}")
        
        return triples
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get assisted extraction performance metrics"""
        metrics = dict(self.metrics)
        metrics.update({
            'total_calls': self._assisted_calls,
            'successful_calls': self._assisted_success,
            'success_rate': self._assisted_success / max(1, self._assisted_calls),
            'cache_size': len(self._classifier_cache)
        })
        return metrics
    
    def clear_cache(self):
        """Clear classifier cache"""
        self._classifier_cache.clear()
        logger.debug("[AssistedExtractor] Classifier cache cleared")


logger.info("🎯 AssistedExtractor initialized - dedicated LLM-assisted extraction service")
logger.info("📊 Features: Classifier-based, JSON-based, fallback methods, performance optimization")