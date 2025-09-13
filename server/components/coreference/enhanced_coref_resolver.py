"""
EnhancedCoreferenceResolver: Advanced Coreference Resolution with spaCy-Coref
========================================================================

SOTA coreference resolution using spacy-coref for enhanced understanding
of entity references and pronoun resolution in text.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except Exception as e:
    SPACY_AVAILABLE = False
    logger.warning(f"[EnhancedCoreferenceResolver] spacy not available: {e}")

try:
    # Try to import spacy-coref if available
    try:
        import spacy_coref
        SPACY_COREF_AVAILABLE = True
    except Exception:
        SPACY_COREF_AVAILABLE = False
        logger.warning("[EnhancedCoreferenceResolver] spacy-coref not available")
except Exception as e:
    SPACY_COREF_AVAILABLE = False
    logger.warning(f"[EnhancedCoreferenceResolver] spacy-coref import failed: {e}")


@dataclass
class CoreferenceChain:
    """Represents a coreference chain"""
    main_entity: str
    mentions: List[str]
    span_indices: List[Tuple[int, int]]
    confidence: float


@dataclass
class EnhancedCoreferenceResult:
    """Result of enhanced coreference resolution"""
    resolved_triples: List[Tuple[str, str, str]]
    coreference_chains: List[CoreferenceChain]
    resolution_stats: Dict[str, Any]
    processing_time_ms: float


class EnhancedCoreferenceResolver:
    """
    Enhanced coreference resolution using spacy and spacy-coref.
    Provides advanced entity resolution and pronoun handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced coreference resolver"""
        self.enabled = config.get('enhanced_coref_enabled', False) and SPACY_AVAILABLE
        self.use_spacy_coref = config.get('use_spacy_coref', True) and SPACY_COREF_AVAILABLE
        self.confidence_threshold = config.get('coref_confidence_threshold', 0.7)
        self.max_entities = config.get('max_coref_entities', 50)
        
        # spaCy models (lazy loaded)
        self._nlp = None
        self._coref_component = None
        
        # Performance optimizations
        self._doc_cache = {}
        self._resolution_cache = {}
        
        # Rule-based enhancements
        self.pronoun_categories = {
            'first_person': {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'},
            'second_person': {'you', 'your', 'yours', 'yourself', 'yourselves'},
            'third_person': {'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
                            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
        }
        
        logger.info(f"[EnhancedCoreferenceResolver] Initialized with enabled={'✓' if self.enabled else '✗'}, "
                   f"spacy-coref={'✓' if self.use_spacy_coref else '✗'}")
    
    def resolve_coreferences(self, triples: List[Tuple[str, str, str]], text: str = "") -> EnhancedCoreferenceResult:
        """
        Main entry point for enhanced coreference resolution
        """
        start = time.perf_counter()
        
        try:
            if not triples or not self.enabled or not text:
                return EnhancedCoreferenceResult(
                    triples, [], {'method': 'none'}, 0.0
                )
            
            # Performance guard
            if len(triples) > self.max_entities:
                logger.debug(f"[EnhancedCoreferenceResolver] Too many triples ({len(triples)}), skipping")
                return EnhancedCoreferenceResult(
                    triples, [], {'method': 'skipped', 'reason': 'too_many_triples'}, 0.0
                )
            
            # Check cache first
            cache_key = str(sorted(triples)) + str(hash(text))
            if cache_key in self._resolution_cache:
                cached_result = self._resolution_cache[cache_key]
                processing_time = (time.perf_counter() - start) * 1000
                return EnhancedCoreferenceResult(
                    cached_result['resolved_triples'],
                    cached_result['coreference_chains'],
                    {'method': 'cached'},
                    processing_time
                )
            
            # Process document
            doc = self._process_text(text)
            if not doc:
                return EnhancedCoreferenceResult(
                    triples, [], {'method': 'document_processing_failed'}, 0.0
                )
            
            # Extract coreference chains
            coreference_chains = self._extract_coreference_chains(doc, text)
            
            # Resolve triples
            resolved_triples = self._resolve_triples_with_chains(triples, coreference_chains, doc)
            
            # Cache result
            self._resolution_cache[cache_key] = {
                'resolved_triples': resolved_triples,
                'coreference_chains': coreference_chains
            }
            
            # Manage cache size
            if len(self._resolution_cache) > 100:
                oldest_keys = list(self._resolution_cache.keys())[:20]
                for key in oldest_keys:
                    del self._resolution_cache[key]
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'enhanced_coreference',
                'original_count': len(triples),
                'resolved_count': len(resolved_triples),
                'chains_found': len(coreference_chains),
                'spacy_coref_used': self.use_spacy_coref,
                'entities_resolved': len([t for t, r in zip(triples, resolved_triples) if t != r])
            }
            
            return EnhancedCoreferenceResult(resolved_triples, coreference_chains, stats, processing_time)
            
        except Exception as e:
            logger.error(f"[EnhancedCoreferenceResolver] Coreference resolution failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return EnhancedCoreferenceResult(
                triples, [], {'error': str(e)}, processing_time
            )
    
    def _process_text(self, text: str) -> Optional[Doc]:
        """Process text with spaCy"""
        
        # Check cache first
        if text in self._doc_cache:
            return self._doc_cache[text]
        
        if self._nlp is None:
            try:
                self._load_spacy()
            except Exception as e:
                logger.error(f"[EnhancedCoreferenceResolver] Failed to load spaCy: {e}")
                return None
        
        try:
            doc = self._nlp(text)
            
            # Cache document
            if len(self._doc_cache) < 50:
                self._doc_cache[text] = doc
            
            return doc
            
        except Exception as e:
            logger.error(f"[EnhancedCoreferenceResolver] Document processing failed: {e}")
            return None
    
    def _extract_coreference_chains(self, doc: Doc, text: str) -> List[CoreferenceChain]:
        """Extract coreference chains from document"""
        
        if self.use_spacy_coref and self._coref_component:
            return self._extract_spacy_coref_chains(doc, text)
        else:
            return self._extract_rule_based_chains(doc, text)
    
    def _extract_spacy_coref_chains(self, doc: Doc, text: str) -> List[CoreferenceChain]:
        """Extract coreference chains using spacy-coref"""
        
        chains = []
        
        try:
            # Use spacy-coref to get coreference clusters
            if hasattr(doc, '_.coref_clusters'):
                clusters = doc._.coref_clusters
                
                for cluster in clusters:
                    if len(cluster) > 1:  # Only consider chains with multiple mentions
                        # Find main entity (usually the most complete form)
                        main_mention = self._find_main_mention(cluster)
                        mentions = [mention.text for mention in cluster]
                        span_indices = [(mention.start_char, mention.end_char) for mention in cluster]
                        
                        chain = CoreferenceChain(
                            main_entity=main_mention,
                            mentions=mentions,
                            span_indices=span_indices,
                            confidence=0.8  # Default confidence for spacy-coref
                        )
                        chains.append(chain)
            
        except Exception as e:
            logger.debug(f"[EnhancedCoreferenceResolver] spacy-coref extraction failed: {e}")
            # Fallback to rule-based
            return self._extract_rule_based_chains(doc, text)
        
        return chains
    
    def _extract_rule_based_chains(self, doc: Doc, text: str) -> List[CoreferenceChain]:
        """Extract coreference chains using rule-based methods"""
        
        chains = []
        
        try:
            # Extract named entities
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
            
            # Group similar entities
            entity_groups = self._group_similar_entities(entities)
            
            for group_name, group_entities in entity_groups.items():
                if len(group_entities) > 1:
                    mentions = [ent[0] for ent in group_entities]
                    span_indices = [(ent[2], ent[3]) for ent in group_entities]
                    
                    chain = CoreferenceChain(
                        main_entity=group_name,
                        mentions=mentions,
                        span_indices=span_indices,
                        confidence=0.6  # Lower confidence for rule-based
                    )
                    chains.append(chain)
            
            # Add pronoun resolution chains
            pronoun_chains = self._resolve_pronouns(doc, text)
            chains.extend(pronoun_chains)
            
        except Exception as e:
            logger.debug(f"[EnhancedCoreferenceResolver] Rule-based extraction failed: {e}")
        
        return chains
    
    def _group_similar_entities(self, entities: List[Tuple[str, str, int, int]]) -> Dict[str, List[Tuple[str, str, int, int]]]:
        """Group similar entities together"""
        
        groups = defaultdict(list)
        
        for entity_text, entity_type, start, end in entities:
            # Normalize entity text
            normalized = self._normalize_entity_text(entity_text)
            
            # Use type + normalized text as key
            key = f"{entity_type}:{normalized}"
            groups[key].append((entity_text, entity_type, start, end))
        
        return dict(groups)
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for comparison"""
        return text.lower().strip()
    
    def _find_main_mention(self, cluster) -> str:
        """Find the main mention in a coreference cluster"""
        
        # Prefer the longest mention as the main entity
        main_mention = max(cluster, key=lambda m: len(m.text))
        return main_mention.text
    
    def _resolve_pronouns(self, doc: Doc, text: str) -> List[CoreferenceChain]:
        """Resolve pronouns using rule-based methods"""
        
        chains = []
        
        try:
            # Simple pronoun resolution based on proximity to entities
            entities = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents]
            
            for token in doc:
                if token.text.lower() in self.pronoun_categories['third_person']:
                    # Find closest entity before this pronoun
                    closest_entity = self._find_closest_entity(token.i, entities)
                    
                    if closest_entity:
                        chain = CoreferenceChain(
                            main_entity=closest_entity[0],
                            mentions=[token.text, closest_entity[0]],
                            span_indices=[(token.idx, token.idx + len(token.text)), 
                                        (closest_entity[1], closest_entity[2])],
                            confidence=0.5  # Lower confidence for pronoun resolution
                        )
                        chains.append(chain)
                        
        except Exception as e:
            logger.debug(f"[EnhancedCoreferenceResolver] Pronoun resolution failed: {e}")
        
        return chains
    
    def _find_closest_entity(self, token_index: int, entities: List[Tuple[str, int, int]]) -> Optional[Tuple[str, int, int]]:
        """Find the closest entity before a token"""
        
        closest_entity = None
        min_distance = float('inf')
        
        for entity_text, start, end in entities:
            # Convert character position to token position (approximate)
            entity_token_pos = start // 3  # Rough estimate
            
            if entity_token_pos < token_index:
                distance = token_index - entity_token_pos
                if distance < min_distance:
                    min_distance = distance
                    closest_entity = (entity_text, start, end)
        
        return closest_entity
    
    def _resolve_triples_with_chains(self, triples: List[Tuple[str, str, str]], 
                                    chains: List[CoreferenceChain], 
                                    doc: Doc) -> List[Tuple[str, str, str]]:
        """Resolve triples using coreference chains"""
        
        resolved_triples = []
        
        # Create resolution map from chains
        resolution_map = {}
        for chain in chains:
            if chain.confidence >= self.confidence_threshold:
                for mention in chain.mentions:
                    if mention != chain.main_entity:
                        resolution_map[mention.lower()] = chain.main_entity
        
        # Apply resolution to triples
        for subject, predicate, obj in triples:
            resolved_subject = resolution_map.get(subject.lower(), subject)
            resolved_object = resolution_map.get(obj.lower(), obj)
            
            resolved_triples.append((resolved_subject, predicate, resolved_object))
        
        return resolved_triples
    
    def _load_spacy(self):
        """Load spaCy model with coreference component"""
        try:
            # Try to load coref model first
            if self.use_spacy_coref:
                try:
                    # Try to load a model with coreference support
                    self._nlp = spacy.load("en_core_web_sm")
                    
                    # Add coreference component if available
                    if SPACY_COREF_AVAILABLE:
                        try:
                            self._nlp.add_pipe("coref", source="spacy_coref")
                            self._coref_component = True
                            logger.debug("[EnhancedCoreferenceResolver] spaCy with coreference loaded")
                        except Exception:
                            self._coref_component = False
                            logger.debug("[EnhancedCoreferenceResolver] spaCy loaded without coreference")
                    else:
                        self._coref_component = False
                        logger.debug("[EnhancedCoreferenceResolver] spaCy loaded without coreference")
                    
                    return
                    
                except Exception:
                    pass
            
            # Fallback to basic spaCy model
            self._nlp = spacy.load("en_core_web_sm")
            self._coref_component = False
            logger.debug("[EnhancedCoreferenceResolver] Basic spaCy loaded")
            
        except Exception:
            try:
                # Fallback to blank model
                self._nlp = spacy.blank("en")
                self._coref_component = False
                logger.debug("[EnhancedCoreferenceResolver] Blank spaCy loaded")
            except Exception as e:
                logger.error(f"[EnhancedCoreferenceResolver] Failed to load any spaCy model: {e}")
                self._nlp = None
    
    def clear_cache(self):
        """Clear all caches"""
        self._doc_cache.clear()
        self._resolution_cache.clear()
        logger.debug("[EnhancedCoreferenceResolver] Caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        return {
            'enabled': self.enabled,
            'spacy_available': SPACY_AVAILABLE,
            'spacy_coref_available': SPACY_COREF_AVAILABLE,
            'coref_component_loaded': self._coref_component,
            'doc_cache_size': len(self._doc_cache),
            'resolution_cache_size': len(self._resolution_cache),
            'confidence_threshold': self.confidence_threshold
        }


logger.info("🎯 EnhancedCoreferenceResolver initialized - advanced coreference resolution")
logger.info("📊 Features: spaCy integration, coreference chains, pronoun resolution")