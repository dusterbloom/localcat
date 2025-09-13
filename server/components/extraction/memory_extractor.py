"""
MemoryExtractor: Dedicated Entity and Relation Extraction Service
=================================================================

Extracted from HotMemory monolith - now focused solely on extraction:
- Entity recognition and mapping
- Dependency pattern extraction  
- Multiple extraction strategies (UD, SRL, ONNX, ReLiK)
- Light entity extraction for retrieval
"""

import os
import re
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from loguru import logger
import spacy
from spacy.tokens import Token, Doc

# Optional extraction components
try:
    from components.processing.semantic_roles import SRLExtractor  # type: ignore
except Exception:
    SRLExtractor = None
try:
    from services.onnx_nlp import OnnxTokenNER, OnnxSRLTagger  # type: ignore
except Exception:
    OnnxTokenNER = None

# GLiREL integration for zero-shot relation extraction
try:
    from components.extraction.glirel_extractor import GLiRELExtractor
    GLIREL_AVAILABLE = True
except ImportError:
    GLIREL_AVAILABLE = None

# Import centralized UD patterns
try:
    from services.ud_utils import extract_all_ud_patterns, ExtractedRelation
except ImportError:
    extract_all_ud_patterns = None
    logger.debug("[MemoryExtractor] UD patterns not available")
    OnnxSRLTagger = None
try:
    from components.extraction.hotmem_extractor import HotMemExtractor  # type: ignore
except Exception:
    HotMemExtractor = None
try:
    from components.extraction.enhanced_hotmem_extractor import EnhancedHotMemExtractor  # type: ignore
except Exception:
    EnhancedHotMemExtractor = None
try:
    from components.extraction.hybrid_spacy_llm_extractor import HybridRelationExtractor  # type: ignore
except Exception:
    HybridRelationExtractor = None
try:
    from components.extraction.gliner_extractor import GLiNERExtractor  # type: ignore
except Exception:
    GLiNERExtractor = None
try:
    from components.extraction.tiered_extractor import TieredRelationExtractor  # type: ignore
except Exception:
    TieredRelationExtractor = None


@dataclass
class ExtractionResult:
    """Result of extraction operation"""
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    negation_count: int
    doc: Optional[Any] = None


# Global singleton cache for expensive models
_GLOBAL_MODEL_CACHE = {
    'relik': None,
    'relik_loading': False,
    'glirel': None,
    'glirel_loading': False,
    'gliner': None,
    'gliner_loading': False,
    'tiered': None,
    'tiered_loading': False,
    'spacy_models': {},  # Cache for spaCy models by language
    'spacy_loading': {}
}


class MemoryExtractor:
    """
    Dedicated extraction service for entities and relations.
    Handles all extraction strategies and patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize extractor with configuration"""
        self.config = config
        
        # Extraction strategy flags
        self.use_srl = config.get('use_srl', False)
        self.use_onnx_ner = config.get('use_onnx_ner', False)
        self.use_onnx_srl = config.get('use_onnx_srl', False)
        self.use_relik = config.get('use_relik', False)  # Legacy ReLiK (deprecated)
        self.use_glirel = config.get('use_glirel', GLIREL_AVAILABLE is not None)  # Use GLiREL if available
        self.use_dspy = config.get('use_dspy', False)
        self.use_gliner = config.get('use_gliner', True)  # Enable GLiNER by default
        
        # Optional extractors (lazy loaded)
        self._srl: Optional[Any] = None
        self._onnx_ner = None
        self._onnx_srl = None
        self._relik = None  # Legacy ReLiK
        self._glirel = None  # New GLiREL extractor
        self._dspy_extractor = None
        self._gliner = None
        self._tiered_extractor = None
        
        # Performance tracking
        self.metrics = defaultdict(list)
        
    def extract(self, text: str, lang: str = "en", use_cache: bool = True) -> ExtractionResult:
        """
        Main extraction entry point - extracts entities and relations from text
        """
        start = time.perf_counter()
        
        try:
            # Hybrid caching check for performance
            cache_key = None
            if use_cache:
                import hashlib
                cache_key = hashlib.sha256(text.encode()).hexdigest()
                if hasattr(self, '_extraction_cache') and cache_key in self._extraction_cache:
                    cached_result = self._extraction_cache[cache_key]
                    self.metrics['cache_hits'] = self.metrics.get('cache_hits', 0) + 1
                    logger.debug(f"[MemoryExtractor] Cache hit for {len(text)} chars")
                    return cached_result
            
            # Load language model
            doc = _load_nlp(lang)(text) if text else None
            if not doc:
                return ExtractionResult([], [], 0, None)
                
            # Stage 1: Extract using multiple strategies
            entities, triples, neg_count = self._extract_strategies(doc, text)
            
            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics['extraction_ms'].append(elapsed_ms)
            
            result = ExtractionResult(entities, triples, neg_count, doc)
            
            # Cache result for future use
            if use_cache and cache_key is not None:
                if not hasattr(self, '_extraction_cache'):
                    self._extraction_cache = {}
                    self._cache_size = 0
                    self._max_cache_size = 1000
                
                # Manage cache size
                if len(self._extraction_cache) >= self._max_cache_size:
                    # Remove oldest 20% of entries
                    oldest_keys = list(self._extraction_cache.keys())[:self._max_cache_size // 5]
                    for key in oldest_keys:
                        del self._extraction_cache[key]
                
                self._extraction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"[MemoryExtractor] Extraction failed: {e}")
            return ExtractionResult([], [], 0, None)
    
    def extract_entities_light(self, text: str, entity_index: Optional[Set[str]] = None) -> List[str]:
        """
        Light entity extraction for retrieval context - prioritizes GLiNER for accuracy
        """
        try:
            # Use GLiNER if available for superior entity extraction (96.7% accuracy)
            if self.use_gliner and GLiNERExtractor is not None:
                if self._gliner is None:
                    # Check global cache first
                    if _GLOBAL_MODEL_CACHE['gliner'] is not None:
                        self._gliner = _GLOBAL_MODEL_CACHE['gliner']
                    elif not _GLOBAL_MODEL_CACHE['gliner_loading']:
                        _GLOBAL_MODEL_CACHE['gliner_loading'] = True
                        try:
                            self._gliner = GLiNERExtractor()
                            _GLOBAL_MODEL_CACHE['gliner'] = self._gliner
                        finally:
                            _GLOBAL_MODEL_CACHE['gliner_loading'] = False
                try:
                    if self._gliner:
                        gliner_result = self._gliner.extract(text, entity_index)
                        if gliner_result.entities:
                            logger.debug(f"[GLiNER Light] Extracted {len(gliner_result.entities)} entities")
                            return gliner_result.entities
                except Exception as e:
                    logger.debug(f"[GLiNER Light] Failed: {e}")
            
            # Fallback to spaCy if GLiNER unavailable
            try:
                nlp = _load_nlp("en")
                doc = nlp(text)
                entities = [_canon_entity_text(ent.text) for ent in doc.ents]
                if entities:
                    return entities
            except Exception:
                pass
            
            # Final fallback to pattern matching
            return self._extract_entities_light_fallback(text)
                
        except Exception as e:
            logger.debug(f"[MemoryExtractor] Light entity extraction failed: {e}")
            return self._extract_entities_light_fallback(text)
    
    def _extract_entities_light_fallback(self, text: str) -> List[str]:
        """Fallback entity extraction using simple patterns"""
        import re
        
        # Simple pattern-based extraction
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        for word in words:
            # Clean word from punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.append(clean_word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_strategies(self, doc: Doc, text: str) -> Tuple[List[str], List[Tuple[str, str, str]], int]:
        """Extract using multiple complementary strategies"""
        all_entities = set()
        all_triples = []
        neg_count = 0
        
        # Strategy 0: GLiNER entity extraction (if enabled)
        if self.use_gliner and GLiNERExtractor is not None:
            if self._gliner is None:
                # Check global cache first
                if _GLOBAL_MODEL_CACHE['gliner'] is not None:
                    self._gliner = _GLOBAL_MODEL_CACHE['gliner']
                elif not _GLOBAL_MODEL_CACHE['gliner_loading']:
                    _GLOBAL_MODEL_CACHE['gliner_loading'] = True
                    try:
                        self._gliner = GLiNERExtractor()
                        _GLOBAL_MODEL_CACHE['gliner'] = self._gliner
                    finally:
                        _GLOBAL_MODEL_CACHE['gliner_loading'] = False
            try:
                if self._gliner:
                    gliner_result = self._gliner.extract(text)
                    all_entities.update(gliner_result.entities)
                    logger.debug(f"[GLiNER] Extracted {len(gliner_result.entities)} entities")
            except Exception as e:
                logger.debug(f"[GLiNER] Extraction failed: {e}")
        
        # Strategy 1: Tiered extraction (FORCE TIER 1 ONLY for performance)
        if TieredRelationExtractor is not None:
            if self._tiered_extractor is None:
                # Check global cache first
                if _GLOBAL_MODEL_CACHE['tiered'] is not None:
                    self._tiered_extractor = _GLOBAL_MODEL_CACHE['tiered']
                elif not _GLOBAL_MODEL_CACHE['tiered_loading']:
                    _GLOBAL_MODEL_CACHE['tiered_loading'] = True
                    try:
                        # Initialize with coref and SRL enabled
                        self._tiered_extractor = TieredRelationExtractor(
                            enable_srl=self.use_srl,
                            enable_coref=self.config.get('use_coref', False),  # Disable coref for speed
                            llm_base_url=self.config.get('llm_base_url', 'http://127.0.0.1:1234/v1')
                        )
                        _GLOBAL_MODEL_CACHE['tiered'] = self._tiered_extractor
                    finally:
                        _GLOBAL_MODEL_CACHE['tiered_loading'] = False
            try:
                # FORCE TIER 1 ONLY - directly call _extract_tier1 instead of extract
                if hasattr(self._tiered_extractor, '_extract_tier1'):
                    tiered_result = self._tiered_extractor._extract_tier1(text, doc)
                    all_entities.update(tiered_result.entities)
                    all_triples.extend(tiered_result.relationships)
                    logger.debug(f"[Tiered] Extracted {len(tiered_result.relationships)} relationships using tier 1 (forced)")
                else:
                    # Fallback if _extract_tier1 not available
                    tiered_result = self._tiered_extractor.extract(text, doc)
                    all_entities.update(tiered_result.entities)
                    all_triples.extend(tiered_result.relationships)
                    logger.debug(f"[Tiered] Extracted {len(tiered_result.relationships)} relationships using tier {tiered_result.tier_used}")
            except Exception as e:
                logger.debug(f"[Tiered] Extraction failed: {e}")
                # Fallback to UD patterns
                entities, triples, neg_count = self._extract_from_doc(doc)
                all_entities.update(entities)
                all_triples.extend(triples)
        else:
            # Fallback: Universal Dependencies extraction (base)
            entities, triples, neg_count = self._extract_from_doc(doc)
            all_entities.update(entities)
            all_triples.extend(triples)
        
        # Strategy 2: SRL enhancement (if enabled)
        if self.use_srl and SRLExtractor is not None and self._srl is None:
            try:
                self._srl = SRLExtractor(use_normalizer=True)
            except Exception:
                pass
                
        if self.use_srl and self._srl:
            try:
                srl_trips = self._srl.extract(doc)
                for (s, r, d) in srl_trips:
                    all_entities.add(_canon_entity_text(s))
                    all_entities.add(_canon_entity_text(d))
                all_triples.extend(srl_trips)
            except Exception:
                pass
        
        # Strategy 3: ONNX enhancement (if enabled)
        if self.use_onnx_ner and self._onnx_ner is None:
            self._init_onnx_ner()
        if self.use_onnx_srl and self._onnx_srl is None:
            self._init_onnx_srl()
            
        # Strategy 4: Rule-based fast paths for common patterns
        if len(all_triples) < 3:  # Only apply if we need more relations
            rule_triples = self._extract_rule_based_fast_paths(text, all_entities)
            all_triples.extend(rule_triples)
            logger.debug(f"[Rule-based] Extracted {len(rule_triples)} fast-path relations")
            
        # Strategy 5: GLiREL enhancement (if enabled) - 2025 SOTA zero-shot relation extraction
        if self.use_glirel:
            if self._glirel is None:
                self._init_glirel()

            if self._glirel:
                try:
                    glirel_start = time.perf_counter()

                    # Convert our entities to GLiREL format
                    glirel_entities = []
                    for entity_text in all_entities:
                        # Find entity position in text
                        pos = text.find(entity_text)
                        if pos != -1:
                            glirel_entities.append({
                                'text': entity_text,
                                'start': pos,
                                'end': pos + len(entity_text),
                                'label': 'ENTITY'  # GLiREL will infer types
                            })

                    # Extract relations using GLiREL
                    glirel_triples = self._glirel.extract_with_gliner_integration(
                        text=text,
                        gliner_result=glirel_entities,
                        threshold=0.5  # Confidence threshold
                    )

                    # Add GLiREL relations to results
                    glirel_count = 0
                    for triple in glirel_triples:
                        subject, relation, obj = triple
                        # Avoid duplicates and low-quality relations
                        if (subject, relation, obj) not in all_triples and len(relation) > 1:
                            all_triples.append((subject, relation, obj))
                            all_entities.add(subject)
                            all_entities.add(obj)
                            glirel_count += 1
                            logger.debug(f"[GLiREL] Relation: {subject} --{relation}--> {obj}")

                    glirel_time = (time.perf_counter() - glirel_start) * 1000
                    logger.info(f"[GLiREL] Added {glirel_count} relations in {glirel_time:.1f}ms")
                    self.metrics['glirel_ms'].append(glirel_time)

                except Exception as e:
                    logger.warning(f"[GLiREL] Enhancement failed: {e}")

        # Legacy ReLiK support (if explicitly enabled)
        if self.use_relik and not self.use_glirel:
            if self._relik is None:
                self._init_relik()
            
            if self._relik:
                try:
                    relik_start = time.perf_counter()

                    # Check if this is a full Relik model (from our new initialization)
                    if hasattr(self._relik, '__call__') and hasattr(self._relik, 'reader'):
                        # This is a full Relik model - use it directly
                        # It handles sample formatting internally
                        # Optionally provide entity candidates from GLiNER
                        candidates = list(all_entities) if all_entities else None

                        import torch
                        with torch.no_grad():
                            try:
                                # Call the Relik model directly - it handles formatting
                                relik_result = self._relik(text, candidates=candidates)
                                logger.debug(f"[ReLiK] Model returned result type: {type(relik_result)}")
                            except Exception as relik_error:
                                logger.debug(f"[ReLiK] Model call failed: {relik_error}")
                                relik_result = None

                    # Handle wrapper extractors (HybridRelationExtractor, etc.)
                    elif hasattr(self._relik, 'extract'):
                        relik_result = self._relik.extract(text)
                    elif callable(self._relik):
                        relik_result = self._relik(text)
                    else:
                        logger.warning("[ReLiK] ReLiK object is not callable and has no extract method")
                        relik_result = None
                    
                    # CRITICAL FIX: Handle case where ReLiK returns None when retriever is disabled
                    if relik_result is None:
                        logger.debug("[ReLiK] ReLiK returned None (likely due to disabled retriever)")
                        # Try to use the reader directly if available
                        if hasattr(self._relik, 'reader') and self._relik.reader is not None:
                            try:
                                # Import proper ReLiK data structures
                                from relik.reader.pytorch_modules.hf.modeling_relik import RelikReaderSample
                                from dataclasses import dataclass

                                @dataclass
                                class SimpleSpan:
                                    start: int
                                    end: int
                                    text: str
                                    label: str = "--NME--"

                                # Convert existing entities to spans
                                spans = []
                                for entity_text in all_entities:
                                    # Find entity position in text (approximate)
                                    pos = text.find(entity_text)
                                    if pos != -1:
                                        spans.append(SimpleSpan(pos, pos + len(entity_text), entity_text))

                                # Create proper RelikReaderSample with all required attributes
                                sample = RelikReaderSample(
                                    text=text,
                                    spans=spans,
                                    candidates=[],
                                    offset=0,  # Required attribute
                                    _mixin_prediction_position=None  # Required by ReLiK
                                )

                                # Use the proper ReLiK reader API
                                import torch
                                with torch.no_grad():
                                    reader_result = self._relik.reader.read(
                                        text=[text],
                                        samples=[sample],
                                        max_length=256
                                    )

                                # Process reader result
                                if reader_result and len(reader_result) > 0 and hasattr(reader_result[0], 'triplets'):
                                    relik_result = reader_result[0]
                                    logger.debug("[ReLiK] Successfully used reader directly")
                                else:
                                    logger.debug("[ReLiK] Reader returned empty result")

                            except Exception as reader_e:
                                logger.debug(f"[ReLiK] Reader direct call failed: {reader_e}")
                        else:
                            logger.debug("[ReLiK] No reader available for fallback")
                    
                    # Convert ReLiK result to our format - handle triplets properly
                    if relik_result is not None and hasattr(relik_result, 'triplets'):
                        # ReLiK returns triplets with subject, object, and label attributes
                        relik_triplet_count = 0
                        for triplet in relik_result.triplets:
                            if hasattr(triplet, 'subject') and hasattr(triplet, 'object') and hasattr(triplet, 'label'):
                                # Extract text from subject/object spans
                                subject = triplet.subject.text if hasattr(triplet.subject, 'text') else str(triplet.subject)
                                obj = triplet.object.text if hasattr(triplet.object, 'text') else str(triplet.object)
                                relation = triplet.label
                                
                                # Add to results
                                all_triples.append((subject, relation, obj))
                                all_entities.add(subject)
                                all_entities.add(obj)
                                relik_triplet_count += 1
                                
                                logger.debug(f"[ReLiK] Triplet: {subject} --{relation}--> {obj}")
                        
                        logger.info(f"[ReLiK] Added {relik_triplet_count} relations from triplets")
                    
                    # Fallback for other formats
                    elif relik_result is not None and hasattr(relik_result, 'triples'):
                        all_triples.extend(relik_result.triples)
                    elif relik_result is not None and hasattr(relik_result, 'relationships'):
                        all_triples.extend(relik_result.relationships)
                    
                    relik_time = (time.perf_counter() - relik_start) * 1000
                    logger.info(f"[ReLiK] Extracted {len(all_triples)} total relations in {relik_time:.1f}ms")
                    self.metrics['relik_ms'].append(relik_time)
                except Exception as e:
                    logger.warning(f"[ReLiK] Enhancement failed: {e}")
            
        return list(all_entities), all_triples, neg_count
    
    def _extract_from_doc(self, doc) -> Tuple[List[str], List[Tuple[str, str, str]], int]:
        """Extract entities and triples using centralized UD utilities when available.

        Keeps this class focused on orchestration (SRP) and depends on a
        reusable extractor module (DIP) instead of re‑implementing patterns.
        """
        ents_set: Set[str] = set()
        triples_list: List[Tuple[str, str, str]] = []
        neg_count = 0

        # Prefer centralized UD pattern extractor if available
        if extract_all_ud_patterns is not None:
            try:
                ud_relations = extract_all_ud_patterns(doc.text, _load_nlp("en"))
                for rel in ud_relations:
                    if hasattr(rel, 'subject') and hasattr(rel, 'relation') and hasattr(rel, 'object'):
                        s, r, d = rel.subject, rel.relation, rel.object
                        if s and r and d:
                            triples_list.append((s, r, d))
                            ents_set.add(s)
                            ents_set.add(d)
                # Approximate negation count
                try:
                    for sent in doc.sents:
                        if any(ch.dep_ == "neg" for ch in sent.root.children):
                            neg_count += 1
                except Exception:
                    pass
                return list(ents_set), triples_list, neg_count
            except Exception as e:
                logger.debug(f"[MemoryExtractor] UD utils fallback error: {e}")

        # Minimal local fallback if UD utils are unavailable
        try:
            for ent in getattr(doc, 'ents', []):
                ents_set.add(_canon_entity_text(ent.text))
            for token in doc:
                if token.dep_ in {"nsubj", "nsubjpass"} and token.head.pos_ == "VERB":
                    subj = _canon_entity_text(token.text)
                    verb = token.head.lemma_.lower()
                    obj = None
                    for ch in token.head.children:
                        if ch.dep_ in {"dobj", "obj"}:
                            obj = _canon_entity_text(ch.text)
                            break
                    if obj:
                        triples_list.append((subj, verb, obj))
                        ents_set.update([subj, obj])
            return list(ents_set), triples_list, neg_count
        except Exception:
            return [], [], 0
    
    def _build_entity_map(self, doc, entities: Set[str]) -> Dict[int, str]:
        """Build token index to entity mapping"""
        entity_map = {}
        
        # Add pre-defined entities
        for ent in getattr(doc, 'ents', []):
            for token_idx in range(ent.start, ent.end):
                entity_map[token_idx] = ent.text
        
        # Add entities from extraction
        for entity in entities:
            entity_text = entity.lower()
            for token in doc:
                if token.text.lower() == entity_text:
                    entity_map[token.i] = entity_text
                    
        return entity_map
    
    def _init_onnx_ner(self):
        """Initialize ONNX NER model"""
        try:
            ner_model = os.getenv("HOTMEM_ONNX_NER_MODEL", "")
            ner_labels = os.getenv("HOTMEM_ONNX_NER_LABELS", "")
            base_dir = os.path.dirname(__file__)
            
            if ner_model and not os.path.isabs(ner_model):
                ner_model = os.path.abspath(os.path.join(base_dir, ner_model))
            if ner_labels and not os.path.isabs(ner_labels):
                ner_labels = os.path.abspath(os.path.join(base_dir, ner_labels))
                
            ner_tok = os.getenv("HOTMEM_ONNX_NER_TOKENIZER", "bert-base-cased")
            self._onnx_ner = OnnxTokenNER(ner_model, ner_labels, tokenizer_name=ner_tok)
            logger.info("[MemoryExtractor ONNX] NER ready")
        except Exception as e:
            logger.warning(f"[MemoryExtractor ONNX] NER unavailable: {e}")
            self._onnx_ner = None
    
    def _init_onnx_srl(self):
        """Initialize ONNX SRL model"""
        try:
            srl_model = os.getenv("HOTMEM_ONNX_SRL_MODEL", "")
            srl_labels = os.getenv("HOTMEM_ONNX_SRL_LABELS", "")
            srl_tok = os.getenv("HOTMEM_ONNX_SRL_TOKENIZER", "bert-base-cased")
            self._onnx_srl = OnnxSRLTagger(srl_model, srl_labels, tokenizer_name=srl_tok)
            logger.info("[MemoryExtractor ONNX] SRL ready")
        except Exception as e:
            logger.warning(f"[MemoryExtractor ONNX] SRL unavailable: {e}")
            self._onnx_srl = None
    
    def _init_relik(self):
        """Initialize ReLiK extractor with MPS optimization and global caching"""
        global _GLOBAL_MODEL_CACHE

        # Check if already loaded globally
        if _GLOBAL_MODEL_CACHE['relik'] is not None:
            self._relik = _GLOBAL_MODEL_CACHE['relik']
            logger.debug("[MemoryExtractor ReLiK] Using cached global ReLiK model")
            return

        # Check if another thread is loading
        if _GLOBAL_MODEL_CACHE['relik_loading']:
            logger.debug("[MemoryExtractor ReLiK] Waiting for another thread to load ReLiK...")
            import time
            timeout = 30  # 30 second timeout
            start = time.time()
            while _GLOBAL_MODEL_CACHE['relik_loading'] and (time.time() - start) < timeout:
                time.sleep(0.1)

            if _GLOBAL_MODEL_CACHE['relik'] is not None:
                self._relik = _GLOBAL_MODEL_CACHE['relik']
                logger.debug("[MemoryExtractor ReLiK] Using ReLiK loaded by another thread")
                return

        # Mark as loading
        _GLOBAL_MODEL_CACHE['relik_loading'] = True

        try:
            import torch
            # Auto-detect best device for M4 optimization
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("[MemoryExtractor ReLiK] Using MPS acceleration for M4")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("[MemoryExtractor ReLiK] Using CUDA acceleration")
            else:
                device = "cpu"
                logger.info("[MemoryExtractor ReLiK] Using CPU (no acceleration available)")

            # Set optimized parameters for M4
            relik_id = os.getenv("HOTMEM_RELIK_MODEL_ID", "relik-ie/relik-relation-extraction-small")

            # Try loading full ReLiK model without retriever (avoids Wikipedia loading)
            try:
                from relik import Relik

                # Use the full Relik model but disable retriever to avoid Wikipedia loading
                # This gives us the proper interface that handles sample formatting
                relik_model = Relik.from_pretrained(relik_id, retriever=None, device=device)
                logger.info(f"[MemoryExtractor ReLiK] Loaded Relik model without retriever: {relik_id}")
                _GLOBAL_MODEL_CACHE['relik'] = relik_model
                self._relik = relik_model
                relik_loaded = True
            except Exception as e:
                logger.debug(f"[MemoryExtractor ReLiK] Full Relik model failed: {e}")
                relik_loaded = False

            # Fallback to hybrid extractor if ReLiK failed
            if not relik_loaded:
                if HybridRelationExtractor is not None:
                    relik_model = HybridRelationExtractor(device=device)
                    logger.info("[MemoryExtractor ReLiK] Using hybrid spaCy+LLM extractor")
                    _GLOBAL_MODEL_CACHE['relik'] = relik_model
                    self._relik = relik_model
                # Try enhanced replacement second
                elif EnhancedHotMemExtractor is not None:
                    relik_model = EnhancedHotMemExtractor(device=device)
                    logger.info("[MemoryExtractor] Using enhanced HotMem extractor")
                    _GLOBAL_MODEL_CACHE['relik'] = relik_model
                    self._relik = relik_model
                elif HotMemExtractor is not None:
                    # Pass optimized parameters for M4
                    relik_model = HotMemExtractor(
                        model_id=relik_id,
                        device=device
                    )
                    logger.info(f"[MemoryExtractor ReLiK] ready: {relik_id} on {device}")

                    # Apply M4-specific optimizations if available
                    if hasattr(relik_model, 'relik'):
                        try:
                            # Optimize for speed on M4
                            if hasattr(relik_model.relik, 'top_k'):
                                relik_model.relik.top_k = 10  # Reduced from 30 for speed
                            if hasattr(relik_model.relik, 'window_size'):
                                relik_model.relik.window_size = 64
                            if hasattr(relik_model.relik, 'window_stride'):
                                relik_model.relik.window_stride = 32
                            logger.info("[MemoryExtractor ReLiK] Applied M4 optimizations")
                        except Exception as e:
                            logger.debug(f"[MemoryExtractor ReLiK] Optimization failed: {e}")

                    _GLOBAL_MODEL_CACHE['relik'] = relik_model
                    self._relik = relik_model
                else:
                    logger.warning("[MemoryExtractor ReLiK] No extractor available")
                    self._relik = None
        except Exception as e:
            logger.warning(f"[MemoryExtractor ReLiK] unavailable: {e}")
            self._relik = None
        finally:
            # Mark as not loading
            _GLOBAL_MODEL_CACHE['relik_loading'] = False

    def _init_glirel(self):
        """Initialize GLiREL extractor with MPS optimization and global caching"""
        global _GLOBAL_MODEL_CACHE

        if not GLIREL_AVAILABLE:
            logger.warning("[MemoryExtractor GLiREL] GLiREL not available")
            self._glirel = None
            return

        # Check if already loaded globally
        if _GLOBAL_MODEL_CACHE['glirel'] is not None:
            self._glirel = _GLOBAL_MODEL_CACHE['glirel']
            logger.debug("[MemoryExtractor GLiREL] Using cached global GLiREL model")
            return

        # Check if another thread is loading
        if _GLOBAL_MODEL_CACHE['glirel_loading']:
            logger.debug("[MemoryExtractor GLiREL] Waiting for another thread to load GLiREL...")
            import time
            timeout = 60  # 60 second timeout for larger model
            start = time.time()
            while _GLOBAL_MODEL_CACHE['glirel_loading'] and (time.time() - start) < timeout:
                time.sleep(0.1)
            if _GLOBAL_MODEL_CACHE['glirel'] is not None:
                self._glirel = _GLOBAL_MODEL_CACHE['glirel']
                logger.debug("[MemoryExtractor GLiREL] Using GLiREL loaded by another thread")
                return

        # Mark as loading
        _GLOBAL_MODEL_CACHE['glirel_loading'] = True
        try:
            import torch
            # Auto-detect best device for M4 optimization
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("[MemoryExtractor GLiREL] Using MPS acceleration for M4")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("[MemoryExtractor GLiREL] Using CUDA acceleration")
            else:
                device = "cpu"
                logger.info("[MemoryExtractor GLiREL] Using CPU (no acceleration available)")

            # Try loading GLiREL model
            glirel_id = self.config.get('glirel_model_id', 'jackboyla/glirel-large-v0')
            try:
                glirel_model = GLiRELExtractor(model_id=glirel_id, device=device)
                logger.info(f"[MemoryExtractor GLiREL] Loaded GLiREL model: {glirel_id}")
                _GLOBAL_MODEL_CACHE['glirel'] = glirel_model
                self._glirel = glirel_model
            except Exception as e:
                logger.warning(f"[MemoryExtractor GLiREL] Failed to load GLiREL: {e}")
                self._glirel = None

        except Exception as e:
            logger.warning(f"[MemoryExtractor GLiREL] unavailable: {e}")
            self._glirel = None
        finally:
            # Mark as not loading
            _GLOBAL_MODEL_CACHE['glirel_loading'] = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction performance metrics"""
        return dict(self.metrics)
    
    def _extract_rule_based_fast_paths(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Extract common relations using fast rule-based patterns"""
        triples = []
        text_lower = text.lower()
        
        # Common patterns for fast extraction
        patterns = [
            (r'(\w+)\s+works?\s+(?:at|for)\s+(\w+)', 'works_at'),
            (r'(\w+)\s+(?:is|was)\s+(?:a|an)?\s*(\w+)\s+(?:director|manager|ceo|president)', 'has_position'),
            (r'(\w+)\s+(?:studied|graduated)\s+(?:at|from)\s+(\w+)', 'educated_at'),
            (r'(\w+)\s+(?:joined|started)\s+(\w+)', 'works_at'),
            (r'(\w+)\s+(?:founded|created)\s+(\w+)', 'founder_of'),
            (r'(\w+)\s+(?:has|with)\s+(\d+)\s+employees', 'has_employees'),
        ]
        
        for pattern, relation in patterns:
            for match in re.finditer(pattern, text_lower):
                subject = match.group(1).title()
                obj = match.group(2).title()
                if subject in entities or obj in entities:
                    triples.append((subject, relation, obj))
        
        return triples


# Helper functions (extracted from original)
def _canon_entity_text(text: str) -> str:
    """Canonicalize entity text"""
    return text.strip().lower()

def _load_nlp(lang: str = "en"):
    """Load spaCy model with global caching for performance"""
    global _GLOBAL_MODEL_CACHE

    # Check if already cached
    if lang in _GLOBAL_MODEL_CACHE['spacy_models']:
        return _GLOBAL_MODEL_CACHE['spacy_models'][lang]

    # Check if another thread is loading this model
    if lang in _GLOBAL_MODEL_CACHE.get('spacy_loading', {}) and _GLOBAL_MODEL_CACHE['spacy_loading'][lang]:
        import time
        timeout = 10  # 10 second timeout for spaCy models
        start = time.time()
        while _GLOBAL_MODEL_CACHE['spacy_loading'].get(lang, False) and (time.time() - start) < timeout:
            time.sleep(0.05)

        # Check if it was loaded
        if lang in _GLOBAL_MODEL_CACHE['spacy_models']:
            return _GLOBAL_MODEL_CACHE['spacy_models'][lang]

    # Mark as loading
    if 'spacy_loading' not in _GLOBAL_MODEL_CACHE:
        _GLOBAL_MODEL_CACHE['spacy_loading'] = {}
    _GLOBAL_MODEL_CACHE['spacy_loading'][lang] = True

    try:
        # Try to load the English model
        try:
            nlp = spacy.load("en_core_web_sm")
            _GLOBAL_MODEL_CACHE['spacy_models'][lang] = nlp
            return nlp
        except Exception:
            try:
                # Fallback to loading any available model
                nlp = spacy.load("en_core_web_md")
                _GLOBAL_MODEL_CACHE['spacy_models'][lang] = nlp
                return nlp
            except Exception:
                try:
                    # Final fallback - create a blank model
                    nlp = spacy.blank(lang)
                    _GLOBAL_MODEL_CACHE['spacy_models'][lang] = nlp
                    return nlp
                except Exception:
                    # If all else fails, return None
                    return None
    finally:
        # Mark as not loading
        _GLOBAL_MODEL_CACHE['spacy_loading'][lang] = False

    

logger.info("🎯 MemoryExtractor initialized - dedicated extraction service")
logger.info("📊 Strategies: UD, SRL, ONNX, GLiREL, DSPy")
