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
# GLiREL integration removed - using Enhanced Level3 instead

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
           

from components.extraction.enhanced_level3_extractor import QualityExtractor

@dataclass
class ExtractionResult:
    """Result of extraction operation"""
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    negation_count: int
    doc: Optional[Any] = None


# Global singleton cache for expensive models
_GLOBAL_MODEL_CACHE = {
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
        self.use_dspy = config.get('use_dspy', False)
        self.use_gliner = config.get('use_gliner', True)  # Enable GLiNER by default
        
        # Optional extractors (lazy loaded)
        self._srl: Optional[Any] = None
        self._onnx_ner = None
        self._onnx_srl = None
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
        
       
        if 'enhanced_level3' in self.config.get('default_strategy', ''):
            extractor = QualityExtractor(entity_threshold=0.70, relation_threshold=0.65)
            kg = extractor.extract_quality_kg(doc)
            all_entities = [e.text for e in kg['entities']]
            all_triples = [(r.subject, r.predicate, r.object) for r in kg['relations']]
            logger.debug(f"[Enhanced Level3] Extracted {len(all_entities)} entities, {len(all_triples)} relations")
        

                 

           
  
                  
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
            nlp = spacy.load("en_core_web_rtf")
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
logger.info("📊 Strategies: level3")
