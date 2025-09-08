"""
Memory Extraction Module

Handles all extraction logic and triple processing for the HotMem system.
Extracts entities, relations, and facts from text using multiple strategies.

Author: SOLID Refactoring
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict
import math
import json
import urllib.request
import urllib.error

from loguru import logger
import spacy
from spacy.tokens import Token
try:
    from unidecode import unidecode  # type: ignore
except Exception:
    def unidecode(s: str) -> str:  # type: ignore
        return s

try:
    from components.processing.semantic_roles import SRLExtractor  # type: ignore
except Exception:
    SRLExtractor = None  # Optional SRL layer
try:
    from services.onnx_nlp import OnnxTokenNER, OnnxSRLTagger  # type: ignore
except Exception:
    OnnxTokenNER = None
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
    from components.extraction.improved_ud_extractor import ImprovedUDExtractor  # type: ignore
except Exception:
    ImprovedUDExtractor = None
try:
    from fastcoref import FCoref  # type: ignore
except Exception:
    FCoref = None
try:
    from components.memory.memory_decomposer import decompose as _decompose_clauses  # type: ignore
except Exception:
    _decompose_clauses = None  # Optional
try:
    from memory_quality import calculate_extraction_confidence as _extra_confidence  # type: ignore
except Exception:
    _extra_confidence = None  # Optional

# Try to import language detection
try:
    import pycld3
    PYCLD3_AVAILABLE = True
except ImportError:
    PYCLD3_AVAILABLE = False
    logger.info("pycld3 not available, defaulting to English")

# Singleton NLP model cache
_nlp_cache = {}

def _load_nlp(lang: str = "en"):
    """Load spaCy model (cached singleton)"""
    if lang not in _nlp_cache:
        try:
            # Keep lemmatizer+parser across all supported langs; disable only NER/textcat for speed
            if lang == "en":
                nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            else:
                nlp = spacy.load(f"{lang}_core_news_sm", disable=["ner", "textcat"])
            _nlp_cache[lang] = nlp
            logger.info(f"Loaded spaCy model {lang}_core_web_sm")
        except:
            _nlp_cache[lang] = None
            logger.warning(f"Could not load spaCy model for {lang}")
    return _nlp_cache[lang]

def _norm(text: str) -> str:
    """Fast normalization"""
    return (text or "").lower().strip()

def _strip_leading_dets(text: str) -> str:
    """Strip leading determiners for entity canon"""
    text = (text or "").strip()
    for d in ["the ", "a ", "an "]:
        if text.startswith(d):
            return text[len(d):]
    return text

def _canon_entity_text(text: str) -> str:
    """Canonical entity text (lower, strip dets, unidecode)"""
    t = _strip_leading_dets(text)
    try:
        t = unidecode(t)
    except:
        pass
    return t.lower().strip()

class MemoryExtractor:
    """
    Handles extraction of entities, relations, and facts from text.
    
    Responsibilities:
    - Text preprocessing and normalization
    - Multiple extraction strategy coordination
    - Triple extraction and refinement
    - Entity resolution and coreference
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.extractors = self._initialize_extractors()
        self.srl_extractor = None
        self.onnx_ner = None
        self.onnx_srl = None
        self.coref_model = None
        
    def _initialize_extractors(self) -> Dict[str, Any]:
        """Initialize available extraction strategies"""
        extractors = {}
        
        # Initialize HotMem extractors
        if HotMemExtractor is not None:
            try:
                extractors['hotmem'] = HotMemExtractor()
                logger.info("Initialized HotMem extractor")
            except Exception as e:
                logger.warning(f"Failed to initialize HotMem extractor: {e}")
        
        if EnhancedHotMemExtractor is not None:
            try:
                extractors['enhanced_hotmem'] = EnhancedHotMemExtractor()
                logger.info("Initialized Enhanced HotMem extractor")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced HotMem extractor: {e}")
        
        if HybridRelationExtractor is not None:
            try:
                extractors['hybrid'] = HybridRelationExtractor()
                logger.info("Initialized Hybrid extractor")
            except Exception as e:
                logger.warning(f"Failed to initialize Hybrid extractor: {e}")
        
        if ImprovedUDExtractor is not None:
            try:
                extractors['improved_ud'] = ImprovedUDExtractor()
                logger.info("Initialized Improved UD extractor")
            except Exception as e:
                logger.warning(f"Failed to initialize Improved UD extractor: {e}")
        
        return extractors
    
    def extract_triples(self, text: str, lang: str = "en") -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and triples from text using multiple strategies.
        
        Args:
            text: Input text to process
            lang: Language code
            
        Returns:
            Tuple of (entities, triples, extraction_time, metadata)
        """
        start_time = time.time()
        
        # Load NLP model
        nlp = _load_nlp(lang)
        if nlp is None:
            return [], [], 0, {}
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract entities and triples
        entities, triples = self._extract_from_doc(doc)
        
        # Apply coreference resolution
        if self.coref_model is not None:
            triples = self._apply_coref_resolution(triples, doc)
        
        extraction_time = int((time.time() - start_time) * 1000)
        
        return entities, triples, extraction_time, {'lang': lang, 'strategies_used': list(self.extractors.keys())}
    
    def _extract_from_doc(self, doc) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Extract entities and triples from spaCy document"""
        entities = set()
        triples = []
        
        # Extract entities using NER
        for ent in doc.ents:
            entities.add(_canon_entity_text(ent.text))
        
        # Extract triples using available strategies
        for strategy_name, extractor in self.extractors.items():
            try:
                if hasattr(extractor, 'extract_triples'):
                    strategy_triples = extractor.extract_triples(doc.text)
                    triples.extend(strategy_triples)
                elif hasattr(extractor, 'extract'):
                    strategy_triples = extractor.extract(doc.text)
                    triples.extend(strategy_triples)
            except Exception as e:
                logger.warning(f"Failed to extract with {strategy_name}: {e}")
        
        # Apply quality filtering
        triples = self._filter_triples(triples)
        
        return list(entities), triples
    
    def _filter_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Filter triples based on quality criteria"""
        filtered = []
        
        for s, r, d in triples:
            # Skip empty or meaningless triples
            if not s or not r or not d:
                continue
            
            # Skip triples with overly generic relations
            if r.lower() in ['is', 'are', 'was', 'were', 'have', 'has', 'do', 'does']:
                continue
            
            # Skip triples where subject and object are the same
            if _canon_entity_text(s) == _canon_entity_text(d):
                continue
            
            filtered.append((s, r, d))
        
        return filtered
    
    def _apply_coref_resolution(self, triples: List[Tuple[str, str, str]], doc) -> List[Tuple[str, str, str]]:
        """Apply coreference resolution to triples"""
        # This would be implemented with the coreference model
        # For now, return triples as-is
        return triples
    
    def get_extraction_strategies(self) -> List[str]:
        """Get list of available extraction strategies"""
        return list(self.extractors.keys())
    
    def is_strategy_available(self, strategy: str) -> bool:
        """Check if a specific extraction strategy is available"""
        return strategy in self.extractors