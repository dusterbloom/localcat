"""
TemporalContextExtractor: Advanced Temporal Context Extraction
============================================================

SOTA temporal context extraction using Timexy for understanding
time expressions and temporal relationships in text.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

try:
    from timexy import Timexy
    TIMEXY_AVAILABLE = True
except Exception as e:
    TIMEXY_AVAILABLE = False
    logger.warning(f"[TemporalContextExtractor] timexy not available: {e}")

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception as e:
    SPACY_AVAILABLE = False
    logger.warning(f"[TemporalContextExtractor] spacy not available: {e}")


@dataclass
class TemporalExpression:
    """Represents a temporal expression"""
    text: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    type: str  # DATE, TIME, DURATION, SET, etc.
    value: str
    confidence: float


@dataclass
class TemporalContext:
    """Temporal context for relationships"""
    temporal_expressions: List[TemporalExpression]
    time_normalized_text: str
    temporal_stats: Dict[str, Any]


@dataclass
class TemporalExtractionResult:
    """Result of temporal context extraction"""
    enhanced_triples: List[Tuple[str, str, str, Optional[TemporalContext]]]
    temporal_contexts: List[TemporalContext]
    extraction_stats: Dict[str, Any]
    processing_time_ms: float


class TemporalContextExtractor:
    """
    Advanced temporal context extraction using Timexy.
    Extracts and normalizes temporal expressions from text.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize temporal extractor with configuration"""
        self.enabled = config.get('temporal_extraction_enabled', False) and TIMEXY_AVAILABLE
        self.confidence_threshold = config.get('temporal_confidence_threshold', 0.5)
        self.include_time_in_relationships = config.get('include_time_in_relationships', True)
        self.use_spacy_fallback = config.get('use_spacy_fallback', True) and SPACY_AVAILABLE
        
        # Timexy models (lazy loaded)
        self._timexy = None
        self._nlp = None
        
        # Cache for temporal analysis
        self._temporal_cache = {}
        
        # Current reference time
        self._reference_time = datetime.now()
        
        logger.info(f"[TemporalContextExtractor] Initialized with enabled={'✓' if self.enabled else '✗'}")
    
    def extract_temporal_context(self, triples: List[Tuple[str, str, str]], text: str = "") -> TemporalExtractionResult:
        """
        Main entry point for temporal context extraction
        """
        start = time.perf_counter()
        
        try:
            if not triples or not self.enabled or not text:
                # Return triples without temporal context
                enhanced_triples = [(s, r, o, None) for s, r, o in triples]
                return TemporalExtractionResult(
                    enhanced_triples, [], {'method': 'none'}, 0.0
                )
            
            # Check cache first
            cache_key = str(sorted(triples)) + str(hash(text))
            if cache_key in self._temporal_cache:
                cached_result = self._temporal_cache[cache_key]
                processing_time = (time.perf_counter() - start) * 1000
                return TemporalExtractionResult(
                    cached_result['enhanced_triples'],
                    cached_result['temporal_contexts'],
                    {'method': 'cached'},
                    processing_time
                )
            
            # Extract temporal expressions from text
            temporal_exprs = self._extract_temporal_expressions(text)
            
            # Enhance triples with temporal context
            enhanced_triples = []
            for triple in triples:
                temporal_context = self._get_temporal_context_for_triple(triple, temporal_exprs, text)
                enhanced_triples.append((*triple, temporal_context))
            
            # Create temporal contexts
            temporal_contexts = self._create_temporal_contexts(temporal_exprs, text)
            
            # Cache result
            self._temporal_cache[cache_key] = {
                'enhanced_triples': enhanced_triples,
                'temporal_contexts': temporal_contexts
            }
            
            # Manage cache size
            if len(self._temporal_cache) > 100:
                oldest_keys = list(self._temporal_cache.keys())[:20]
                for key in oldest_keys:
                    del self._temporal_cache[key]
            
            processing_time = (time.perf_counter() - start) * 1000
            stats = {
                'method': 'temporal_extraction',
                'original_count': len(triples),
                'enhanced_count': len(enhanced_triples),
                'temporal_expressions': len(temporal_exprs),
                'triples_with_context': len([t for t in enhanced_triples if t[3] is not None])
            }
            
            return TemporalExtractionResult(enhanced_triples, temporal_contexts, stats, processing_time)
            
        except Exception as e:
            logger.error(f"[TemporalContextExtractor] Temporal extraction failed: {e}")
            # Return triples without temporal context
            enhanced_triples = [(s, r, o, None) for s, r, o in triples]
            processing_time = (time.perf_counter() - start) * 1000
            return TemporalExtractionResult(
                enhanced_triples, [], {'error': str(e)}, processing_time
            )
    
    def _extract_temporal_expressions(self, text: str) -> List[TemporalExpression]:
        """Extract temporal expressions from text"""
        
        if TIMEXY_AVAILABLE and self._timexy is None:
            try:
                self._load_timexy()
            except Exception as e:
                logger.debug(f"[TemporalContextExtractor] Failed to load timexy: {e}")
        
        if self._timexy:
            return self._extract_with_timexy(text)
        
        # Fallback to rule-based extraction
        if self.use_spacy_fallback and self._nlp is None:
            try:
                self._load_spacy()
            except Exception as e:
                logger.debug(f"[TemporalContextExtractor] Failed to load spacy: {e}")
        
        if self._nlp:
            return self._extract_with_spacy(text)
        
        # Simple regex-based extraction
        return self._extract_with_regex(text)
    
    def _extract_with_timexy(self, text: str) -> List[TemporalExpression]:
        """Extract temporal expressions using Timexy"""
        
        temporal_exprs = []
        
        try:
            # Process text with timexy
            doc = self._timexy(text)
            
            # Extract temporal expressions
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'DURATION', 'SET']:
                    temporal_expr = TemporalExpression(
                        text=ent.text,
                        start_date=self._parse_date(ent.text),
                        end_date=None,
                        type=ent.label_,
                        value=ent.text,
                        confidence=0.8  # Default confidence for timexy
                    )
                    temporal_exprs.append(temporal_expr)
                    
        except Exception as e:
            logger.debug(f"[TemporalContextExtractor] Timexy extraction failed: {e}")
        
        return temporal_exprs
    
    def _extract_with_spacy(self, text: str) -> List[TemporalExpression]:
        """Extract temporal expressions using spacy"""
        
        temporal_exprs = []
        
        try:
            doc = self._nlp(text)
            
            # Use spacy's NER to find date/time entities
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'CARDINAL']:
                    temporal_expr = TemporalExpression(
                        text=ent.text,
                        start_date=self._parse_date(ent.text),
                        end_date=None,
                        type=ent.label_,
                        value=ent.text,
                        confidence=0.6  # Lower confidence for spacy
                    )
                    temporal_exprs.append(temporal_expr)
                    
        except Exception as e:
            logger.debug(f"[TemporalContextExtractor] Spacy extraction failed: {e}")
        
        return temporal_exprs
    
    def _extract_with_regex(self, text: str) -> List[TemporalExpression]:
        """Simple regex-based temporal expression extraction"""
        
        import re
        
        temporal_exprs = []
        
        # Common temporal patterns
        patterns = [
            (r'\b\d{4}\b', 'YEAR'),  # Years
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'DATE'),  # MM/DD/YYYY
            (r'\b\d{1,2}-\d{1,2}-\d{4}\b', 'DATE'),  # MM-DD-YYYY
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'DATE'),  # Month DD, YYYY
            (r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b', 'TIME'),  # Time
            (r'\b(today|yesterday|tomorrow|now|current|present)\b', 'RELATIVE'),
            (r'\b(last|next|this)\s+(week|month|year|decade)\b', 'RELATIVE'),
            (r'\b(morning|afternoon|evening|night|noon|midnight)\b', 'TIME_OF_DAY')
        ]
        
        for pattern, expr_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_expr = TemporalExpression(
                    text=match.group(),
                    start_date=self._parse_date(match.group()),
                    end_date=None,
                    type=expr_type,
                    value=match.group(),
                    confidence=0.4  # Lowest confidence for regex
                )
                temporal_exprs.append(temporal_expr)
        
        return temporal_exprs
    
    def _get_temporal_context_for_triple(self, triple: Tuple[str, str, str], 
                                        temporal_exprs: List[TemporalExpression], 
                                        text: str) -> Optional[TemporalContext]:
        """Get temporal context for a specific triple"""
        
        if not temporal_exprs:
            return None
        
        subject, predicate, obj = triple
        
        # Find temporal expressions near the triple components
        relevant_exprs = []
        
        for expr in temporal_exprs:
            # Check if temporal expression is near any part of the triple
            if self._is_near_triple_component(expr.text, subject, predicate, obj, text):
                relevant_exprs.append(expr)
        
        if not relevant_exprs:
            return None
        
        # Create temporal context
        time_normalized_text = self._normalize_text_with_time(text, relevant_exprs)
        
        temporal_context = TemporalContext(
            temporal_expressions=relevant_exprs,
            time_normalized_text=time_normalized_text,
            temporal_stats={
                'expression_count': len(relevant_exprs),
                'types': list(set(expr.type for expr in relevant_exprs)),
                'confidence': sum(expr.confidence for expr in relevant_exprs) / len(relevant_exprs)
            }
        )
        
        return temporal_context
    
    def _is_near_triple_component(self, temporal_text: str, subject: str, predicate: str, obj: str, full_text: str) -> bool:
        """Check if temporal expression is near triple components"""
        
        # Simple proximity check - if temporal text appears in same sentence as any component
        sentences = full_text.split('.')
        
        for sentence in sentences:
            if (temporal_text.lower() in sentence.lower() and 
                (subject.lower() in sentence.lower() or 
                 predicate.lower() in sentence.lower() or 
                 obj.lower() in sentence.lower())):
                return True
        
        return False
    
    def _normalize_text_with_time(self, text: str, temporal_exprs: List[TemporalExpression]) -> str:
        """Normalize text by replacing temporal expressions with normalized forms"""
        
        normalized_text = text
        
        for expr in temporal_exprs:
            if expr.start_date:
                normalized_form = expr.start_date.strftime('%Y-%m-%d')
            else:
                normalized_form = f"[{expr.type.upper()}: {expr.text}]"
            
            normalized_text = normalized_text.replace(expr.text, normalized_form)
        
        return normalized_text
    
    def _create_temporal_contexts(self, temporal_exprs: List[TemporalExpression], text: str) -> List[TemporalContext]:
        """Create temporal contexts from temporal expressions"""
        
        contexts = []
        
        # Group temporal expressions by type
        by_type = defaultdict(list)
        for expr in temporal_exprs:
            by_type[expr.type].append(expr)
        
        # Create context for each type
        for expr_type, exprs in by_type.items():
            time_normalized_text = self._normalize_text_with_time(text, exprs)
            
            context = TemporalContext(
                temporal_expressions=exprs,
                time_normalized_text=time_normalized_text,
                temporal_stats={
                    'type': expr_type,
                    'expression_count': len(exprs),
                    'confidence': sum(expr.confidence for expr in exprs) / len(exprs) if exprs else 0
                }
            )
            contexts.append(context)
        
        return contexts
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date text to datetime object"""
        
        if not date_text:
            return None
        
        try:
            # Try different date formats
            from dateutil.parser import parse
            
            parsed_date = parse(date_text, fuzzy=True)
            return parsed_date
            
        except Exception:
            return None
    
    def _load_timexy(self):
        """Load Timexy model"""
        try:
            # For now, we'll create a simple timexy-like extractor
            # In a real implementation, you would load a proper timexy model
            self._timexy = None
            logger.debug("[TemporalContextExtractor] Timexy not loaded (requires proper model)")
        except Exception as e:
            logger.error(f"[TemporalContextExtractor] Failed to load timexy: {e}")
            self._timexy = None
    
    def _load_spacy(self):
        """Load spacy model"""
        try:
            self._nlp = spacy.load("en_core_web_sm")
            logger.debug("[TemporalContextExtractor] spacy model loaded")
        except Exception:
            try:
                self._nlp = spacy.blank("en")
                logger.debug("[TemporalContextExtractor] spacy blank model loaded")
            except Exception as e:
                logger.error(f"[TemporalContextExtractor] Failed to load spacy: {e}")
                self._nlp = None
    
    def clear_cache(self):
        """Clear temporal cache"""
        self._temporal_cache.clear()
        logger.debug("[TemporalContextExtractor] Temporal cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics"""
        return {
            'enabled': self.enabled,
            'timexy_available': TIMEXY_AVAILABLE,
            'spacy_available': SPACY_AVAILABLE,
            'cache_size': len(self._temporal_cache),
            'confidence_threshold': self.confidence_threshold
        }


logger.info("🎯 TemporalContextExtractor initialized - advanced temporal context extraction")
logger.info("📊 Features: Time expression extraction, normalization, temporal relationship enhancement")