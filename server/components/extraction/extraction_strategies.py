"""
Extraction Strategy Module

Unified extraction strategy pattern for consolidating multiple extractors.
Provides consistent interface and pluggable architecture for extraction methods.

Author: SOLID Refactoring
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import time
import re
from loguru import logger

from components.memory.memory_interfaces import IExtractionStrategy

class ExtractionStrategyBase(IExtractionStrategy):
    """Base class for all extraction strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strategy_name = self.__class__.__name__.replace('ExtractionStrategy', '').lower()
        self.enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 1.0)
        self.max_triples = self.config.get('max_triples', 50)
        
        # Performance tracking
        self.extraction_count = 0
        self.total_extraction_time = 0
        self.last_extraction_time = 0
    
    @abstractmethod
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples from text using this strategy."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy is available and ready to use."""
        pass
    
    def get_strategy_name(self) -> str:
        """Get the name of this extraction strategy."""
        return self.strategy_name
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get configuration for this strategy."""
        return {
            'name': self.strategy_name,
            'enabled': self.enabled,
            'priority': self.priority,
            'max_triples': self.max_triples,
            'available': self.is_available(),
            'extraction_count': self.extraction_count,
            'avg_extraction_time': self.total_extraction_time / max(self.extraction_count, 1)
        }
    
    def record_extraction(self, extraction_time_ms: int) -> None:
        """Record extraction performance metrics."""
        self.extraction_count += 1
        self.total_extraction_time += extraction_time_ms
        self.last_extraction_time = extraction_time_ms
    
    def filter_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Basic triple filtering."""
        filtered = []
        
        for subject, relation, object_ in triples:
            # Skip empty components
            if not subject or not relation or not object_:
                continue
            
            # Skip overly long components
            if (len(subject) > 100 or len(relation) > 50 or len(object_) > 100):
                continue
            
            # Skip reflexive triples
            if subject.lower() == object_.lower():
                continue
            
            # Limit number of triples
            if len(filtered) >= self.max_triples:
                break
            
            filtered.append((subject, relation, object_))
        
        return filtered


# Strategy registry - keep only Enhanced Level3 for performance
EXTRACTION_STRATEGIES = {
    'enhanced_level3': EnhancedLevel3ExtractionStrategy,
}

def create_strategy(strategy_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[IExtractionStrategy]:
    """Create an extraction strategy by name."""
    strategy_class = EXTRACTION_STRATEGIES.get(strategy_name)
    if strategy_class:
        try:
            return strategy_class(config)
        except Exception as e:
            logger.error(f"Failed to create strategy '{strategy_name}': {e}")
    return None

def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    return list(EXTRACTION_STRATEGIES.keys())

def get_strategy_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all strategies."""
    info = {}
    for name, strategy_class in EXTRACTION_STRATEGIES.items():
        try:
            strategy = strategy_class()
            info[name] = strategy.get_strategy_config()
        except Exception as e:
            info[name] = {'name': name, 'error': str(e)}
    return info


# -----------------------
# ASI YAML-based strategies
# -----------------------

class _AsiYamlAdapter:
    """Lightweight adapter around ULTRAGROKSpacyV821Processor to emit (s, r, o) triples."""

    def __init__(self, yaml_path: Optional[str] = None, spacy_model: Optional[str] = None):
        import os
        from pathlib import Path
        # Resolve server root for relative YAMLs
        base_dir = Path(__file__).resolve().parents[2]

        # Pick defaults if not provided
        self.yaml_path = yaml_path or str(base_dir / 'ULTRAGROK_V8.2.1_SPACY.yaml')
        self.spacy_model = spacy_model or os.getenv('ASI_SPACY_MODEL', 'en_core_web_rtf')

        # Ensure server directory is on sys.path for importing asi1_processor
        try:
            import sys
            server_dir = str(base_dir)
            if server_dir not in sys.path:
                sys.path.insert(0, server_dir)
        except Exception:
            pass

        # Import processor
        try:
            from asi1_processor import ULTRAGROKSpacyV821Processor  # type: ignore
        except Exception as e:
            ULTRAGROKSpacyV821Processor = None  # type: ignore
            logger.warning(f"ASI YAML processor unavailable: {e}")

        if ULTRAGROKSpacyV821Processor is None:
            self.processor = None
        else:
            try:
                self.processor = ULTRAGROKSpacyV821Processor(self.yaml_path, self.spacy_model)
            except Exception as e:
                logger.warning(f"Failed to initialize ASI YAML processor ({self.yaml_path}): {e}")
                self.processor = None

    def is_available(self) -> bool:
        return self.processor is not None

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        if not self.processor or not text:
            return []
        try:
            result = self.processor.process_spacy_semantics(text)
            triples = result.get('triples') or []
            out: List[Tuple[str, str, str]] = []
            for t in triples:
                # Accept both dataclass-like and dict-like
                subj = getattr(t, 'subj', None) if hasattr(t, 'subj') else t.get('subj') if isinstance(t, dict) else None
                pred = getattr(t, 'pred', None) if hasattr(t, 'pred') else t.get('pred') if isinstance(t, dict) else None
                obj = getattr(t, 'obj', None) if hasattr(t, 'obj') else t.get('obj') if isinstance(t, dict) else None
                if subj and pred is not None:
                    out.append((str(subj), str(pred), str(obj or '')))
            return out
        except Exception as e:
            logger.warning(f"ASI YAML extraction failed: {e}")
            return []


class ASI1ExtractionStrategy(ExtractionStrategyBase):
    """ASI1 strategy using the ASI1 YAML rule set as the primary extractor."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[2]
        yaml_override = (self.config or {}).get('yaml_path')
        default_yaml = base_dir / 'ASI1_8_2_3.yaml'
        self.adapter = _AsiYamlAdapter(yaml_path=str(yaml_override or default_yaml))

    def extract(self, text: str, lang: str = 'en') -> List[Tuple[str, str, str]]:
        if not self.is_available():
            return []
        start = time.time()
        triples = self.adapter.extract(text)
        triples = self.filter_triples(triples)
        self.record_extraction(int((time.time() - start) * 1000))
        return triples

    def is_available(self) -> bool:
        return self.enabled and self.adapter.is_available()


class ASI2ExtractionStrategy(ExtractionStrategyBase):
    """ASI2 strategy using the ULTRAGROK V8.2.1 spaCy-compatible YAML or ALT_REFINED."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[2]
        yaml_override = (self.config or {}).get('yaml_path')
        # Prefer ALT_REFINED if present, else fall back to ULTRAGROK V8.2.1
        alt_refined = base_dir / 'ASI_ALT_REFINED.yaml'
        default_yaml = base_dir / 'ULTRAGROK_V8.2.1_SPACY.yaml'
        chosen = str(yaml_override or (alt_refined if alt_refined.exists() else default_yaml))
        self.adapter = _AsiYamlAdapter(yaml_path=chosen)

    def extract(self, text: str, lang: str = 'en') -> List[Tuple[str, str, str]]:
        if not self.is_available():
            return []
        start = time.time()
        triples = self.adapter.extract(text)
        triples = self.filter_triples(triples)
        self.record_extraction(int((time.time() - start) * 1000))
        return triples

    def is_available(self) -> bool:
        return self.enabled and self.adapter.is_available()


class EnhancedLevel3ExtractionStrategy(ExtractionStrategyBase):
    """Enhanced Level3 strategy - <1ms quality extraction champion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self.nlp = None
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize the Enhanced Level3 Quality extractor."""
        try:
            import sys
            import os
            # Add server directory to path if needed
            server_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if server_dir not in sys.path:
                sys.path.insert(0, server_dir)

            from components.extraction.enhanced_level3_extractor import QualityExtractor
            from services.nlp_cache import get_en_model_from_env

            # Use centralized cached spaCy model
            self.nlp = get_en_model_from_env()

            # Thresholds/targets tuning
            try:
                ent_thr = float(os.getenv('ENHANCED_LEVEL3_ENTITY_CONF', '0.70'))
            except Exception:
                ent_thr = 0.70
            try:
                rel_thr = float(os.getenv('ENHANCED_LEVEL3_RELATION_CONF', '0.65'))
            except Exception:
                rel_thr = 0.65
            try:
                tgt_ent = int(os.getenv('ENHANCED_LEVEL3_TARGET_ENT', '50'))
            except Exception:
                tgt_ent = 50
            try:
                tgt_rel = int(os.getenv('ENHANCED_LEVEL3_TARGET_REL', '30'))
            except Exception:
                tgt_rel = 30

            self.extractor = QualityExtractor(entity_threshold=ent_thr,
                                              relation_threshold=rel_thr,
                                              target_entities=tgt_ent,
                                              target_relations=tgt_rel)
            logger.info("Enhanced Level3 Quality extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Enhanced Level3 extractor: {e}")
            self.extractor = None
            self.nlp = None

    def extract(self, text: str, lang: str = 'en') -> List[Tuple[str, str, str]]:
        """Extract high-quality triples using Enhanced Level3."""
        if not self.is_available():
            return []

        start = time.time()

        try:
            # Process with spaCy
            doc = self.nlp(text)

            # Extract with quality filtering
            result = self.extractor.extract_quality_kg(doc)

            # Convert to triples format
            triples = []
            # Use extractor's configured relation threshold for output gating
            rel_out_thr = getattr(self.extractor, 'RELATION_CONFIDENCE_THRESHOLD', 0.65)
            # Store last props map for downstream consumers (confidence, verb, prep)
            self.last_props_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for relation in result.get('relations', []):
                if relation.confidence >= rel_out_thr:
                    s = relation.subject
                    p = relation.predicate
                    o = relation.object
                    triples.append((s, p, o))
                    verb = p.split('_', 1)[0] if '_' in p else p
                    prep = p.split('_', 1)[1] if '_' in p else ''
                    try:
                        self.last_props_map[(s, p, o)] = {
                            'confidence': float(getattr(relation, 'confidence', 0.0) or 0.0),
                            'verb': verb,
                            'prep': prep,
                            'normalized_relation': p,
                        }
                    except Exception:
                        pass

            # Filter and record
            filtered_triples = self.filter_triples(triples)
            self.record_extraction(int((time.time() - start) * 1000))

            return filtered_triples

        except Exception as e:
            logger.error(f"Enhanced Level3 extraction failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if Enhanced Level3 extractor is available."""
        return self.enabled and self.extractor is not None and self.nlp is not None
