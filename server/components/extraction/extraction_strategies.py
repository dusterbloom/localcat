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

class HotMemExtractionStrategy(ExtractionStrategyBase):
    """HotMem-based extraction strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the HotMem extractor."""
        try:
            from components.extraction.hotmem_extractor import HotMemExtractor
            model_id = self.config.get('model_id', 'relik-ie/relik-relation-extraction-small')
            device = self.config.get('device', 'cpu')
            self.extractor = HotMemExtractor(model_id=model_id, device=device)
            logger.info(f"HotMem extractor initialized with model: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize HotMem extractor: {e}")
            self.extractor = None
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using HotMem."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Use HotMem extractor
            if hasattr(self.extractor, 'extract_triples'):
                triples = self.extractor.extract_triples(text)
            elif hasattr(self.extractor, 'extract'):
                triples = self.extractor.extract(text)
            else:
                logger.warning("HotMem extractor has no extraction method")
                return []
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"HotMem extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if HotMem extractor is available."""
        return self.extractor is not None and self.enabled

class UDExtractionStrategy(ExtractionStrategyBase):
    """Universal Dependencies extraction strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the UD extractor."""
        try:
            from components.extraction.improved_ud_extractor import ImprovedUDExtractor
            self.extractor = ImprovedUDExtractor(config=self.config)
            logger.info("UD extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize UD extractor: {e}")
            self.extractor = None
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using Universal Dependencies."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Use UD extractor
            triples = self.extractor.extract(text, lang)
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"UD extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if UD extractor is available."""
        return self.extractor is not None and self.enabled

class HybridExtractionStrategy(ExtractionStrategyBase):
    """Hybrid spaCy-LLM extraction strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the hybrid extractor."""
        try:
            from components.extraction.hybrid_spacy_llm_extractor import HybridRelationExtractor
            self.extractor = HybridRelationExtractor(config=self.config)
            logger.info("Hybrid extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid extractor: {e}")
            self.extractor = None
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using hybrid spaCy-LLM approach."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Use hybrid extractor
            triples = self.extractor.extract(text, lang)
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if hybrid extractor is available."""
        return self.extractor is not None and self.enabled

class LightweightExtractionStrategy(ExtractionStrategyBase):
    """Lightweight relation extraction strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the lightweight extractor."""
        try:
            from components.extraction.lightweight_relation_extractor import LightweightRelationExtractor
            self.extractor = LightweightRelationExtractor(config=self.config)
            logger.info("Lightweight extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize lightweight extractor: {e}")
            self.extractor = None
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using lightweight approach."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Use lightweight extractor
            triples = self.extractor.extract(text, lang)
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"Lightweight extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if lightweight extractor is available."""
        return self.extractor is not None and self.enabled

class MultilingualExtractionStrategy(ExtractionStrategyBase):
    """Multilingual graph extraction strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the multilingual extractor."""
        try:
            from components.extraction.multilingual_graph_extractor import MultilingualGraphExtractor
            self.extractor = MultilingualGraphExtractor(config=self.config)
            logger.info("Multilingual extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize multilingual extractor: {e}")
            self.extractor = None
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using multilingual approach."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Use multilingual extractor
            triples = self.extractor.extract(text, lang)
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"Multilingual extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if multilingual extractor is available."""
        return self.extractor is not None and self.enabled

class EnhancedHotMemExtractionStrategy(ExtractionStrategyBase):
    """Enhanced HotMem extraction strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extractor = None
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the enhanced HotMem extractor."""
        try:
            from components.extraction.enhanced_hotmem_extractor import EnhancedHotMemExtractor
            self.extractor = EnhancedHotMemExtractor(config=self.config)
            logger.info("Enhanced HotMem extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced HotMem extractor: {e}")
            self.extractor = None
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using enhanced HotMem approach."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Use enhanced HotMem extractor
            triples = self.extractor.extract(text, lang)
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"Enhanced HotMem extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if enhanced HotMem extractor is available."""
        return self.extractor is not None and self.enabled

class PatternBasedExtractionStrategy(ExtractionStrategyBase):
    """Pattern-based extraction strategy for simple patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Initialize extraction patterns."""
        patterns = []
        
        # Name patterns
        patterns.append((re.compile(r'\b(?:I am|I\'m|My name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE), 'name'))
        patterns.append((re.compile(r'\b(?:call me|you can call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE), 'name'))
        
        # Work patterns
        patterns.append((re.compile(r'\b(I work|work|working)\s+(?:at|for|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE), 'works_at'))
        patterns.append((re.compile(r'\b(I\'m|I am)\s+(?:a|an)\s+([A-Za-z\s]+)\b', re.IGNORECASE), 'is'))
        
        # Location patterns
        patterns.append((re.compile(r'\b(I live|live|living)\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE), 'lives_in'))
        
        return patterns
    
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Extract triples using pattern matching."""
        if not self.is_available():
            return []
        
        start_time = time.time()
        triples = []
        
        try:
            for pattern, relation in self.patterns:
                matches = pattern.findall(text)
                for match in matches:
                    if isinstance(match, tuple):
                        subject = 'you'
                        object_ = match[0]
                    else:
                        subject = 'you'
                        object_ = match
                    
                    triples.append((subject, relation, object_))
            
            # Filter and return
            filtered_triples = self.filter_triples(triples)
            
            extraction_time = int((time.time() - start_time) * 1000)
            self.record_extraction(extraction_time)
            
            return filtered_triples
            
        except Exception as e:
            logger.error(f"Pattern-based extraction failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if pattern-based extraction is available."""
        return self.enabled and len(self.patterns) > 0

# Strategy registry
EXTRACTION_STRATEGIES = {
    'hotmem': HotMemExtractionStrategy,
    'ud': UDExtractionStrategy,
    'hybrid': HybridExtractionStrategy,
    'lightweight': LightweightExtractionStrategy,
    'multilingual': MultilingualExtractionStrategy,
    'enhanced_hotmem': EnhancedHotMemExtractionStrategy,
    'pattern': PatternBasedExtractionStrategy,
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
        self.spacy_model = spacy_model or os.getenv('ASI_SPACY_MODEL', 'en_core_web_sm')

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
            from enhanced_level3_extractor import QualityExtractor
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
            for relation in result.get('relations', []):
                if relation.confidence >= rel_out_thr:
                    triples.append((relation.subject, relation.predicate, relation.object))

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
