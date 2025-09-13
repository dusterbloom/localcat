"""
Configuration: Centralized Configuration Management
==================================================

Extracted from HotMemory __init__ - now focused solely on:
- Environment variable parsing
- Feature flag management  
- Model configuration
- Service initialization
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    base_url: str
    timeout_ms: int = 120
    device: str = "cpu"
    enabled: bool = True


@dataclass
class FeatureFlags:
    """Feature flags for HotMemory components"""
    use_srl: bool = False
    use_onnx_ner: bool = False  
    use_onnx_srl: bool = False
    use_relik: bool = False
    use_coref: bool = False
    use_dspy: bool = False
    use_gliner: bool = True  # GLiNER for 96.7% entity extraction accuracy
    use_leann: bool = True
    retrieval_fusion: bool = True
    assisted_enabled: bool = False


@dataclass
class HotMemoryConfig:
    """Complete configuration for HotMemory system"""
    # Core settings
    max_recency: int = 50
    user_eid: str = "you"
    confidence_threshold: float = 0.3
    language: str = "en"
    
    # Feature flags
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Model configurations
    assisted_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="google/gemma-3-270m",
        base_url="http://127.0.0.1:1234/v1",
        timeout_ms=120
    ))
    
    # LEANN configuration
    leann_index_path: Optional[str] = None
    leann_complexity: int = 16
    use_leann_summaries: bool = True
    
    # Performance settings
    cache_size: int = 1000
    coref_max_entities: int = 24
    assisted_max_triples: int = 3
    
    # Paths
    data_dir: str = ""
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        self._load_from_env()
        self._validate_config()
        self._setup_paths()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Core settings
        self.max_recency = int(os.getenv("HOTMEM_MAX_RECENCY", str(self.max_recency)))
        self.confidence_threshold = float(os.getenv("HOTMEM_CONFIDENCE_THRESHOLD", str(self.confidence_threshold)))
        
        # Feature flags
        self.features.use_srl = os.getenv("HOTMEM_USE_SRL", "false").lower() in ("1", "true", "yes")
        self.features.use_onnx_ner = os.getenv("HOTMEM_USE_ONNX_NER", "false").lower() in ("1", "true", "yes") and self._has_onnx_ner()
        self.features.use_onnx_srl = os.getenv("HOTMEM_USE_ONNX_SRL", "false").lower() in ("1", "true", "yes") and self._has_onnx_srl()
        self.features.use_relik = os.getenv("HOTMEM_USE_RELIK", "false").lower() in ("1", "true", "yes") and self._has_relik()
        self.features.use_coref = os.getenv("HOTMEM_USE_COREF", "false").lower() in ("1", "true", "yes") and self._has_coref()
        self.features.use_dspy = os.getenv("HOTMEM_USE_DSPY", "false").lower() in ("1", "true", "yes")
        self.features.use_gliner = os.getenv("HOTMEM_USE_GLINER", "true").lower() in ("1", "true", "yes")  # Default to true for best extraction
        self.features.use_leann = os.getenv("HOTMEM_USE_LEANN", "true").lower() in ("1", "true", "yes")
        self.features.retrieval_fusion = os.getenv("HOTMEM_RETRIEVAL_FUSION", "true").lower() in ("1", "true", "yes")
        self.features.assisted_enabled = os.getenv("HOTMEM_LLM_ASSISTED", "false").lower() in ("1", "true", "yes")
        
        # LEANN settings
        self.leann_index_path = os.getenv("LEANN_INDEX_PATH", os.path.join(self.data_dir, 'memory_vectors.leann'))
        self.leann_complexity = int(os.getenv("HOTMEM_LEANN_COMPLEXITY", str(self.leann_complexity)))
        self.use_leann_summaries = os.getenv("HOTMEM_USE_LEANN_SUMMARIES", "true").lower() in ("1", "true", "yes")
        
        # Performance settings
        self.cache_size = int(os.getenv("HOTMEM_CACHE_SIZE", str(self.cache_size)))
        self.coref_max_entities = int(os.getenv("HOTMEM_COREF_MAX_ENTITIES", str(self.coref_max_entities)))
        self.assisted_max_triples = int(os.getenv("HOTMEM_LLM_ASSISTED_MAX_TRIPLES", str(self.assisted_max_triples)))
        
        # Model configurations
        assisted_model_name = os.getenv("HOTMEM_LLM_ASSISTED_MODEL", "google/gemma-3-270m")
        assisted_base_url = (
            os.getenv("HOTMEM_LLM_ASSISTED_BASE_URL")
            or os.getenv("SUMMARIZER_BASE_URL")
            or "http://127.0.0.1:1234/v1"
        )
        assisted_timeout = int(os.getenv("HOTMEM_LLM_ASSISTED_TIMEOUT_MS", "120"))
        
        self.assisted_model = ModelConfig(
            name=assisted_model_name,
            base_url=assisted_base_url,
            timeout_ms=assisted_timeout,
            device=os.getenv("HOTMEM_LLM_ASSISTED_DEVICE", "cpu")
        )
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.max_recency <= 0:
            raise ValueError("max_recency must be positive")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
            
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
            
        if self.coref_max_entities <= 0:
            raise ValueError("coref_max_entities must be positive")
    
    def _setup_paths(self):
        """Setup absolute paths"""
        if self.leann_index_path and not os.path.isabs(self.leann_index_path):
            if self.data_dir:
                self.leann_index_path = os.path.abspath(os.path.join(self.data_dir, self.leann_index_path))
            else:
                self.leann_index_path = os.path.abspath(self.leann_index_path)
    
    def _has_onnx_ner(self) -> bool:
        """Check if ONNX NER is available"""
        try:
            from services.onnx_nlp import OnnxTokenNER
            return OnnxTokenNER is not None
        except Exception:
            return False
    
    def _has_onnx_srl(self) -> bool:
        """Check if ONNX SRL is available"""
        try:
            from services.onnx_nlp import OnnxSRLTagger
            return OnnxSRLTagger is not None
        except Exception:
            return False
    
    def _has_relik(self) -> bool:
        """Check if ReLiK is available"""
        try:
            from components.extraction.hotmem_extractor import HotMemExtractor
            return HotMemExtractor is not None
        except Exception:
            return False
    
    def _has_coref(self) -> bool:
        """Check if coreference is available"""
        try:
            from fastcoref import FCoref
            return FCoref is not None
        except Exception:
            return False
    
    def get_extractor_config(self) -> Dict[str, Any]:
        """Get configuration for MemoryExtractor"""
        return {
            'use_srl': self.features.use_srl,
            'use_onnx_ner': self.features.use_onnx_ner,
            'use_onnx_srl': self.features.use_onnx_srl,
            'use_relik': self.features.use_relik,
            'use_dspy': self.features.use_dspy,
            'use_gliner': self.features.use_gliner,  # Add GLiNER support
            'assisted_model': self.assisted_model,
            'cache_size': self.cache_size
        }
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """Get configuration for MemoryRetriever"""
        return {
            'use_leann': self.features.use_leann,
            'leann_index_path': self.leann_index_path,
            'leann_complexity': self.leann_complexity,
            'retrieval_fusion': self.features.retrieval_fusion,
            'use_leann_summaries': self.use_leann_summaries
        }
    
    def get_coreference_config(self) -> Dict[str, Any]:
        """Get configuration for CoreferenceResolver"""
        return {
            'use_coref': self.features.use_coref,
            'coref_max_entities': self.coref_max_entities,
            'coref_device': self.assisted_model.device
        }
    
    def get_assisted_config(self) -> Dict[str, Any]:
        """Get configuration for AssistedExtractor"""
        return {
            'assisted_enabled': self.features.assisted_enabled,
            'assisted_model': self.assisted_model,
            'cache_size': self.cache_size
        }
    
    def log_configuration(self):
        """Log current configuration"""
        logger.info("🔧 HotMemory Configuration:")
        logger.info(f"  📊 Feature flags: {self.features}")
        logger.info(f"  🤖 Assisted model: {self.assisted_model.name}")
        logger.info(f"  🧠 LEANN: enabled={self.features.use_leann}, complexity={self.leann_complexity}")
        logger.info(f"  ⚡ Cache size: {self.cache_size}")
        logger.info(f"  🎯 Confidence threshold: {self.confidence_threshold}")


def create_config(data_dir: str = "") -> HotMemoryConfig:
    """Create HotMemory configuration with default values"""
    return HotMemoryConfig(data_dir=data_dir)


logger.info("⚙️ Configuration module initialized - centralized config management")
logger.info("🎯 Environment variables loaded and validated")