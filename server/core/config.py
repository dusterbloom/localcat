"""
Centralized configuration for LocalCat server
All configuration values should be defined here
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum

# Get base directory
BASE_DIR = Path(__file__).parent


class LogLevel(Enum):
    """Log levels for the application"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class ServerConfig:
    """Server-related configuration"""
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8765")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    
    # API configuration
    api_prefix: str = "/api/v1"
    api_timeout: float = 30.0
    
    # Static files
    static_dir: str = str(BASE_DIR / "static")
    templates_dir: str = str(BASE_DIR / "templates")


@dataclass
class DatabaseConfig:
    """Database configuration"""
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", str(BASE_DIR / "memory.db")))
    lmdb_dir: str = field(default_factory=lambda: os.getenv("LMDB_DIR", str(BASE_DIR / "lmdb")))
    
    # Connection settings
    max_connections: int = 10
    connection_timeout: float = 30.0
    connection_retry_attempts: int = 3
    
    # Performance settings
    enable_connection_pooling: bool = True
    pool_size: int = 5
    max_overflow: int = 10
    
    # Backup settings
    enable_auto_backup: bool = field(default_factory=lambda: os.getenv("ENABLE_AUTO_BACKUP", "true").lower() == "true")
    backup_interval_hours: int = 24
    backup_retention_days: int = 7


@dataclass
class MemoryConfig:
    """Memory and context configuration"""
    # Memory settings
    max_memory_facts: int = 10000
    memory_retention_days: int = 30
    
    # Context settings
    max_context_items: int = 50
    max_context_tokens: int = 2000
    context_ttl: int = 3600  # 1 hour
    
    # HotMem settings
    enable_hotmem: bool = field(default_factory=lambda: os.getenv("ENABLE_HOTMEM", "true").lower() == "true")
    hotmem_max_facts_per_injection: int = 3
    hotmem_temporal_alpha: float = 0.15
    hotmem_temporal_beta: float = 0.60
    
    # Session settings
    max_session_duration: int = 3600  # 1 hour
    max_idle_time: int = 1800  # 30 minutes
    max_sessions_per_user: int = 5
    
    # Cross-session features
    enable_cross_session_context: bool = True
    enable_session_persistence: bool = True


@dataclass
class ExtractionConfig:
    """Text extraction configuration"""
    # Strategy settings
    default_strategy: str = field(default_factory=lambda: os.getenv("DEFAULT_EXTRACTION_STRATEGY", "enhanced_hotmem"))
    fallback_strategy: str = field(default_factory=lambda: os.getenv("FALLBACK_EXTRACTION_STRATEGY", "lightweight"))
    available_strategies: List[str] = field(default_factory=lambda: [
        "enhanced_hotmem", "lightweight", "hybrid_spacy_llm", "multilingual_graph"
    ])
    
    # Quality settings
    min_confidence_threshold: float = 0.6
    min_overall_quality_threshold: float = 0.5
    enable_correction: bool = True
    enable_validation: bool = True
    
    # Performance settings
    max_extraction_time: float = 2.0
    enable_parallel_extraction: bool = True
    max_parallel_strategies: int = 3
    
    # Entity extraction
    min_entity_length: int = 2
    max_entity_length: int = 50
    required_entity_labels: List[str] = field(default_factory=lambda: ["PERSON", "ORG", "GPE", "PRODUCT"])


@dataclass
class PipelineConfig:
    """Pipeline processing configuration"""
    # Pipeline settings
    enable_memory_processor: bool = True
    enable_extraction_processor: bool = True
    enable_quality_processor: bool = True
    enable_context_processor: bool = True
    
    # Processing settings
    max_pipeline_latency: float = 2.0  # seconds
    enable_parallel_processing: bool = True
    enable_metrics: bool = True
    
    # Quality settings
    quality_check_interval: int = 100  # Check quality every N operations
    quality_alert_threshold: float = 0.3  # Alert if quality drops below this
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    # Sample rates
    tts_sample_rate: int = 24000
    stt_sample_rate: int = 16000
    
    # VAD parameters
    vad_stop_secs: float = 0.4
    vad_start_secs: float = 0.2
    
    # TTS settings
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", "mlx-community/Kokoro-82M-bf16"))
    tts_speed: float = 1.0
    tts_streaming_delay: float = 0.0001
    tts_max_workers: int = 1
    
    # STT settings
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "LARGE_V3_TURBO_Q4"))
    stt_language: str = "en"
    
    # Audio processing
    audio_normalization_factor: float = 32768.0
    enable_noise_reduction: bool = True


@dataclass
class ModelConfig:
    """AI model configuration"""
    # LLM settings
    llm_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "google/gemma-3-12b"))
    llm_max_tokens: int = 8192
    llm_context_length: int = field(default_factory=lambda: int(os.getenv("LLM_CONTEXT_LENGTH", "32768")))
    llm_temperature: float = 0.7
    
    # Embedding settings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    embedding_dimension: int = 384
    
    # Smart turn management
    enable_smart_turn: bool = field(default_factory=lambda: os.getenv("ENABLE_SMART_TURN", "true").lower() == "true")
    smart_turn_model: str = "pipecat-ai/smart-turn-v2"


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    # Metrics settings
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    metrics_retention_hours: int = 24
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    health_check_timeout: float = 5.0
    
    # Logging
    log_level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "INFO")))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))
    max_log_size_mb: int = 100
    log_backup_count: int = 5
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    performance_sample_rate: float = 0.1  # Sample 10% of operations
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "latency_ms": 1000.0,
        "error_rate": 0.05,
        "memory_usage_mb": 1024.0
    })


@dataclass
class SecurityConfig:
    """Security configuration"""
    # API security
    api_key_required: bool = field(default_factory=lambda: os.getenv("API_KEY_REQUIRED", "false").lower() == "true")
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # CORS
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Data protection
    enable_data_encryption: bool = field(default_factory=lambda: os.getenv("ENABLE_DATA_ENCRYPTION", "false").lower() == "true")
    encryption_key: Optional[str] = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY"))


@dataclass
class DevelopmentConfig:
    """Development-specific configuration"""
    # Debug settings
    enable_debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    enable_hot_reload: bool = field(default_factory=lambda: os.getenv("HOT_RELOAD", "false").lower() == "true")
    
    # Testing
    enable_test_mode: bool = field(default_factory=lambda: os.getenv("TEST_MODE", "false").lower() == "true")
    test_data_dir: str = str(BASE_DIR / "tests" / "data")
    
    # Development tools
    enable_dev_tools: bool = True
    dev_tools_port: int = 8080
    enable_profiling: bool = field(default_factory=lambda: os.getenv("ENABLE_PROFILING", "false").lower() == "true")
    
    # Code quality
    enable_linting: bool = True
    enable_type_checking: bool = True
    max_line_length: int = 100


@dataclass
class Config:
    """Main configuration class combining all settings"""
    # Environment
    environment: EnvironmentType = field(default_factory=lambda: EnvironmentType(os.getenv("ENVIRONMENT", "development")))
    base_dir: Path = BASE_DIR
    
    # Component configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    
    # Validation
    validate_on_init: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.validate_on_init:
            self.validate()
    
    def validate(self):
        """Validate configuration values"""
        errors = []
        
        # Validate server config
        if not (1 <= self.server.port <= 65535):
            errors.append("Server port must be between 1 and 65535")
        
        # Validate database paths
        if not self.database.sqlite_path:
            errors.append("SQLite path cannot be empty")
        
        if not self.database.lmdb_dir:
            errors.append("LMDB directory cannot be empty")
        
        # Validate model URLs
        if not self.model.llm_base_url:
            errors.append("LLM base URL cannot be empty")
        
        # Validate thresholds
        if not (0.0 <= self.extraction.min_confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.extraction.min_overall_quality_threshold <= 1.0):
            errors.append("Quality threshold must be between 0.0 and 1.0")
        
        # Validate audio settings
        if self.audio.tts_sample_rate <= 0:
            errors.append("TTS sample rate must be positive")
        
        if self.audio.stt_sample_rate <= 0:
            errors.append("STT sample rate must be positive")
        
        # Validate security settings
        if self.security.api_key_required and not self.security.api_key:
            errors.append("API key is required when API_KEY_REQUIRED is true")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def get_for_environment(self, env: EnvironmentType) -> 'Config':
        """Get configuration for specific environment"""
        config = Config()
        config.environment = env
        
        # Override settings based on environment
        if env == EnvironmentType.PRODUCTION:
            config.server.debug = False
            config.monitoring.log_level = LogLevel.WARNING
            config.development.enable_debug_mode = False
            config.development.enable_hot_reload = False
            config.security.enable_rate_limiting = True
        elif env == EnvironmentType.TESTING:
            config.database.sqlite_path = str(self.base_dir / "test_memory.db")
            config.database.lmdb_dir = str(self.base_dir / "test_lmdb")
            config.monitoring.log_level = LogLevel.DEBUG
            config.development.enable_test_mode = True
        elif env == EnvironmentType.DEVELOPMENT:
            config.server.debug = True
            config.monitoring.log_level = LogLevel.DEBUG
            config.development.enable_debug_mode = True
            config.development.enable_hot_reload = True
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (ServerConfig, DatabaseConfig, MemoryConfig, 
                                 ExtractionConfig, PipelineConfig, AudioConfig,
                                 ModelConfig, MonitoringConfig, SecurityConfig, 
                                 DevelopmentConfig)):
                result[key] = value.__dict__
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        config = cls()
        
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(getattr(config, key), (ServerConfig, DatabaseConfig, MemoryConfig, 
                                                     ExtractionConfig, PipelineConfig, AudioConfig,
                                                     ModelConfig, MonitoringConfig, SecurityConfig, 
                                                     DevelopmentConfig)):
                    # Update nested config
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration with environment variables applied"""
        effective = self.to_dict()
        
        # Apply environment variable overrides
        env_overrides = {
            'SERVER_HOST': ('server', 'host'),
            'SERVER_PORT': ('server', 'port'),
            'DEBUG': ('server', 'debug'),
            'LOG_LEVEL': ('monitoring', 'log_level'),
            'ENABLE_HOTMEM': ('memory', 'enable_hotmem'),
            'DEFAULT_EXTRACTION_STRATEGY': ('extraction', 'default_strategy'),
            'TTS_MODEL': ('audio', 'tts_model'),
            'STT_MODEL': ('audio', 'stt_model'),
            'LLM_MODEL': ('model', 'llm_model'),
            'OPENAI_BASE_URL': ('model', 'llm_base_url'),
            'ENVIRONMENT': ('environment',),
        }
        
        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the correct config section
                current = effective
                for section in config_path[:-1]:
                    if section in current:
                        current = current[section]
                
                # Set the value
                final_key = config_path[-1]
                if isinstance(env_value, str):
                    # Type conversion
                    if final_key in ['port', 'max_connections', 'pool_size', 'max_overflow', 
                                   'backup_interval_hours', 'backup_retention_days', 
                                   'max_memory_facts', 'memory_retention_days', 
                                   'max_context_items', 'max_context_tokens', 
                                   'context_ttl', 'max_session_duration', 'max_idle_time',
                                   'max_sessions_per_user', 'max_extraction_time', 
                                   'max_parallel_strategies', 'quality_check_interval',
                                   'cache_ttl', 'max_cache_size', 'tts_sample_rate', 
                                   'stt_sample_rate', 'tts_max_workers', 'llm_max_tokens',
                                   'llm_context_length', 'embedding_dimension',
                                   'metrics_interval', 'metrics_retention_hours',
                                   'health_check_interval', 'max_log_size_mb', 
                                   'log_backup_count', 'rate_limit_requests', 
                                   'rate_limit_window', 'dev_tools_port', 'max_line_length']:
                        current[final_key] = int(env_value)
                    elif final_key in ['vad_stop_secs', 'vad_start_secs', 'tts_speed', 
                                     'tts_streaming_delay', 'audio_normalization_factor',
                                     'max_pipeline_latency', 'performance_sample_rate',
                                     'health_check_timeout', 'llm_temperature']:
                        current[final_key] = float(env_value)
                    elif env_value.lower() in ['true', 'false']:
                        current[final_key] = env_value.lower() == 'true'
                    elif final_key == 'log_level':
                        current[final_key] = LogLevel(env_value.upper())
                    elif final_key == 'environment':
                        current[final_key] = EnvironmentType(env_value.lower())
                    else:
                        current[final_key] = env_value
        
        return effective


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """Set the global configuration instance"""
    global _config
    _config = config


def create_config_for_environment(env: Union[str, EnvironmentType]) -> Config:
    """Create configuration for specific environment"""
    if isinstance(env, str):
        env = EnvironmentType(env.lower())
    
    base_config = get_config()
    return base_config.get_for_environment(env)


def load_config_from_file(file_path: str) -> Config:
    """Load configuration from file"""
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return Config.from_dict(data)


def save_config_to_file(config: Config, file_path: str):
    """Save configuration to file"""
    import json
    
    with open(file_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)