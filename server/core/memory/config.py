"""
Type-safe Configuration Management for Memory System

Centralized configuration following the Single Responsibility Principle.
Provides type safety, validation, and environment variable integration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from loguru import logger


@dataclass
class CoreferenceConfig:
    """Configuration for coreference resolution functionality."""

    enabled: bool = False
    timeout_ms: int = 50
    min_text_length: int = 10
    model_name: str = "en_core_web_sm"
    fallback_enabled: bool = True
    lang: str = "en"

    def __post_init__(self):
        """Validate configuration values."""
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")

    @classmethod
    def from_env(cls) -> 'CoreferenceConfig':
        """Create configuration from environment variables."""
        return cls(
            enabled=_get_bool_env("MEMORY_COREFERENCE_ENABLED", False),
            timeout_ms=_get_int_env("MEMORY_COREFERENCE_TIMEOUT_MS", 50),
            min_text_length=_get_int_env("MEMORY_COREFERENCE_MIN_LENGTH", 10),
            model_name=os.getenv("MEMORY_COREFERENCE_MODEL", "en_core_web_sm"),
            fallback_enabled=_get_bool_env("MEMORY_COREFERENCE_FALLBACK", True),
            lang=os.getenv("MEMORY_COREFERENCE_LANG", "en")
        )


@dataclass
class NLPConfig:
    """Configuration for NLP model management."""

    cache_enabled: bool = True
    default_lang: str = "en"
    model_timeout_ms: int = 30000  # 30 seconds for model loading
    max_cached_models: int = 5

    @classmethod
    def from_env(cls) -> 'NLPConfig':
        """Create configuration from environment variables."""
        return cls(
            cache_enabled=_get_bool_env("NLP_CACHE_ENABLED", True),
            default_lang=os.getenv("NLP_DEFAULT_LANG", "en"),
            model_timeout_ms=_get_int_env("NLP_MODEL_TIMEOUT_MS", 30000),
            max_cached_models=_get_int_env("NLP_MAX_CACHED_MODELS", 5)
        )


@dataclass
class ProcessorConfig:
    """Configuration for text processor chains."""

    enabled: bool = True
    max_processors: int = 10
    total_timeout_ms: int = 200  # Total budget for all processors
    metrics_enabled: bool = True
    metrics_max_entries: int = 1000

    @classmethod
    def from_env(cls) -> 'ProcessorConfig':
        """Create configuration from environment variables."""
        return cls(
            enabled=_get_bool_env("MEMORY_PROCESSORS_ENABLED", True),
            max_processors=_get_int_env("MEMORY_MAX_PROCESSORS", 10),
            total_timeout_ms=_get_int_env("MEMORY_PROCESSOR_TIMEOUT_MS", 200),
            metrics_enabled=_get_bool_env("MEMORY_PROCESSOR_METRICS", True),
            metrics_max_entries=_get_int_env("MEMORY_PROCESSOR_METRICS_MAX", 1000)
        )


@dataclass
class MemoryConfig:
    """
    Comprehensive memory system configuration.

    Consolidates all memory-related settings in one place following
    the Single Responsibility Principle for configuration management.
    """

    # Core memory settings (existing)
    enabled: bool = True
    bullets_max: int = 3
    interim_min_words: int = 6
    inject_role: str = "user"
    inject_header: str = "[Memory context]"

    # New modular settings
    coreference: CoreferenceConfig = field(default_factory=CoreferenceConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    processors: ProcessorConfig = field(default_factory=ProcessorConfig)

    # Legacy compatibility
    sources: List[str] = field(default_factory=lambda: ["graph"])
    convo_index_enabled: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.bullets_max < 0:
            raise ValueError("bullets_max must be non-negative")
        if self.interim_min_words < 0:
            raise ValueError("interim_min_words must be non-negative")
        if self.inject_role not in ("user", "system"):
            raise ValueError("inject_role must be 'user' or 'system'")

    @classmethod
    def from_env(cls) -> 'MemoryConfig':
        """
        Create comprehensive configuration from environment variables.

        This method consolidates environment variable parsing and provides
        a single source of truth for memory system configuration.
        """
        # Parse legacy environment variables for backward compatibility
        enabled = _get_bool_env("MEMORY_ENABLED", True)
        bullets_max = _get_int_env("MEMORY_BULLETS_MAX", 3)
        interim_min_words = _get_int_env("MEMORY_INTERIM_MIN_WORDS", 6)
        inject_role = os.getenv("MEMORY_INJECT_ROLE", "user").strip().lower()
        inject_header = os.getenv("MEMORY_INJECT_HEADER", "[Memory context]")

        # Parse list environment variables
        sources = _get_list_env("MEMORY_SOURCES", ["graph"])
        convo_index_enabled = _get_bool_env("MEMORY_CONVO_INDEX", False)

        # Create nested configurations
        coreference_config = CoreferenceConfig.from_env()
        nlp_config = NLPConfig.from_env()
        processor_config = ProcessorConfig.from_env()

        return cls(
            enabled=enabled,
            bullets_max=bullets_max,
            interim_min_words=interim_min_words,
            inject_role=inject_role,
            inject_header=inject_header,
            coreference=coreference_config,
            nlp=nlp_config,
            processors=processor_config,
            sources=sources,
            convo_index_enabled=convo_index_enabled
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "enabled": self.enabled,
            "bullets_max": self.bullets_max,
            "interim_min_words": self.interim_min_words,
            "inject_role": self.inject_role,
            "inject_header": self.inject_header,
            "coreference": {
                "enabled": self.coreference.enabled,
                "timeout_ms": self.coreference.timeout_ms,
                "min_text_length": self.coreference.min_text_length,
                "lang": self.coreference.lang
            },
            "nlp": {
                "cache_enabled": self.nlp.cache_enabled,
                "default_lang": self.nlp.default_lang,
                "max_cached_models": self.nlp.max_cached_models
            },
            "processors": {
                "enabled": self.processors.enabled,
                "max_processors": self.processors.max_processors,
                "total_timeout_ms": self.processors.total_timeout_ms,
                "metrics_enabled": self.processors.metrics_enabled
            },
            "sources": self.sources,
            "convo_index_enabled": self.convo_index_enabled
        }

    def log_configuration(self) -> None:
        """Log current configuration for debugging."""
        config_dict = self.to_dict()
        logger.info(f"Memory system configuration: {config_dict}")


# Utility functions for environment variable parsing

def _get_bool_env(name: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(name, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    elif value in ("0", "false", "no", "off"):
        return False
    else:
        return default


def _get_int_env(name: str, default: int) -> int:
    """Parse integer environment variable with validation."""
    try:
        value = os.getenv(name)
        if value is not None:
            return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {name}: {os.getenv(name)}, using default {default}")
    return default


def _get_list_env(name: str, default: List[str]) -> List[str]:
    """Parse comma-separated list environment variable."""
    value = os.getenv(name)
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return default


# Global configuration instance
_memory_config: Optional[MemoryConfig] = None


def get_memory_config() -> MemoryConfig:
    """
    Get the global memory configuration instance.

    This follows the singleton pattern to ensure consistent configuration
    across the memory system.
    """
    global _memory_config
    if _memory_config is None:
        _memory_config = MemoryConfig.from_env()
        _memory_config.log_configuration()
    return _memory_config


def reload_memory_config() -> MemoryConfig:
    """
    Reload configuration from environment variables.

    Useful for testing and configuration updates.
    """
    global _memory_config
    _memory_config = MemoryConfig.from_env()
    _memory_config.log_configuration()
    return _memory_config


# Configuration validation function
def validate_memory_config(config: MemoryConfig) -> List[str]:
    """
    Validate memory configuration and return list of issues.

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    try:
        # This will raise exceptions for invalid configurations
        config.__post_init__()
        config.coreference.__post_init__()
    except ValueError as e:
        issues.append(str(e))

    # Additional validation
    if config.coreference.enabled and not config.processors.enabled:
        issues.append("Coreference resolution requires processors to be enabled")

    if config.processors.total_timeout_ms < config.coreference.timeout_ms:
        issues.append("Processor total timeout must be >= coreference timeout")

    return issues