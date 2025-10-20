"""
Base configuration classes for voice agent system.

Provides abstract base classes and common functionality for configuration
management with validation and environment variable parsing.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from loguru import logger


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


@dataclass
class BaseConfiguration(ABC):
    """
    Abstract base class for all configuration sections.

    Provides common functionality for:
    - Environment variable parsing
    - Validation
    - Serialization
    - Error handling
    """

    @classmethod
    @abstractmethod
    def from_env(cls) -> 'BaseConfiguration':
        """
        Load configuration from environment variables.

        Must be implemented by subclasses to define how to parse
        their specific environment variables.

        Returns:
            Configuration instance populated from environment

        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        pass

    @abstractmethod
    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation warnings/errors (empty if valid)
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for logging/debugging.

        Default implementation uses dataclass fields. Override if needed.

        Returns:
            Dictionary representation of configuration
        """
        from dataclasses import asdict
        return asdict(self)

    def log_configuration(self, level: str = "INFO") -> None:
        """
        Log current configuration at specified level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        config_dict = self.to_dict()
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"{self.__class__.__name__} loaded: {config_dict}")

    def validate_and_raise(self) -> None:
        """
        Validate configuration and raise exception if invalid.

        Raises:
            ConfigurationError: If validation fails
        """
        warnings = self.validate()
        if warnings:
            error_msg = f"{self.__class__.__name__} validation failed:\n"
            error_msg += "\n".join(f"  - {w}" for w in warnings)
            raise ConfigurationError(error_msg)

    @staticmethod
    def _get_env(key: str, legacy_key: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable with optional legacy fallback.

        Args:
            key: Primary environment variable name
            legacy_key: Optional legacy name for backward compatibility

        Returns:
            Environment variable value or None
        """
        value = os.getenv(key)
        if value is None and legacy_key:
            value = os.getenv(legacy_key)
            if value is not None:
                logger.debug(f"Using legacy env var {legacy_key} (prefer {key})")
        return value


@dataclass
class LLMConfiguration(BaseConfiguration):
    """LLM service configuration."""

    # Connection settings
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "not-needed"
    model: str = "gemma3n:e2b"

    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.7

    # Context management
    context_max_tokens: int = 3000
    context_prune_threshold: float = 0.70
    context_min_turns: int = 3

    # Streaming
    use_streaming: bool = True

    @classmethod
    def from_env(cls) -> 'LLMConfiguration':
        """Load LLM configuration from environment variables."""
        from .parsers import _parse_int, _parse_float, _parse_bool

        return cls(
            base_url=cls._get_env("LLM_BASE_URL", "VOICE_AGENT_LLM_BASE_URL") or cls.base_url,
            api_key=cls._get_env("LLM_API_KEY", "VOICE_AGENT_LLM_API_KEY") or cls.api_key,
            model=cls._get_env("LLM_MODEL", "VOICE_AGENT_LLM_MODEL") or cls.model,
            max_tokens=_parse_int(cls._get_env("LLM_MAX_TOKENS", "VOICE_AGENT_LLM_MAX_TOKENS"), cls.max_tokens),
            temperature=_parse_float(cls._get_env("LLM_TEMPERATURE", "VOICE_AGENT_LLM_TEMPERATURE"), cls.temperature),
            context_max_tokens=_parse_int(os.getenv("LLM_CONTEXT_MAX_TOKENS"), cls.context_max_tokens),
            context_prune_threshold=_parse_float(os.getenv("LLM_CONTEXT_PRUNE_THRESHOLD"), cls.context_prune_threshold),
            context_min_turns=_parse_int(os.getenv("LLM_CONTEXT_MIN_TURNS"), cls.context_min_turns),
            use_streaming=_parse_bool(os.getenv("LLM_USE_STREAMING")) if os.getenv("LLM_USE_STREAMING") else cls.use_streaming,
        )

    def validate(self) -> List[str]:
        """Validate LLM configuration."""
        warnings = []

        if self.context_max_tokens < 1000:
            warnings.append(f"context_max_tokens={self.context_max_tokens} is very low (minimum 1000 recommended)")

        if self.context_prune_threshold < 0.5 or self.context_prune_threshold > 1.0:
            warnings.append(f"context_prune_threshold={self.context_prune_threshold} outside valid range [0.5-1.0]")

        if self.context_min_turns < 1:
            warnings.append(f"context_min_turns={self.context_min_turns} must be at least 1")

        if self.temperature < 0 or self.temperature > 2:
            warnings.append(f"temperature={self.temperature} outside typical range [0-2]")

        return warnings


@dataclass
class STTConfiguration(BaseConfiguration):
    """Speech-to-text service configuration."""

    engine: str = "parakeet_streaming"
    model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    language: str = "en"
    chunk_length_ms: int = 100

    @classmethod
    def from_env(cls) -> 'STTConfiguration':
        """Load STT configuration from environment variables."""
        from .parsers import _parse_int

        return cls(
            engine=cls._get_env("VOICE_AGENT_STT_ENGINE", "STT_ENGINE") or cls.engine,
            model=cls._get_env("VOICE_AGENT_STT_MODEL", "STT_MODEL") or cls.model,
            language=cls._get_env("VOICE_AGENT_STT_LANGUAGE", "STT_LANGUAGE") or cls.language,
            chunk_length_ms=_parse_int(
                cls._get_env("VOICE_AGENT_STT_CHUNK_LENGTH_MS", "STT_CHUNK_LENGTH_MS"),
                cls.chunk_length_ms
            ),
        )

    def validate(self) -> List[str]:
        """Validate STT configuration."""
        warnings = []

        valid_engines = ["parakeet_streaming", "parakeet_batch", "whisper_mlx"]
        if self.engine not in valid_engines:
            warnings.append(f"engine='{self.engine}' not in {valid_engines}")

        if self.chunk_length_ms < 50 or self.chunk_length_ms > 1000:
            warnings.append(f"chunk_length_ms={self.chunk_length_ms} outside recommended range [50-1000]")

        return warnings


@dataclass
class TTSConfiguration(BaseConfiguration):
    """Text-to-speech service configuration."""

    engine: str = "kokoro_mlx"
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
    fade_duration_ms: float = 50.0
    target_peak_db: float = -3.0
    enable_quality_logging: bool = True
    chunk_size_chars: int = 25

    @classmethod
    def from_env(cls) -> 'TTSConfiguration':
        """Load TTS configuration from environment variables."""
        from .parsers import _parse_float, _parse_int, _parse_bool

        return cls(
            engine=cls._get_env("VOICE_AGENT_TTS_ENGINE", "TTS_ENGINE") or cls.engine,
            voice=cls._get_env("VOICE_AGENT_TTS_VOICE", "TTS_VOICE") or cls.voice,
            speed=_parse_float(cls._get_env("VOICE_AGENT_TTS_SPEED", "TTS_SPEED"), cls.speed),
            sample_rate=_parse_int(cls._get_env("VOICE_AGENT_TTS_SAMPLE_RATE", "TTS_SAMPLE_RATE"), cls.sample_rate),
            fade_duration_ms=_parse_float(cls._get_env("VOICE_AGENT_TTS_FADE_DURATION_MS"), cls.fade_duration_ms),
            target_peak_db=_parse_float(cls._get_env("TTS_TARGET_PEAK_DB"), cls.target_peak_db),
            enable_quality_logging=_parse_bool(os.getenv("TTS_ENABLE_QUALITY_LOGGING")) if os.getenv("TTS_ENABLE_QUALITY_LOGGING") else cls.enable_quality_logging,
            chunk_size_chars=_parse_int(os.getenv("TTS_CHUNK_SIZE_CHARS"), cls.chunk_size_chars),
        )

    def validate(self) -> List[str]:
        """Validate TTS configuration."""
        warnings = []

        valid_engines = ["kokoro_professional", "kokoro_mlx", "siri_streaming"]
        if self.engine not in valid_engines:
            warnings.append(f"engine='{self.engine}' not in {valid_engines}")

        if self.speed < 0.5 or self.speed > 2.0:
            warnings.append(f"speed={self.speed} outside reasonable range [0.5-2.0]")

        if self.chunk_size_chars < 20 or self.chunk_size_chars > 50:
            warnings.append(f"chunk_size_chars={self.chunk_size_chars} outside optimal range [20-50]")

        return warnings


@dataclass
class VisionConfiguration(BaseConfiguration):
    """Vision processing configuration."""

    input_enabled: bool = True
    target_fps: float = 0.5
    out_enabled: bool = False
    model_enabled: bool = False

    # Optimization settings
    image_size: int = 384
    image_quality: int = 85
    max_images_in_context: int = 2
    enable_deduplication: bool = True

    # Keyword filtering
    keyword_filter: bool = True
    keywords: str = "see,look,show,what,describe,image,picture,video,color,object,room,view,watch,observe"

    @classmethod
    def from_env(cls) -> 'VisionConfiguration':
        """Load vision configuration from environment variables."""
        from .parsers import _parse_bool, _parse_float, _parse_int

        return cls(
            input_enabled=_parse_bool(os.getenv("VIDEO_INPUT_ENABLED")),
            target_fps=_parse_float(os.getenv("VIDEO_TARGET_FPS"), cls.target_fps),
            out_enabled=_parse_bool(os.getenv("VIDEO_OUT_ENABLED")),
            model_enabled=_parse_bool(os.getenv("VISION_MODEL_ENABLED")),
            image_size=_parse_int(os.getenv("VISION_IMAGE_SIZE"), cls.image_size),
            image_quality=_parse_int(os.getenv("VISION_IMAGE_QUALITY"), cls.image_quality),
            max_images_in_context=_parse_int(os.getenv("VISION_MAX_IMAGES_IN_CONTEXT"), cls.max_images_in_context),
            enable_deduplication=_parse_bool(os.getenv("VISION_ENABLE_DEDUPLICATION")) if os.getenv("VISION_ENABLE_DEDUPLICATION") else cls.enable_deduplication,
            keyword_filter=_parse_bool(os.getenv("VISION_KEYWORD_FILTER")) if os.getenv("VISION_KEYWORD_FILTER") else cls.keyword_filter,
            keywords=os.getenv("VISION_KEYWORDS") or cls.keywords,
        )

    def validate(self) -> List[str]:
        """Validate vision configuration."""
        warnings = []

        if self.image_quality < 1 or self.image_quality > 100:
            warnings.append(f"image_quality={self.image_quality} outside valid range [1-100]")

        if self.max_images_in_context < 1:
            warnings.append(f"max_images_in_context={self.max_images_in_context} must be at least 1")

        if self.target_fps < 0.1 or self.target_fps > 10:
            warnings.append(f"target_fps={self.target_fps} outside reasonable range [0.1-10]")

        return warnings
