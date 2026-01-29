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

    # Parakeet streaming configuration (optimal defaults based on NVIDIA recommendations)
    chunk_duration: float = 0.5  # Audio buffering duration (2-4s optimal for reduced hallucinations)
    context_size: tuple = (256, 256)  # Attention context (128-256 tokens optimal)
    beam_width: int = 8  # Beam search width (8-10 optimal for streaming)
    depth: int = 1  # Model depth for streaming processing
    temperature: float = 0.0  # Sampling temperature for transcription
    sentence_pause_threshold: float = 1.2  # Threshold for sentence boundary detection
    max_chunk_duration: float = 4.0  # Maximum duration for audio chunks
    volume_threshold: float = 0.001  # Volume gating threshold
    enable_vad: bool = False  # Enable internal VAD
    streaming: bool = True  # Enable streaming mode for compatible engines

    @classmethod
    def from_env(cls) -> 'STTConfiguration':
        """Load STT configuration from environment variables."""
        from .parsers import _parse_int, _parse_float, _parse_bool, _parse_list

        return cls(
            engine=cls._get_env("VOICE_AGENT_STT_ENGINE", "STT_ENGINE") or cls.engine,
            model=cls._get_env("VOICE_AGENT_STT_MODEL", "STT_MODEL") or cls.model,
            language=cls._get_env("VOICE_AGENT_STT_LANGUAGE", "STT_LANGUAGE") or cls.language,
            chunk_length_ms=_parse_int(
                cls._get_env("VOICE_AGENT_STT_CHUNK_LENGTH_MS", "STT_CHUNK_LENGTH_MS"),
                cls.chunk_length_ms
            ),
            # Parakeet streaming parameters with optimal defaults
            chunk_duration=_parse_float(
                cls._get_env("VOICE_AGENT_STT_CHUNK_DURATION", "PARAKEET_CHUNK_DURATION"),
                cls.chunk_duration
            ),
            context_size=tuple(
                map(int, _parse_list(
                    cls._get_env("VOICE_AGENT_STT_CONTEXT_SIZE", "PARAKEET_CONTEXT_SIZE"),
                    [str(v) for v in cls.context_size]
                ))
            ) or cls.context_size,
            beam_width=_parse_int(
                cls._get_env("VOICE_AGENT_STT_BEAM_WIDTH", "PARAKEET_BEAM_WIDTH"),
                cls.beam_width
            ),
            depth=_parse_int(
                cls._get_env("VOICE_AGENT_STT_DEPTH", "PARAKEET_DEPTH"),
                cls.depth
            ),
            temperature=_parse_float(
                cls._get_env("VOICE_AGENT_STT_TEMPERATURE", "PARAKEET_TEMPERATURE"),
                cls.temperature
            ),
            sentence_pause_threshold=_parse_float(
                cls._get_env("VOICE_AGENT_STT_SENTENCE_PAUSE_THRESHOLD", "PARAKEET_SENTENCE_PAUSE_THRESHOLD"),
                cls.sentence_pause_threshold
            ),
            max_chunk_duration=_parse_float(
                cls._get_env("VOICE_AGENT_STT_MAX_CHUNK_DURATION", "PARAKEET_MAX_CHUNK_DURATION"),
                cls.max_chunk_duration
            ),
            volume_threshold=_parse_float(
                cls._get_env("VOICE_AGENT_STT_VOLUME_THRESHOLD", "PARAKEET_VOLUME_THRESHOLD"),
                cls.volume_threshold
            ),
            enable_vad=_parse_bool(
                cls._get_env("VOICE_AGENT_STT_ENABLE_VAD", "PARAKEET_ENABLE_VAD")
            ) or cls.enable_vad,
            streaming=_parse_bool(
                cls._get_env("VOICE_AGENT_STT_STREAMING", "PARAKEET_STREAMING")
            ) if cls._get_env("VOICE_AGENT_STT_STREAMING", "PARAKEET_STREAMING") is not None else cls.streaming,
        )

    def validate(self) -> List[str]:
        """Validate STT configuration."""
        warnings = []

        valid_engines = ["parakeet_isolated", "parakeet_streaming", "parakeet_batch", "whisper_mlx", "macos_native"]
        if self.engine not in valid_engines:
            warnings.append(f"engine='{self.engine}' not in {valid_engines}")

        if self.chunk_length_ms < 50 or self.chunk_length_ms > 1000:
            warnings.append(f"chunk_length_ms={self.chunk_length_ms} outside recommended range [50-1000]")

        # Parakeet-specific validation
        if self.engine.startswith("parakeet"):
            if self.chunk_duration < 2.0 or self.chunk_duration > 4.0:
                warnings.append(f"chunk_duration={self.chunk_duration}s outside optimal range [2.0-4.0] for reduced hallucinations")

            if len(self.context_size) != 2:
                warnings.append(f"context_size={self.context_size} must be a tuple of (left_context, right_context)")
            else:
                left_ctx, right_ctx = self.context_size
                if left_ctx < 128 or left_ctx > 256 or right_ctx < 128 or right_ctx > 256:
                    warnings.append(f"context_size={self.context_size} outside optimal range [128-256] for each dimension")

            if self.beam_width < 8 or self.beam_width > 10:
                warnings.append(f"beam_width={self.beam_width} outside optimal range [8-10] for streaming hallucination reduction")

            if self.depth < 1 or self.depth > 5:
                warnings.append(f"depth={self.depth} outside reasonable range [1-5]")

            if self.temperature < 0.0 or self.temperature > 1.0:
                warnings.append(f"temperature={self.temperature} outside recommended range [0.0-1.0]")

        return warnings


@dataclass
class TTSConfiguration(BaseConfiguration):
    """Text-to-speech service configuration."""

    engine: str = "supertonic"  # Default to Supertonic (fastest, most reliable)
    voice: str = "af_heart"  # Kokoro voice (for backward compat)
    speed: float = 1.0
    sample_rate: int = 24000
    fade_duration_ms: float = 50.0
    target_peak_db: float = -3.0
    enable_quality_logging: bool = True
    chunk_size_chars: int = 25

    # Supertonic-specific settings
    supertonic_voice: str = "F1"  # M1-M5 (male), F1-F5 (female)
    supertonic_total_steps: int = 2  # 2=fast, 5=higher quality
    supertonic_model_dir: Optional[str] = None  # Custom model path for bundling

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
            # Supertonic-specific
            supertonic_voice=os.getenv("SUPERTONIC_VOICE") or cls.supertonic_voice,
            supertonic_total_steps=_parse_int(os.getenv("SUPERTONIC_TOTAL_STEPS"), cls.supertonic_total_steps),
            supertonic_model_dir=os.getenv("SUPERTONIC_MODEL_PATH") or cls.supertonic_model_dir,
        )

    def validate(self) -> List[str]:
        """Validate TTS configuration."""
        warnings = []

        valid_engines = ["qwen3", "supertonic", "kokoro_professional", "kokoro_mlx", "kokoro_pytorch", "siri_streaming"]
        if self.engine not in valid_engines:
            warnings.append(f"engine='{self.engine}' not in {valid_engines}")

        if self.speed < 0.5 or self.speed > 2.0:
            warnings.append(f"speed={self.speed} outside reasonable range [0.5-2.0]")

        if self.chunk_size_chars < 20 or self.chunk_size_chars > 50:
            warnings.append(f"chunk_size_chars={self.chunk_size_chars} outside optimal range [20-50]")

        # Supertonic-specific validation
        if self.engine == "supertonic":
            valid_voices = {"M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"}
            if self.supertonic_voice not in valid_voices:
                warnings.append(f"supertonic_voice='{self.supertonic_voice}' not in {valid_voices}")
            if self.supertonic_total_steps < 1 or self.supertonic_total_steps > 10:
                warnings.append(f"supertonic_total_steps={self.supertonic_total_steps} outside range [1-10]")

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
