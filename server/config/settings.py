"""
Centralized configuration for voice agent system.
Consolidates all environment variables using composition pattern.

This module provides a unified VoiceAgentConfig that composes specialized
configuration sections (LLM, STT, TTS, Vision) while maintaining full
backward compatibility with existing code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

from loguru import logger
from .base_config import (
    BaseConfiguration,
    LLMConfiguration,
    STTConfiguration,
    TTSConfiguration,
    VisionConfiguration,
)
from .parsers import _parse_bool, _parse_int, _parse_float


@dataclass
class VoiceAgentConfig(BaseConfiguration):
    """
    Comprehensive configuration for voice agent system.

    Uses composition to organize configuration into logical sections while
    maintaining backward compatibility through property accessors.

    Architecture:
    - llm: LLMConfiguration (base_url, model, tokens, temperature, context management)
    - stt: STTConfiguration (engine, model, language, chunk settings)
    - tts: TTSConfiguration (engine, voice, speed, sample rate, quality)
    - vision: VisionConfiguration (video input, FPS, image processing)

    Backward Compatibility:
    All flat attributes (e.g., config.llm_base_url) are preserved via properties
    that delegate to the appropriate configuration section.
    """

    # ============================================================================
    # Composed Configuration Sections
    # ============================================================================
    llm: LLMConfiguration = field(default_factory=LLMConfiguration)
    stt: STTConfiguration = field(default_factory=STTConfiguration)
    tts: TTSConfiguration = field(default_factory=TTSConfiguration)
    vision: VisionConfiguration = field(default_factory=VisionConfiguration)

    # ============================================================================
    # Performance Targets (based on optimization findings)
    # ============================================================================
    target_voice_to_voice_latency_ms: int = 800
    target_stt_latency_ms: int = 150
    target_llm_ttft_ms: int = 303  # Ollama llama3.2:1b benchmark
    target_tts_ttfb_ms: int = 487   # Kokoro 25-char chunks benchmark

    # ============================================================================
    # Memory Configuration (high-level settings)
    # Note: Detailed memory config is in core.memory.config_manager.MemoryConfiguration
    # ============================================================================
    memory_enabled: bool = True
    hotpath_enabled: bool = True
    session_persistence: bool = True
    memory_max_entries: int = 1000
    memory_cleanup_interval_minutes: int = 60

    # Memory injection formatting
    memory_inject_header: str = "Memory context to be used only if needed:"
    memory_sources: str = "convo,summary,graph,semantic"

    # ============================================================================
    # Audio Intelligence & Enrollment UX
    # ============================================================================
    audio_intelligence_enabled: bool = True
    enable_intro_pipeline: bool = True
    skip_intro_for_returning: bool = True
    force_intro: bool = False  # For testing
    include_privacy_explanation: bool = False
    speaker_profile_dir: str = "data/speaker_profiles"
    enable_ephemeral_choice: bool = True

    # ============================================================================
    # Pipeline Configuration
    # ============================================================================
    pipeline_audio_sample_rate: int = 24000
    pipeline_audio_channels: int = 1
    pipeline_frame_size_ms: int = 20
    enable_streaming_tts: bool = True

    # ============================================================================
    # VAD Settings
    # ============================================================================
    vad_threshold: float = 0.5
    vad_chunk_ms: int = 100

    # ============================================================================
    # Development & Debugging
    # ============================================================================
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_performance_logging: bool = True
    enable_audio_quality_validation: bool = True

    # ============================================================================
    # Paths and Files
    # ============================================================================
    models_cache_dir: str = str(Path.home() / ".cache" / "voice-agent")
    session_data_dir: str = str(Path.home() / ".local" / "share" / "voice-agent")

    # ============================================================================
    # Backward Compatibility Properties (delegate to composed sections)
    # ============================================================================

    # STT properties
    @property
    def stt_engine(self) -> str:
        return self.stt.engine

    @stt_engine.setter
    def stt_engine(self, value: str):
        self.stt.engine = value

    @property
    def stt_model(self) -> str:
        return self.stt.model

    @stt_model.setter
    def stt_model(self, value: str):
        self.stt.model = value

    @property
    def stt_language(self) -> str:
        return self.stt.language

    @stt_language.setter
    def stt_language(self, value: str):
        self.stt.language = value

    @property
    def stt_chunk_length_ms(self) -> int:
        return self.stt.chunk_length_ms

    @stt_chunk_length_ms.setter
    def stt_chunk_length_ms(self, value: int):
        self.stt.chunk_length_ms = value

    # TTS properties
    @property
    def tts_engine(self) -> str:
        return self.tts.engine

    @tts_engine.setter
    def tts_engine(self, value: str):
        self.tts.engine = value

    @property
    def tts_voice(self) -> str:
        return self.tts.voice

    @tts_voice.setter
    def tts_voice(self, value: str):
        self.tts.voice = value

    @property
    def tts_speed(self) -> float:
        return self.tts.speed

    @tts_speed.setter
    def tts_speed(self, value: float):
        self.tts.speed = value

    @property
    def tts_sample_rate(self) -> int:
        return self.tts.sample_rate

    @tts_sample_rate.setter
    def tts_sample_rate(self, value: int):
        self.tts.sample_rate = value

    @property
    def tts_fade_duration_ms(self) -> float:
        return self.tts.fade_duration_ms

    @tts_fade_duration_ms.setter
    def tts_fade_duration_ms(self, value: float):
        self.tts.fade_duration_ms = value

    @property
    def tts_target_peak_db(self) -> float:
        return self.tts.target_peak_db

    @tts_target_peak_db.setter
    def tts_target_peak_db(self, value: float):
        self.tts.target_peak_db = value

    @property
    def tts_enable_quality_logging(self) -> bool:
        return self.tts.enable_quality_logging

    @tts_enable_quality_logging.setter
    def tts_enable_quality_logging(self, value: bool):
        self.tts.enable_quality_logging = value

    @property
    def tts_chunk_size_chars(self) -> int:
        return self.tts.chunk_size_chars

    @tts_chunk_size_chars.setter
    def tts_chunk_size_chars(self, value: int):
        self.tts.chunk_size_chars = value

    # LLM properties
    @property
    def llm_base_url(self) -> str:
        return self.llm.base_url

    @llm_base_url.setter
    def llm_base_url(self, value: str):
        self.llm.base_url = value

    @property
    def llm_model(self) -> str:
        return self.llm.model

    @llm_model.setter
    def llm_model(self, value: str):
        self.llm.model = value

    @property
    def llm_api_key(self) -> str:
        return self.llm.api_key

    @llm_api_key.setter
    def llm_api_key(self, value: str):
        self.llm.api_key = value

    @property
    def llm_max_tokens(self) -> int:
        return self.llm.max_tokens

    @llm_max_tokens.setter
    def llm_max_tokens(self, value: int):
        self.llm.max_tokens = value

    @property
    def llm_temperature(self) -> float:
        return self.llm.temperature

    @llm_temperature.setter
    def llm_temperature(self, value: float):
        self.llm.temperature = value

    @property
    def llm_context_max_tokens(self) -> int:
        return self.llm.context_max_tokens

    @llm_context_max_tokens.setter
    def llm_context_max_tokens(self, value: int):
        self.llm.context_max_tokens = value

    @property
    def llm_context_prune_threshold(self) -> float:
        return self.llm.context_prune_threshold

    @llm_context_prune_threshold.setter
    def llm_context_prune_threshold(self, value: float):
        self.llm.context_prune_threshold = value

    @property
    def llm_context_min_turns(self) -> int:
        return self.llm.context_min_turns

    @llm_context_min_turns.setter
    def llm_context_min_turns(self, value: int):
        self.llm.context_min_turns = value

    # Vision properties
    @property
    def video_input_enabled(self) -> bool:
        return self.vision.input_enabled

    @video_input_enabled.setter
    def video_input_enabled(self, value: bool):
        self.vision.input_enabled = value

    @property
    def video_target_fps(self) -> float:
        return self.vision.target_fps

    @video_target_fps.setter
    def video_target_fps(self, value: float):
        self.vision.target_fps = value

    @property
    def video_out_enabled(self) -> bool:
        return self.vision.out_enabled

    @video_out_enabled.setter
    def video_out_enabled(self, value: bool):
        self.vision.out_enabled = value

    @property
    def vision_model_enabled(self) -> bool:
        return self.vision.model_enabled

    @vision_model_enabled.setter
    def vision_model_enabled(self, value: bool):
        self.vision.model_enabled = value

    @property
    def vision_image_size(self) -> int:
        return self.vision.image_size

    @vision_image_size.setter
    def vision_image_size(self, value: int):
        self.vision.image_size = value

    @property
    def vision_image_quality(self) -> int:
        return self.vision.image_quality

    @vision_image_quality.setter
    def vision_image_quality(self, value: int):
        self.vision.image_quality = value

    @property
    def vision_max_images_in_context(self) -> int:
        return self.vision.max_images_in_context

    @vision_max_images_in_context.setter
    def vision_max_images_in_context(self, value: int):
        self.vision.max_images_in_context = value

    @property
    def vision_enable_deduplication(self) -> bool:
        return self.vision.enable_deduplication

    @vision_enable_deduplication.setter
    def vision_enable_deduplication(self, value: bool):
        self.vision.enable_deduplication = value

    @property
    def vision_keyword_filter(self) -> bool:
        return self.vision.keyword_filter

    @vision_keyword_filter.setter
    def vision_keyword_filter(self, value: bool):
        self.vision.keyword_filter = value

    @property
    def vision_keywords(self) -> str:
        return self.vision.keywords

    @vision_keywords.setter
    def vision_keywords(self, value: str):
        self.vision.keywords = value

    # ============================================================================
    # Configuration Loading
    # ============================================================================

    @classmethod
    def from_env(cls) -> 'VoiceAgentConfig':
        """
        Load configuration from environment variables with fallback to defaults.

        Loads each configuration section independently, then combines them
        with additional voice agent specific settings.
        """
        # Load composed configuration sections
        llm = LLMConfiguration.from_env()
        stt = STTConfiguration.from_env()
        tts = TTSConfiguration.from_env()
        vision = VisionConfiguration.from_env()

        # Create config with composed sections
        config = cls(llm=llm, stt=stt, tts=tts, vision=vision)

        # Load performance targets
        config.target_voice_to_voice_latency_ms = _parse_int(
            os.getenv("VOICE_AGENT_TARGET_LATENCY_MS"),
            config.target_voice_to_voice_latency_ms
        )

        # Load memory configuration
        config.memory_enabled = _parse_bool(os.getenv("VOICE_AGENT_MEMORY_ENABLED", "true"))
        config.hotpath_enabled = _parse_bool(os.getenv("VOICE_AGENT_HOTPATH_ENABLED", "true"))
        config.session_persistence = _parse_bool(os.getenv("VOICE_AGENT_SESSION_PERSISTENCE", "true"))
        config.memory_inject_header = os.getenv("MEMORY_INJECT_HEADER", config.memory_inject_header)
        config.memory_sources = os.getenv("MEMORY_SOURCES", config.memory_sources)

        # Load audio intelligence settings
        config.audio_intelligence_enabled = _parse_bool(os.getenv("AUDIO_INTELLIGENCE_ENABLED", "true"))
        config.enable_intro_pipeline = _parse_bool(os.getenv("AUDIO_INTEL_INTRO_PIPELINE", "true"))
        config.skip_intro_for_returning = _parse_bool(os.getenv("AUDIO_INTEL_SKIP_FOR_RETURNING", "true"))
        config.force_intro = _parse_bool(os.getenv("AUDIO_INTEL_FORCE_INTRO", "false"))
        config.include_privacy_explanation = _parse_bool(os.getenv("AUDIO_INTEL_INCLUDE_PRIVACY", "false"))

        if os.getenv("SPEAKER_PROFILE_DIR"):
            config.speaker_profile_dir = os.getenv("SPEAKER_PROFILE_DIR")

        config.enable_ephemeral_choice = _parse_bool(
            os.getenv("ENABLE_EPHEMERAL_CHOICE", str(config.enable_ephemeral_choice))
        )

        # Load development settings
        config.debug_mode = _parse_bool(os.getenv("VOICE_AGENT_DEBUG_MODE", "false"))
        config.log_level = os.getenv("VOICE_AGENT_LOG_LEVEL", config.log_level)

        # Support legacy environment variables for backward compatibility
        config._load_legacy_env_vars()

        return config

    def _load_legacy_env_vars(self):
        """Support for existing environment variables during transition."""
        # Legacy TTS variables
        if os.getenv("TTS_VOICE"):
            self.tts.voice = os.getenv("TTS_VOICE")
        if os.getenv("TTS_SPEED"):
            self.tts.speed = _parse_float(os.getenv("TTS_SPEED"), self.tts.speed)

        # Legacy LLM variables
        if os.getenv("LLM_BASE_URL"):
            self.llm.base_url = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_MODEL"):
            self.llm.model = os.getenv("LLM_MODEL")

        # Legacy debug variables
        if os.getenv("DEBUG"):
            self.debug_mode = _parse_bool(os.getenv("DEBUG"))

    def validate(self) -> List[str]:
        """
        Validate configuration for common issues.

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []

        # Validate composed sections
        warnings.extend(self.llm.validate())
        warnings.extend(self.stt.validate())
        warnings.extend(self.tts.validate())
        warnings.extend(self.vision.validate())

        # Validate performance targets
        if self.target_voice_to_voice_latency_ms < 500:
            warnings.append(
                f"Voice-to-voice latency target too aggressive: {self.target_voice_to_voice_latency_ms}ms"
            )

        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration validation: {warning}")

        return warnings

    def get_component_config(self, component: str) -> Dict[str, Any]:
        """
        Get configuration subset for a specific component.

        This method maintains backward compatibility with code that uses
        get_component_config() to retrieve settings.

        Args:
            component: Component name (stt, tts, llm, memory, pipeline)

        Returns:
            Dictionary of configuration values for the component
        """
        if component == "stt":
            return self.stt.to_dict()
        elif component == "tts":
            return self.tts.to_dict()
        elif component == "llm":
            return self.llm.to_dict()
        elif component == "memory":
            return {
                "enabled": self.memory_enabled,
                "hotpath_enabled": self.hotpath_enabled,
                "session_persistence": self.session_persistence,
                "max_entries": self.memory_max_entries,
                "cleanup_interval_minutes": self.memory_cleanup_interval_minutes,
            }
        elif component == "pipeline":
            return {
                "audio_sample_rate": self.pipeline_audio_sample_rate,
                "audio_channels": self.pipeline_audio_channels,
                "frame_size_ms": self.pipeline_frame_size_ms,
                "enable_streaming_tts": self.enable_streaming_tts,
            }
        elif component == "vision":
            return self.vision.to_dict()
        else:
            raise ValueError(f"Unknown component: {component}")

    def summary(self) -> str:
        """Generate a human-readable configuration summary."""
        return f"""
Voice Agent Configuration Summary:
╭─ Core Components ─────────────────────────────────────
│  STT: {self.stt.engine} ({self.stt.model})
│  TTS: {self.tts.engine} (voice: {self.tts.voice})
│  LLM: {self.llm.model} via {self.llm.base_url}
├─ Performance Targets ────────────────────────────────
│  Voice-to-Voice: <{self.target_voice_to_voice_latency_ms}ms
│  TTS Chunking: {self.tts.chunk_size_chars} chars/chunk
│  Streaming: {'Enabled' if self.enable_streaming_tts else 'Disabled'}
├─ Memory System ──────────────────────────────────────
│  Memory: {'Enabled' if self.memory_enabled else 'Disabled'}
│  HotPath: {'Enabled' if self.hotpath_enabled else 'Disabled'}
│  Session Persistence: {'Enabled' if self.session_persistence else 'Disabled'}
├─ Vision Processing ──────────────────────────────────
│  Video Input: {'Enabled' if self.vision.input_enabled else 'Disabled'}
│  Frame Rate: {self.vision.target_fps}fps
│  Image Size: {self.vision.image_size}×{self.vision.image_size}px
│  Max Images in Context: {self.vision.max_images_in_context}
│  Deduplication: {'Enabled' if self.vision.enable_deduplication else 'Disabled'}
╰─ Debug ──────────────────────────────────────────────
   Debug Mode: {'Enabled' if self.debug_mode else 'Disabled'}
   Log Level: {self.log_level}
   Performance Logging: {'Enabled' if self.enable_performance_logging else 'Disabled'}
"""
