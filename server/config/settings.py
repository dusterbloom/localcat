"""
Centralized configuration for voice agent system.
Consolidates all environment variables and provides structured configuration.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


@dataclass
class VoiceAgentConfig:
    """
    Comprehensive configuration for voice agent system.

    Consolidates all environment variables into a structured configuration
    with sensible defaults based on optimization findings.
    """

    # ============================================================================
    # STT Configuration
    # ============================================================================
    stt_engine: str = "parakeet_streaming"  # parakeet_streaming | parakeet_batch | whisper_mlx
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    stt_chunk_length_ms: int = 100
    stt_language: str = "en"

    # ============================================================================
    # TTS Configuration
    # ============================================================================
    tts_engine: str = "kokoro_mlx"  # kokoro_professional | kokoro_mlx
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    tts_sample_rate: int = 24000
    tts_fade_duration_ms: float = 50.0
    tts_target_peak_db: float = -3.0
    tts_enable_quality_logging: bool = True

    # ============================================================================
    # LLM Configuration
    # ============================================================================
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma3n:e2b"
    llm_api_key: str = "not-needed"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # ============================================================================
    # Performance Targets (based on optimization findings)
    # ============================================================================
    target_voice_to_voice_latency_ms: int = 800
    target_stt_latency_ms: int = 150
    target_llm_ttft_ms: int = 303  # Ollama llama3.2:1b benchmark
    target_tts_ttfb_ms: int = 487   # Kokoro 25-char chunks benchmark

    # ============================================================================
    # Memory Configuration
    # ============================================================================
    memory_enabled: bool = True
    hotpath_enabled: bool = True
    session_persistence: bool = True
    memory_max_entries: int = 1000
    memory_cleanup_interval_minutes: int = 60

    # ============================================================================
    # Audio Intelligence & Enrollment UX
    # ============================================================================
    audio_intelligence_enabled: bool = True
    enable_intro_pipeline: bool = True
    skip_intro_for_returning: bool = True
    force_intro: bool = False  # For testing
    include_privacy_explanation: bool = False
    speaker_profile_dir: str = "data/speaker_profiles"

    # ============================================================================
    # Pipeline Configuration
    # ============================================================================
    pipeline_audio_sample_rate: int = 24000
    pipeline_audio_channels: int = 1
    pipeline_frame_size_ms: int = 20

    # ============================================================================
    # Optimization Settings (based on research findings)
    # ============================================================================
    # Kokoro chunking: 25 chars = 487ms TTFB (optimal)
    tts_chunk_size_chars: int = 25
    # Enable streaming TTS while LLM generates
    enable_streaming_tts: bool = True
    # VAD settings for <100ms detection
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

    @classmethod
    def from_env(cls) -> 'VoiceAgentConfig':
        """
        Load configuration from environment variables with fallback to defaults.

        Environment variables follow the pattern: VOICE_AGENT_<SECTION>_<SETTING>
        Example: VOICE_AGENT_TTS_ENGINE=kokoro_professional
        """
        config = cls()

        # STT Configuration
        config.stt_engine = os.getenv("VOICE_AGENT_STT_ENGINE", config.stt_engine)
        config.stt_model = os.getenv("VOICE_AGENT_STT_MODEL", config.stt_model)
        config.stt_chunk_length_ms = int(os.getenv("VOICE_AGENT_STT_CHUNK_LENGTH_MS", config.stt_chunk_length_ms))
        config.stt_language = os.getenv("VOICE_AGENT_STT_LANGUAGE", config.stt_language)

        # TTS Configuration
        config.tts_engine = os.getenv("VOICE_AGENT_TTS_ENGINE", config.tts_engine)
        config.tts_voice = os.getenv("VOICE_AGENT_TTS_VOICE", config.tts_voice)
        config.tts_speed = float(os.getenv("VOICE_AGENT_TTS_SPEED", config.tts_speed))
        config.tts_sample_rate = int(os.getenv("VOICE_AGENT_TTS_SAMPLE_RATE", config.tts_sample_rate))
        config.tts_fade_duration_ms = float(os.getenv("VOICE_AGENT_TTS_FADE_DURATION_MS", config.tts_fade_duration_ms))

        # LLM Configuration
        config.llm_base_url = os.getenv("VOICE_AGENT_LLM_BASE_URL", config.llm_base_url)
        config.llm_model = os.getenv("VOICE_AGENT_LLM_MODEL", config.llm_model)
        config.llm_api_key = os.getenv("VOICE_AGENT_LLM_API_KEY", config.llm_api_key)
        config.llm_max_tokens = int(os.getenv("VOICE_AGENT_LLM_MAX_TOKENS", config.llm_max_tokens))
        config.llm_temperature = float(os.getenv("VOICE_AGENT_LLM_TEMPERATURE", config.llm_temperature))

        # Performance Targets
        config.target_voice_to_voice_latency_ms = int(os.getenv("VOICE_AGENT_TARGET_LATENCY_MS", config.target_voice_to_voice_latency_ms))

        # Memory Configuration
        config.memory_enabled = os.getenv("VOICE_AGENT_MEMORY_ENABLED", "true").lower() == "true"
        config.hotpath_enabled = os.getenv("VOICE_AGENT_HOTPATH_ENABLED", "true").lower() == "true"
        config.session_persistence = os.getenv("VOICE_AGENT_SESSION_PERSISTENCE", "true").lower() == "true"

        # Development & Debugging
        config.debug_mode = os.getenv("VOICE_AGENT_DEBUG_MODE", "false").lower() == "true"
        config.log_level = os.getenv("VOICE_AGENT_LOG_LEVEL", config.log_level)

        # Audio Intelligence & Enrollment UX
        config.audio_intelligence_enabled = os.getenv("AUDIO_INTELLIGENCE_ENABLED", "true").lower() == "true"
        config.enable_intro_pipeline = os.getenv("AUDIO_INTEL_INTRO_PIPELINE", "true").lower() == "true"
        config.skip_intro_for_returning = os.getenv("AUDIO_INTEL_SKIP_FOR_RETURNING", "true").lower() == "true"
        config.force_intro = os.getenv("AUDIO_INTEL_FORCE_INTRO", "false").lower() == "true"
        config.include_privacy_explanation = os.getenv("AUDIO_INTEL_INCLUDE_PRIVACY", "false").lower() == "true"
        if os.getenv("SPEAKER_PROFILE_DIR"):
            config.speaker_profile_dir = os.getenv("SPEAKER_PROFILE_DIR")

        # Legacy environment variable support for backward compatibility
        config._load_legacy_env_vars()

        return config

    def _load_legacy_env_vars(self):
        """Support for existing environment variables during transition."""
        # Support existing TTS variables
        if os.getenv("TTS_VOICE"):
            self.tts_voice = os.getenv("TTS_VOICE")
        if os.getenv("TTS_SPEED"):
            self.tts_speed = float(os.getenv("TTS_SPEED"))

        # Support existing LLM variables
        if os.getenv("LLM_BASE_URL"):
            self.llm_base_url = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_MODEL"):
            self.llm_model = os.getenv("LLM_MODEL")

        # Support existing debug variables
        if os.getenv("DEBUG"):
            self.debug_mode = os.getenv("DEBUG").lower() == "true"

    def validate(self) -> bool:
        """
        Validate configuration for common issues.

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Validate STT engine
        if self.stt_engine not in ["parakeet_streaming", "parakeet_batch", "whisper_mlx"]:
            errors.append(f"Invalid STT engine: {self.stt_engine}")

        # Validate TTS engine
        if self.tts_engine not in ["kokoro_professional", "kokoro_mlx"]:
            errors.append(f"Invalid TTS engine: {self.tts_engine}")

        # Validate performance targets
        if self.target_voice_to_voice_latency_ms < 500:
            errors.append(f"Voice-to-voice latency target too aggressive: {self.target_voice_to_voice_latency_ms}ms")

        # Validate chunk size (based on optimization findings)
        if self.tts_chunk_size_chars < 20 or self.tts_chunk_size_chars > 50:
            logger.warning(f"TTS chunk size {self.tts_chunk_size_chars} may not be optimal (recommended: 25)")

        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False

        return True

    def get_component_config(self, component: str) -> Dict[str, Any]:
        """
        Get configuration subset for a specific component.

        Args:
            component: Component name (stt, tts, llm, memory, pipeline)

        Returns:
            Dictionary of configuration values for the component
        """
        if component == "stt":
            return {
                "engine": self.stt_engine,
                "model": self.stt_model,
                "chunk_length_ms": self.stt_chunk_length_ms,
                "language": self.stt_language,
            }
        elif component == "tts":
            return {
                "engine": self.tts_engine,
                "voice": self.tts_voice,
                "speed": self.tts_speed,
                "sample_rate": self.tts_sample_rate,
                "fade_duration_ms": self.tts_fade_duration_ms,
                "target_peak_db": self.tts_target_peak_db,
                "chunk_size_chars": self.tts_chunk_size_chars,
                "enable_quality_logging": self.tts_enable_quality_logging,
            }
        elif component == "llm":
            return {
                "base_url": self.llm_base_url,
                "model": self.llm_model,
                "api_key": self.llm_api_key,
                "max_tokens": self.llm_max_tokens,
                "temperature": self.llm_temperature,
            }
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
        else:
            raise ValueError(f"Unknown component: {component}")

    def summary(self) -> str:
        """Generate a human-readable configuration summary."""
        return f"""
Voice Agent Configuration Summary:
╭─ Core Components ─────────────────────────────────────
│  STT: {self.stt_engine} ({self.stt_model})
│  TTS: {self.tts_engine} (voice: {self.tts_voice})
│  LLM: {self.llm_model} via {self.llm_base_url}
├─ Performance Targets ────────────────────────────────
│  Voice-to-Voice: <{self.target_voice_to_voice_latency_ms}ms
│  TTS Chunking: {self.tts_chunk_size_chars} chars/chunk
│  Streaming: {'Enabled' if self.enable_streaming_tts else 'Disabled'}
├─ Memory System ──────────────────────────────────────
│  Memory: {'Enabled' if self.memory_enabled else 'Disabled'}
│  HotPath: {'Enabled' if self.hotpath_enabled else 'Disabled'}
│  Session Persistence: {'Enabled' if self.session_persistence else 'Disabled'}
╰─ Debug ──────────────────────────────────────────────
   Debug Mode: {'Enabled' if self.debug_mode else 'Disabled'}
   Log Level: {self.log_level}
   Performance Logging: {'Enabled' if self.enable_performance_logging else 'Disabled'}
"""
