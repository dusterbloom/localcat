"""
Unit tests for VoiceAgentConfig composition refactor.

Tests verify backward compatibility and proper composition of config sections.
"""

import os
import pytest
from config import VoiceAgentConfig


class TestVoiceAgentConfigComposition:
    """Test composition architecture of VoiceAgentConfig."""

    def test_composition_structure(self):
        """Test that config has proper composition structure."""
        config = VoiceAgentConfig()

        # Verify composed sections exist
        assert hasattr(config, 'llm')
        assert hasattr(config, 'stt')
        assert hasattr(config, 'tts')
        assert hasattr(config, 'vision')

        # Verify sections have correct types
        from config.base_config import (
            LLMConfiguration,
            STTConfiguration,
            TTSConfiguration,
            VisionConfiguration,
        )

        assert isinstance(config.llm, LLMConfiguration)
        assert isinstance(config.stt, STTConfiguration)
        assert isinstance(config.tts, TTSConfiguration)
        assert isinstance(config.vision, VisionConfiguration)

    def test_default_values_via_properties(self):
        """Test that property accessors return correct default values."""
        config = VoiceAgentConfig()

        # STT properties
        assert config.stt_engine == "parakeet_streaming"
        assert config.stt_model == "mlx-community/parakeet-tdt-0.6b-v3"
        assert config.stt_language == "en"
        assert config.stt_chunk_length_ms == 100

        # TTS properties
        assert config.tts_engine == "kokoro_mlx"
        assert config.tts_voice == "af_heart"
        assert config.tts_speed == 1.0
        assert config.tts_sample_rate == 24000

        # LLM properties
        assert config.llm_base_url == "http://127.0.0.1:11434/v1"
        assert config.llm_model == "gemma3n:e2b"
        assert config.llm_max_tokens == 1024
        assert config.llm_context_max_tokens == 3000

        # Vision properties
        assert config.video_input_enabled is True
        assert config.vision_image_size == 384

    def test_property_setters(self):
        """Test that property setters update the composed sections."""
        config = VoiceAgentConfig()

        # Test STT property setters
        config.stt_engine = "whisper_mlx"
        assert config.stt.engine == "whisper_mlx"
        assert config.stt_engine == "whisper_mlx"

        # Test TTS property setters
        config.tts_voice = "custom_voice"
        assert config.tts.voice == "custom_voice"
        assert config.tts_voice == "custom_voice"

        # Test LLM property setters
        config.llm_model = "test-model"
        assert config.llm.model == "test-model"
        assert config.llm_model == "test-model"

        # Test Vision property setters
        config.vision_image_size = 512
        assert config.vision.image_size == 512
        assert config.vision_image_size == 512

    def test_direct_section_access(self):
        """Test that sections can be accessed directly."""
        config = VoiceAgentConfig()

        # Test direct access to composed sections
        assert config.llm.base_url == "http://127.0.0.1:11434/v1"
        assert config.stt.engine == "parakeet_streaming"
        assert config.tts.engine == "kokoro_mlx"
        assert config.vision.input_enabled is True

        # Test direct modification
        config.llm.model = "new-model"
        assert config.llm_model == "new-model"  # Property should reflect change


class TestVoiceAgentConfigLoading:
    """Test configuration loading from environment variables."""

    def test_from_env_loads_all_sections(self, monkeypatch):
        """Test that from_env loads all configuration sections."""
        # Set various environment variables
        monkeypatch.setenv("LLM_BASE_URL", "http://test:8080/v1")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("VOICE_AGENT_STT_ENGINE", "whisper_mlx")
        monkeypatch.setenv("TTS_VOICE", "test_voice")
        monkeypatch.setenv("VIDEO_INPUT_ENABLED", "false")

        config = VoiceAgentConfig.from_env()

        # Verify sections loaded correctly
        assert config.llm_base_url == "http://test:8080/v1"
        assert config.llm_model == "test-model"
        assert config.stt_engine == "whisper_mlx"
        assert config.tts_voice == "test_voice"
        assert config.video_input_enabled is False

    def test_legacy_env_vars_work(self, monkeypatch):
        """Test backward compatibility with legacy environment variables."""
        monkeypatch.setenv("TTS_VOICE", "legacy_voice")
        monkeypatch.setenv("TTS_SPEED", "1.5")
        monkeypatch.setenv("LLM_BASE_URL", "http://legacy:8080")
        monkeypatch.setenv("DEBUG", "true")

        config = VoiceAgentConfig.from_env()

        assert config.tts_voice == "legacy_voice"
        assert config.tts_speed == 1.5
        assert config.llm_base_url == "http://legacy:8080"
        assert config.debug_mode is True

    def test_memory_settings_load(self, monkeypatch):
        """Test that memory-specific settings load correctly."""
        monkeypatch.setenv("VOICE_AGENT_MEMORY_ENABLED", "false")
        monkeypatch.setenv("MEMORY_INJECT_HEADER", "Custom header:")
        monkeypatch.setenv("MEMORY_SOURCES", "convo,graph")

        config = VoiceAgentConfig.from_env()

        assert config.memory_enabled is False
        assert config.memory_inject_header == "Custom header:"
        assert config.memory_sources == "convo,graph"


class TestVoiceAgentConfigValidation:
    """Test configuration validation."""

    def test_validation_delegates_to_sections(self):
        """Test that validation checks all composed sections."""
        # Create config with invalid values
        config = VoiceAgentConfig()
        config.stt_engine = "invalid_engine"
        config.tts_engine = "invalid_tts"
        config.llm_context_max_tokens = 100  # Too low

        warnings = config.validate()

        # Should have warnings from multiple sections
        assert len(warnings) > 0
        assert any("stt" in w.lower() or "engine" in w.lower() for w in warnings)
        assert any("tts" in w.lower() or "engine" in w.lower() for w in warnings)
        assert any("context_max_tokens" in w.lower() for w in warnings)

    def test_valid_config_passes_validation(self):
        """Test that default config passes validation."""
        config = VoiceAgentConfig()
        warnings = config.validate()
        # Default config should have no critical errors
        assert len(warnings) == 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing codebase."""

    def test_get_component_config_stt(self):
        """Test get_component_config for STT returns correct dict."""
        config = VoiceAgentConfig()
        stt_config = config.get_component_config("stt")

        assert "engine" in stt_config
        assert "model" in stt_config
        assert "language" in stt_config
        assert stt_config["engine"] == "parakeet_streaming"

    def test_get_component_config_tts(self):
        """Test get_component_config for TTS returns correct dict."""
        config = VoiceAgentConfig()
        tts_config = config.get_component_config("tts")

        assert "engine" in tts_config
        assert "voice" in tts_config
        assert "speed" in tts_config
        assert tts_config["engine"] == "kokoro_mlx"

    def test_get_component_config_llm(self):
        """Test get_component_config for LLM returns correct dict."""
        config = VoiceAgentConfig()
        llm_config = config.get_component_config("llm")

        assert "base_url" in llm_config
        assert "model" in llm_config
        assert "max_tokens" in llm_config
        assert llm_config["model"] == "gemma3n:e2b"

    def test_get_component_config_memory(self):
        """Test get_component_config for memory returns correct dict."""
        config = VoiceAgentConfig()
        memory_config = config.get_component_config("memory")

        assert "enabled" in memory_config
        assert "hotpath_enabled" in memory_config
        assert memory_config["enabled"] is True

    def test_get_component_config_vision(self):
        """Test get_component_config for vision returns correct dict."""
        config = VoiceAgentConfig()
        vision_config = config.get_component_config("vision")

        assert "input_enabled" in vision_config
        assert "target_fps" in vision_config
        assert "image_size" in vision_config

    def test_summary_method_works(self):
        """Test that summary() method generates readable output."""
        config = VoiceAgentConfig()
        summary = config.summary()

        assert "Voice Agent Configuration Summary" in summary
        assert "STT:" in summary
        assert "TTS:" in summary
        assert "LLM:" in summary
        assert config.stt_engine in summary
        assert config.tts_engine in summary
        assert config.llm_model in summary

    def test_unknown_component_raises_error(self):
        """Test that unknown component raises ValueError."""
        config = VoiceAgentConfig()

        with pytest.raises(ValueError, match="Unknown component"):
            config.get_component_config("invalid_component")


class TestConfigIntegration:
    """Integration tests for the full configuration system."""

    def test_full_lifecycle(self, monkeypatch):
        """Test complete configuration lifecycle."""
        # Set environment variables
        monkeypatch.setenv("LLM_MODEL", "integration-model")
        monkeypatch.setenv("VOICE_AGENT_STT_ENGINE", "whisper_mlx")
        monkeypatch.setenv("TTS_VOICE", "integration_voice")

        # Load configuration
        config = VoiceAgentConfig.from_env()

        # Verify loaded correctly
        assert config.llm_model == "integration-model"
        assert config.stt_engine == "whisper_mlx"
        assert config.tts_voice == "integration_voice"

        # Modify via properties
        config.llm_temperature = 0.9
        assert config.llm.temperature == 0.9

        # Get component configs
        llm_config = config.get_component_config("llm")
        assert llm_config["model"] == "integration-model"
        assert llm_config["temperature"] == 0.9

        # Validate
        warnings = config.validate()
        # With valid values, should have no warnings
        assert len(warnings) == 0

        # Generate summary
        summary = config.summary()
        assert "integration-model" in summary
        assert "whisper_mlx" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
