"""
Unit tests for base configuration classes.
"""

import os
import pytest
from config.base_config import (
    BaseConfiguration,
    ConfigurationError,
    LLMConfiguration,
    STTConfiguration,
    TTSConfiguration,
    VisionConfiguration,
)


class TestLLMConfiguration:
    """Test LLM configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfiguration()
        assert config.base_url == "http://127.0.0.1:11434/v1"
        assert config.model == "gemma3n:e2b"
        assert config.context_max_tokens == 3000
        assert config.context_prune_threshold == 0.70
        assert config.use_streaming is True

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
        monkeypatch.setenv("LLM_CONTEXT_MAX_TOKENS", "4000")

        config = LLMConfiguration.from_env()
        assert config.base_url == "http://localhost:8080/v1"
        assert config.model == "test-model"
        assert config.max_tokens == 2048
        assert config.temperature == 0.9
        assert config.context_max_tokens == 4000

    def test_legacy_env_vars(self, monkeypatch):
        """Test legacy environment variable support."""
        monkeypatch.setenv("VOICE_AGENT_LLM_BASE_URL", "http://legacy:8080/v1")
        monkeypatch.setenv("VOICE_AGENT_LLM_MODEL", "legacy-model")

        config = LLMConfiguration.from_env()
        assert config.base_url == "http://legacy:8080/v1"
        assert config.model == "legacy-model"

    def test_validation_warnings(self):
        """Test configuration validation."""
        # Valid config
        config = LLMConfiguration()
        warnings = config.validate()
        assert len(warnings) == 0

        # Invalid context_max_tokens
        config = LLMConfiguration(context_max_tokens=500)
        warnings = config.validate()
        assert any("context_max_tokens" in w for w in warnings)

        # Invalid prune threshold
        config = LLMConfiguration(context_prune_threshold=1.5)
        warnings = config.validate()
        assert any("context_prune_threshold" in w for w in warnings)

        # Invalid temperature
        config = LLMConfiguration(temperature=3.0)
        warnings = config.validate()
        assert any("temperature" in w for w in warnings)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LLMConfiguration(model="test-model", max_tokens=1024)
        config_dict = config.to_dict()
        assert config_dict["model"] == "test-model"
        assert config_dict["max_tokens"] == 1024


class TestSTTConfiguration:
    """Test STT configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = STTConfiguration()
        assert config.engine == "parakeet_streaming"
        assert config.model == "mlx-community/parakeet-tdt-0.6b-v3"
        assert config.language == "en"
        assert config.chunk_length_ms == 100

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("VOICE_AGENT_STT_ENGINE", "whisper_mlx")
        monkeypatch.setenv("VOICE_AGENT_STT_LANGUAGE", "es")
        monkeypatch.setenv("VOICE_AGENT_STT_CHUNK_LENGTH_MS", "200")

        config = STTConfiguration.from_env()
        assert config.engine == "whisper_mlx"
        assert config.language == "es"
        assert config.chunk_length_ms == 200

    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = STTConfiguration()
        warnings = config.validate()
        assert len(warnings) == 0

        # Invalid engine
        config = STTConfiguration(engine="invalid_engine")
        warnings = config.validate()
        assert any("engine" in w for w in warnings)

        # Invalid chunk length
        config = STTConfiguration(chunk_length_ms=2000)
        warnings = config.validate()
        assert any("chunk_length_ms" in w for w in warnings)


class TestTTSConfiguration:
    """Test TTS configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfiguration()
        assert config.engine == "kokoro_mlx"
        assert config.voice == "af_heart"
        assert config.speed == 1.0
        assert config.sample_rate == 24000
        assert config.chunk_size_chars == 25

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("TTS_ENGINE", "kokoro_professional")
        monkeypatch.setenv("TTS_VOICE", "custom_voice")
        monkeypatch.setenv("TTS_SPEED", "1.2")
        monkeypatch.setenv("TTS_CHUNK_SIZE_CHARS", "30")

        config = TTSConfiguration.from_env()
        assert config.engine == "kokoro_professional"
        assert config.voice == "custom_voice"
        assert config.speed == 1.2
        assert config.chunk_size_chars == 30

    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = TTSConfiguration()
        warnings = config.validate()
        assert len(warnings) == 0

        # Invalid engine
        config = TTSConfiguration(engine="invalid_engine")
        warnings = config.validate()
        assert any("engine" in w for w in warnings)

        # Invalid speed
        config = TTSConfiguration(speed=3.0)
        warnings = config.validate()
        assert any("speed" in w for w in warnings)

        # Invalid chunk size
        config = TTSConfiguration(chunk_size_chars=100)
        warnings = config.validate()
        assert any("chunk_size_chars" in w for w in warnings)


class TestVisionConfiguration:
    """Test Vision configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VisionConfiguration()
        assert config.input_enabled is True
        assert config.target_fps == 0.5
        assert config.image_size == 384
        assert config.max_images_in_context == 2
        assert config.keyword_filter is True

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("VIDEO_INPUT_ENABLED", "true")
        monkeypatch.setenv("VIDEO_TARGET_FPS", "1.0")
        monkeypatch.setenv("VISION_IMAGE_SIZE", "512")
        monkeypatch.setenv("VISION_MAX_IMAGES_IN_CONTEXT", "3")

        config = VisionConfiguration.from_env()
        assert config.input_enabled is True
        assert config.target_fps == 1.0
        assert config.image_size == 512
        assert config.max_images_in_context == 3

    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = VisionConfiguration()
        warnings = config.validate()
        assert len(warnings) == 0

        # Invalid image quality
        config = VisionConfiguration(image_quality=150)
        warnings = config.validate()
        assert any("image_quality" in w for w in warnings)

        # Invalid max images
        config = VisionConfiguration(max_images_in_context=0)
        warnings = config.validate()
        assert any("max_images_in_context" in w for w in warnings)

        # Invalid FPS
        config = VisionConfiguration(target_fps=20.0)
        warnings = config.validate()
        assert any("target_fps" in w for w in warnings)


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_raise_configuration_error(self):
        """Test raising configuration error."""
        with pytest.raises(ConfigurationError, match="Test error"):
            raise ConfigurationError("Test error")

    def test_validate_and_raise_with_invalid_config(self):
        """Test validate_and_raise with invalid configuration."""
        config = LLMConfiguration(context_max_tokens=100)  # Too low
        with pytest.raises(ConfigurationError, match="validation failed"):
            config.validate_and_raise()

    def test_validate_and_raise_with_valid_config(self):
        """Test validate_and_raise with valid configuration."""
        config = LLMConfiguration()
        # Should not raise
        config.validate_and_raise()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
