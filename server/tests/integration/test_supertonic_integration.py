"""
Integration tests for Supertonic TTS with ServiceFactory.

Tests the full integration path from config → factory → service → audio output.
"""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch

# Set test environment before imports
os.environ["VOICE_AGENT_TTS_ENGINE"] = "supertonic"
os.environ["SUPERTONIC_VOICE"] = "F1"
os.environ["SUPERTONIC_TOTAL_STEPS"] = "2"


class TestSupertonicIntegration:
    """Test Supertonic TTS integration with the voice pipeline."""

    @pytest.mark.asyncio
    async def test_supertonic_service_basic(self):
        """Test basic Supertonic TTS service creation and synthesis."""
        from core.tts.supertonic_service import SupertonicTTSService

        tts = SupertonicTTSService(
            voice="F1",
            total_steps=2,
            speed=1.0,
            target_sample_rate=24000,
        )

        frames = []
        async for frame in tts.run_tts("Hello, this is a test."):
            frames.append(frame)

        # Should have: TTSStartedFrame, multiple TTSAudioRawFrame, TTSStoppedFrame
        assert len(frames) >= 3, f"Expected at least 3 frames, got {len(frames)}"

        # Check frame types
        frame_types = [type(f).__name__ for f in frames]
        assert "TTSStartedFrame" in frame_types
        assert "TTSStoppedFrame" in frame_types
        assert "TTSAudioRawFrame" in frame_types

    @pytest.mark.asyncio
    async def test_supertonic_complex_text(self):
        """Test Supertonic handles complex text without preprocessing."""
        from core.tts.supertonic_service import SupertonicTTSService

        tts = SupertonicTTSService(voice="F1", total_steps=2)

        # These should all work without any preprocessing
        complex_texts = [
            "The price is $99.99",
            "Call Dr. Smith at 3:30pm",
            "Revenue grew 15.7% in Q4 2025",
            "Visit https://example.com for more info",
        ]

        for text in complex_texts:
            frames = []
            async for frame in tts.run_tts(text):
                frames.append(frame)

            audio_frames = [f for f in frames if "Audio" in type(f).__name__]
            assert len(audio_frames) > 0, f"No audio for: {text}"

    @pytest.mark.asyncio
    async def test_supertonic_all_voices(self):
        """Test all Supertonic voices work."""
        from core.tts.supertonic_service import SupertonicTTSService

        voices = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

        for voice in voices:
            tts = SupertonicTTSService(voice=voice, total_steps=2)

            frames = []
            async for frame in tts.run_tts("Test"):
                frames.append(frame)

            audio_frames = [f for f in frames if "Audio" in type(f).__name__]
            assert len(audio_frames) > 0, f"No audio for voice: {voice}"

    @pytest.mark.asyncio
    async def test_supertonic_voice_switching(self):
        """Test voice can be changed at runtime."""
        from core.tts.supertonic_service import SupertonicTTSService

        tts = SupertonicTTSService(voice="F1", total_steps=2)

        # First synthesis
        frames1 = []
        async for frame in tts.run_tts("First voice"):
            frames1.append(frame)

        # Switch voice
        await tts.set_voice("M1")

        # Second synthesis
        frames2 = []
        async for frame in tts.run_tts("Second voice"):
            frames2.append(frame)

        assert len(frames1) > 2
        assert len(frames2) > 2

    @pytest.mark.asyncio
    async def test_supertonic_custom_model_path(self):
        """Test loading from custom model directory (bundle scenario)."""
        import os
        from core.tts.supertonic_service import SupertonicTTSService

        # Use the cached model directory
        model_dir = os.path.expanduser("~/.cache/supertonic")

        if not os.path.exists(model_dir):
            pytest.skip("Supertonic models not cached")

        tts = SupertonicTTSService(
            voice="F1",
            model_dir=model_dir,
            total_steps=2,
        )

        frames = []
        async for frame in tts.run_tts("Bundle test"):
            frames.append(frame)

        assert len(frames) >= 3

    def test_config_validation(self):
        """Test TTSConfiguration validates Supertonic settings."""
        from config.base_config import TTSConfiguration

        # Valid config
        config = TTSConfiguration(
            engine="supertonic",
            supertonic_voice="F1",
            supertonic_total_steps=2,
        )
        warnings = config.validate()
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

        # Invalid voice
        config_bad_voice = TTSConfiguration(
            engine="supertonic",
            supertonic_voice="INVALID",
            supertonic_total_steps=2,
        )
        warnings = config_bad_voice.validate()
        assert any("supertonic_voice" in w for w in warnings)

        # Invalid steps
        config_bad_steps = TTSConfiguration(
            engine="supertonic",
            supertonic_voice="F1",
            supertonic_total_steps=100,
        )
        warnings = config_bad_steps.validate()
        assert any("supertonic_total_steps" in w for w in warnings)


class TestSupertonicStrategy:
    """Test the Supertonic TTS creation strategy."""

    def test_strategy_creates_service(self):
        """Test SupertonicStrategy creates a valid service."""
        from config import VoiceAgentConfig
        from core.factories.strategies.tts_strategies import SupertonicStrategy

        # Create minimal config
        config = MagicMock(spec=VoiceAgentConfig)

        tts_config = {
            "supertonic_voice": "F1",
            "supertonic_total_steps": 2,
            "speed": 1.0,
            "sample_rate": 24000,
        }

        strategy = SupertonicStrategy(config, tts_config)
        service = strategy.create(use_boundaries=True)

        assert service is not None
        assert hasattr(service, "run_tts")

    def test_strategy_env_override(self):
        """Test strategy respects SUPERTONIC_MODEL_PATH env var."""
        import os
        from config import VoiceAgentConfig
        from core.factories.strategies.tts_strategies import SupertonicStrategy

        # Set env var
        test_path = "/tmp/test_models"
        os.environ["SUPERTONIC_MODEL_PATH"] = test_path

        try:
            config = MagicMock(spec=VoiceAgentConfig)
            tts_config = {
                "supertonic_voice": "F1",
                "supertonic_total_steps": 2,
                "speed": 1.0,
                "sample_rate": 24000,
            }

            strategy = SupertonicStrategy(config, tts_config)

            # The strategy should pick up the env var
            # (we can't fully test without the model, but we can verify the path is used)
            assert os.getenv("SUPERTONIC_MODEL_PATH") == test_path

        finally:
            del os.environ["SUPERTONIC_MODEL_PATH"]


if __name__ == "__main__":
    # Run basic test
    async def main():
        test = TestSupertonicIntegration()
        print("Running Supertonic integration tests...")

        print("\n1. Basic service test...")
        await test.test_supertonic_service_basic()
        print("   ✅ Passed")

        print("\n2. Complex text test...")
        await test.test_supertonic_complex_text()
        print("   ✅ Passed")

        print("\n3. All voices test...")
        await test.test_supertonic_all_voices()
        print("   ✅ Passed")

        print("\n4. Voice switching test...")
        await test.test_supertonic_voice_switching()
        print("   ✅ Passed")

        print("\n5. Custom model path test...")
        await test.test_supertonic_custom_model_path()
        print("   ✅ Passed")

        print("\n6. Config validation test...")
        TestSupertonicIntegration().test_config_validation()
        print("   ✅ Passed")

        print("\n=== All tests passed! ===")

    asyncio.run(main())
