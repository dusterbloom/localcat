"""
Hybrid TTS Service - Automatic switching between Piper and Kokoro.
Uses Piper for fast conversational responses, Kokoro for quality.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class HybridTTService(TTSService):
    """
    Hybrid TTS service that automatically switches between Piper and Kokoro
    based on text length and complexity for optimal latency vs quality.
    """

    def __init__(
        self,
        *,
        piper_voice: str = "en_US-lessac-medium",
        kokoro_voice: str = "af_heart",
        switch_threshold: int = 80,  # Characters
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._piper_voice = piper_voice
        self._kokoro_voice = kokoro_voice
        self._switch_threshold = switch_threshold

        # Lazy load TTS services
        self._piper_tts = None
        self._kokoro_tts = None

        logger.debug(f"✅ Hybrid TTS initialized (threshold: {switch_threshold} chars)")

    def _get_piper_tts(self):
        """Lazy load Piper TTS."""
        if self._piper_tts is None:
            try:
                from tts_piper_streaming import PiperStreamingTTS
                self._piper_tts = PiperStreamingTTS(
                    voice=self._piper_voice,
                    sample_rate=22050,
                )
                logger.debug("✅ Piper TTS loaded for hybrid service")
            except ImportError as e:
                logger.error(f"❌ Failed to load Piper TTS: {e}")
                raise
        return self._piper_tts

    def _get_kokoro_tts(self):
        """Lazy load Kokoro TTS."""
        if self._kokoro_tts is None:
            try:
                from tts_native_kokoro import NativeKokoroTTSService
                self._kokoro_tts = NativeKokoroTTSService(
                    voice=self._kokoro_voice,
                    speed=1.0,
                    sample_rate=24000
                )
                logger.debug("✅ Kokoro TTS loaded for hybrid service")
            except ImportError as e:
                logger.warning(f"Native Kokoro not available, falling back to MLX: {e}")
                try:
                    from tts_mlx_kokoro import MLXKokoroTTSService
                    self._kokoro_tts = MLXKokoroTTSService(
                        voice=self._kokoro_voice,
                        speed=1.0,
                        sample_rate=24000
                    )
                    logger.debug("✅ MLX Kokoro TTS loaded as fallback")
                except ImportError as e2:
                    logger.error(f"❌ No Kokoro TTS available: {e2}")
                    raise e2
        return self._kokoro_tts

    def _should_use_piper(self, text: str) -> bool:
        """Determine if text should use Piper (speed) or Kokoro (quality)."""
        text_length = len(text.strip())

        # Use Piper for short conversational responses
        if text_length < self._switch_threshold:
            return True

        # Use Piper for simple sentences, Kokoro for complex content
        # Count sentences - if it's a single short sentence, use Piper
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) == 1 and text_length < 120:
            return True

        # Use Kokoro for longer or more complex content
        return False

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using optimal TTS engine for the content."""

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # Choose optimal TTS engine
        use_piper = self._should_use_piper(text)

        if use_piper:
            tts_service = self._get_piper_tts()
            engine_name = "Piper"
        else:
            tts_service = self._get_kokoro_tts()
            engine_name = "Kokoro"

        logger.debug(f"🎯 Hybrid TTS: Using {engine_name} for '{text[:50]}{'...' if len(text) > 50 else ''}'")

        try:
            yield TTSStartedFrame()

            # Delegate to the chosen TTS service
            async for frame in tts_service.run_tts(text):
                yield frame

        except Exception as e:
            logger.error(f"❌ Hybrid TTS error with {engine_name}: {e}")
            yield ErrorFrame(error=f"TTS failed: {str(e)}")
        finally:
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Clean up TTS services if they exist
        if self._piper_tts and hasattr(self._piper_tts, '__aexit__'):
            await self._piper_tts.__aexit__(exc_type, exc_val, exc_tb)
        if self._kokoro_tts and hasattr(self._kokoro_tts, '__aexit__'):
            await self._kokoro_tts.__aexit__(exc_type, exc_val, exc_tb)
        await super().__aexit__(exc_type, exc_val, exc_tb)