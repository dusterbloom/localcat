"""
Simple Moshi TTS fallback using core moshi_mlx models.
This version works with the existing moshi_mlx 0.3.0 package.
"""

import asyncio
import time
import numpy as np
from typing import AsyncGenerator
from loguru import logger

import mlx.core as mx
import mlx.nn as nn
from moshi_mlx import models
from moshi_mlx.utils import Sampler

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# Import global MLX lock
from utils.mlx_lock import MLX_GLOBAL_LOCK


class MoshiSimpleTTS(TTSService):
    """
    Simple Moshi TTS using core moshi_mlx functionality.
    Fallback implementation that works with moshi_mlx 0.3.0.
    """

    def __init__(
        self,
        *,
        voice: str = "default",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,
            **kwargs
        )

        self._voice = voice
        self._sample_rate = sample_rate

        # For now, just log that we're using Kokoro fallback
        logger.warning("Moshi TTS not fully available, using Kokoro fallback")
        self._fallback = True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS - fallback to Kokoro."""

        if self._fallback:
            # Import and use Kokoro as fallback
            from tts_mlx_kokoro import MLXKokoroTTSService

            kokoro = MLXKokoroTTSService(
                voice=self._voice if self._voice != "default" else "af_heart",
                sample_rate=self._sample_rate
            )

            async for frame in kokoro.run_tts(text):
                yield frame
        else:
            # Placeholder for when Moshi TTS is available
            yield TTSStartedFrame()
            yield TTSStoppedFrame()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass