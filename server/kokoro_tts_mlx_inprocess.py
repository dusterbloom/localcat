#
# Kokoro TTS service for Pipecat using MLX Audio (In-Process - UNSTABLE)
# WARNING: This implementation causes Metal threading conflicts and is unreliable
# Use kokoro_tts_mlx_isolated.py for production - this version needs fixes
#

import asyncio
import concurrent.futures
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

import mlx.core as mx
from mlx_audio.tts.utils import load_model

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KokoroTTSMLXInProcess(TTSService):
    """Kokoro TTS service implementation using MLX Audio (In-Process).

    Provides text-to-speech synthesis using Kokoro models running in-process
    on Apple Silicon through the mlx-audio library. Uses a separate thread
    for audio generation to avoid blocking the pipeline.
    
    Note: This in-process version may cause Metal threading conflicts.
    Use KokoroTTSIsolated for production stability.
    """

    def __init__(
        self,
        *,
        model: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        device: Optional[str] = None,
        sample_rate: int = 24000,
        max_workers: int = 1,
        **kwargs,
    ):
        """Initialize the Kokoro TTS service.

        Args:
            model: The Kokoro model to use (default: "mlx-community/Kokoro-82M-bf16").
            voice: The voice to use for synthesis (default: "af_heart").
            device: The device to run on (None for default MLX device).
            sample_rate: Output sample rate (default: 24000).
            max_workers: Number of threads for audio generation (default: 1).
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        # Explicitly set sample_rate (workaround for base class issue)
        self._sample_rate = sample_rate

        self._model_name = model
        self._voice = voice
        self._device = device
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Initialize model lazily to avoid threading issues
        self._model = None
        self._init_future = None

        self._settings = {
            "model": model,
            "voice": voice,
            "sample_rate": sample_rate,
        }

    def _initialize_model(self):
        """Initialize the Kokoro model. This runs in a separate thread."""
        try:
            logger.debug(f"Loading Kokoro model: {self._model_name}")
            self._model = load_model(self._model_name)
            logger.debug("Kokoro model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def can_generate_metrics(self) -> bool:
        return True

    def _generate_audio_sync(self, text: str) -> bytes:
        """Synchronously generate audio from text. This runs in a separate thread."""
        try:
            if self._model is None:
                self._initialize_model()  # Initialize synchronously in the worker thread

            logger.debug(f"Generating audio for: {text}")

            audio_segments = []
            for result in self._model.generate(
                text=text,
                voice=self._voice,
                speed=1.0,
            ):
                audio_segments.append(result.audio)

            if len(audio_segments) == 0:
                raise ValueError("No audio generated")
            elif len(audio_segments) == 1:
                audio_array = audio_segments[0]
            else:
                audio_array = mx.concatenate(audio_segments, axis=0)

            # Convert MLX array to NumPy array
            audio_np = np.array(audio_array, copy=False)

            # Convert to raw PCM bytes (16-bit signed integer)
            # MLX audio returns float32 normalized audio
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            return audio_bytes

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Run audio generation in executor (separate thread) to avoid blocking
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                self._executor, self._generate_audio_sync, text
            )

            # Chunk the audio data for streaming
            CHUNK_SIZE = self.chunk_size

            await self.stop_ttfb_metrics()

            # Stream the audio in chunks
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i : i + CHUNK_SIZE]
                if len(chunk) > 0:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    # Small delay to prevent overwhelming the pipeline
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        # Ensure model is initialized
        if self._model is None and self._init_future:
            await asyncio.get_event_loop().run_in_executor(None, self._init_future.result)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)
        await super().__aexit__(exc_type, exc_val, exc_tb)
