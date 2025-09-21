"""
Whisper-MLX STT Service - Backup implementation for voice agent.
Uses MLX-optimized Whisper for Apple Silicon hardware.
"""

import asyncio
import time
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    StartFrame,
    EndFrame,
)
from pipecat.services.ai_services import STTService


class WhisperMLXSTTService(STTService):
    """
    Backup STT service using Whisper-MLX for Apple Silicon optimization.

    Features:
    - MLX-optimized inference for Apple Silicon
    - Configurable model size for latency/accuracy trade-offs
    - Streaming audio processing
    - Fallback for when Kyutai streaming is unavailable
    """

    def __init__(
        self,
        *,
        model: str = "openai/whisper-small",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._model = model
        self._language = language

        # Initialize MLX Whisper
        self._whisper = None
        self._initialize_whisper()

        logger.info(f"✅ Whisper-MLX STT initialized: model={model}")

    def _initialize_whisper(self):
        """Initialize MLX Whisper model."""
        try:
            import mlx_whisper

            logger.debug(f"Loading Whisper model: {self._model}")
            # Load model (this will cache it locally)
            self._whisper = mlx_whisper

            # Test the model
            import numpy as np
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            result = mlx_whisper.transcribe(test_audio, path_or_hf_repo=self._model)

            logger.debug("✅ Whisper-MLX model loaded and tested successfully")

        except ImportError:
            logger.error("MLX Whisper not available. Install with: pip install mlx-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Whisper-MLX: {e}")
            raise

    async def run_stt(self, audio: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        """Process audio frame and generate transcription."""

        if not self._whisper:
            logger.error("Whisper-MLX not initialized")
            return

        try:
            start_time = time.time()

            # Convert audio frame to numpy array
            import numpy as np
            audio_data = np.frombuffer(audio.audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe using MLX Whisper
            result = self._whisper.transcribe(
                audio_data,
                path_or_hf_repo=self._model,
                language=self._language
            )

            transcription_time = (time.time() - start_time) * 1000

            if result and "text" in result:
                text = result["text"].strip()

                if text:
                    logger.debug(f"🎯 Whisper-MLX transcription: '{text}' ({transcription_time:.1f}ms)")
                    yield TranscriptionFrame(text=text, user_id=audio.user_id, timestamp=audio.timestamp)
                else:
                    logger.debug("Empty transcription from Whisper-MLX")
            else:
                logger.warning("No transcription result from Whisper-MLX")

        except Exception as e:
            logger.error(f"Whisper-MLX STT error: {e}")


# Backward compatibility alias
class WhisperMLXService(WhisperMLXSTTService):
    """Alias for backward compatibility."""
    pass