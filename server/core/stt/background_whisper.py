"""
Background Whisper Processor - Parallel STT processing for sub-second latency.
Runs Whisper-MLX in background asyncio task to avoid blocking main pipeline.
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
from pipecat.processors.frame_processor import FrameProcessor


class BackgroundWhisperProcessor(FrameProcessor):
    """
    Background Whisper processor for parallel STT processing.

    Runs Whisper transcription in separate asyncio task to prevent
    blocking the main pipeline, enabling sub-second end-to-end latency.
    """

    def __init__(self, model: str = "openai/whisper-small", language: str = "en", **kwargs):
        super().__init__(**kwargs)

        self._model = model
        self._language = language
        self._whisper = None
        self._processing_task = None
        self._audio_queue = asyncio.Queue()
        self._running = False

        # Initialize MLX Whisper
        self._initialize_whisper()

        logger.info(f"✅ Background Whisper initialized: model={model}, language={language}")

    def _initialize_whisper(self):
        """Initialize MLX Whisper model."""
        try:
            import mlx_whisper

            logger.debug(f"Loading Whisper model: {self._model}")
            # Test the model
            import numpy as np
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            result = mlx_whisper.transcribe(test_audio, path_or_hf_repo=self._model)

            self._whisper = mlx_whisper
            logger.debug("✅ Background Whisper model loaded and tested")

        except ImportError:
            logger.error("MLX Whisper not available. Install with: pip install mlx-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Background Whisper: {e}")
            raise

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        # Start background processing on first frame
        if not self._running and isinstance(frame, StartFrame):
            self._running = True
            self._processing_task = asyncio.create_task(self._process_audio_queue())
            logger.debug("🎯 Background Whisper processing started")

        # Stop background processing on end
        if isinstance(frame, EndFrame):
            self._running = False
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            logger.debug("🛑 Background Whisper processing stopped")

        # Queue audio frames for background processing
        if isinstance(frame, AudioRawFrame):
            await self._audio_queue.put(frame)

        # Always pass frame through
        await self.push_frame(frame, direction)

    async def _process_audio_queue(self):
        """Background task to process queued audio."""
        while self._running:
            try:
                # Wait for audio frame with timeout
                audio_frame = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                await self._transcribe_audio(audio_frame)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background Whisper processing error: {e}")

    async def _transcribe_audio(self, audio_frame: AudioRawFrame):
        """Transcribe audio frame in background."""
        if not self._whisper:
            return

        try:
            start_time = time.time()

            # Convert audio frame to numpy array
            import numpy as np
            audio_data = np.frombuffer(audio_frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe using MLX Whisper
            result = self._whisper.transcribe(
                audio_data,
                path_or_hf_repo=self._model,
                language=self._language
            )

            transcription_time = (time.time() - start_time) * 1000

            if result and "text" in result:
                text = str(result["text"]).strip()

                if text:
                    logger.debug(f"🎯 Background Whisper transcription: '{text}' ({transcription_time:.1f}ms)")

                    # Create and push transcription frame
                    transcription_frame = TranscriptionFrame(
                        text=text,
                        user_id="background_whisper",  # Default user ID
                        timestamp=str(int(time.time() * 1000))  # Current timestamp as string
                    )
                    await self.push_frame(transcription_frame)
                else:
                    logger.debug("Empty transcription from Background Whisper")
            else:
                logger.warning("No transcription result from Background Whisper")

        except Exception as e:
            logger.error(f"Background Whisper transcription error: {e}")