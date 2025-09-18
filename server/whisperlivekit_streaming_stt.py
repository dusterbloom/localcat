"""
WhisperLiveKit Streaming STT Service for Pipecat
Provides ultra-low latency streaming transcription using SimulStreaming backend
"""

import asyncio
import numpy as np
from typing import AsyncGenerator, Optional, List
from loguru import logger
import threading
import queue

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    ErrorFrame,
    AudioRawFrame
)
from pipecat.services.ai_services import STTService

try:
    from whisperlivekit import TranscriptionEngine
    from whisperlivekit.simul_whisper import SimulStreamingASR, SimulStreamingOnlineProcessor
    WHISPERLIVEKIT_AVAILABLE = True
except ImportError:
    logger.warning("WhisperLiveKit not available. Install with: pip install whisperlivekit")
    WHISPERLIVEKIT_AVAILABLE = False


class WhisperLiveKitStreamingSTT(STTService):
    """
    Streaming STT using WhisperLiveKit with SimulStreaming backend.
    Provides ultra-low latency transcription with intelligent buffering.
    """

    def __init__(
        self,
        *,
        model: str = "base",
        language: str = "en",
        backend: str = "simulstreaming",
        frame_threshold: int = 25,  # AlignAtt frame threshold
        chunk_size_ms: int = 100,  # Process every 100ms
        sample_rate: int = 16000,
        use_mlx_encoder: bool = True,  # Use MLX for Apple Silicon optimization
        **kwargs
    ):
        super().__init__(**kwargs)

        if not WHISPERLIVEKIT_AVAILABLE:
            raise ImportError("WhisperLiveKit is required but not installed")

        self._model = model
        self._language = language
        self._backend = backend
        self._frame_threshold = frame_threshold
        self._chunk_size_ms = chunk_size_ms
        self._sample_rate = sample_rate
        self._use_mlx_encoder = use_mlx_encoder

        # Calculate chunk size in samples
        self._chunk_samples = int(sample_rate * chunk_size_ms / 1000)

        # Audio buffer for accumulating chunks
        self._audio_buffer = []
        self._processing_lock = asyncio.Lock()

        # Initialize WhisperLiveKit components
        self._asr = None
        self._processor = None
        self._init_backend()

        logger.info(f"WhisperLiveKit STT initialized: {model} @ {chunk_size_ms}ms chunks")

    def _init_backend(self):
        """Initialize the WhisperLiveKit backend."""
        try:
            if self._backend == "simulstreaming":
                # Initialize SimulStreaming ASR with MLX encoder for Apple Silicon
                self._asr = SimulStreamingASR(
                    modelsize=self._model,
                    lan=self._language,
                    frame_threshold=self._frame_threshold,
                    segment_length=1.0,  # 1 second segments
                    fast_encoder=self._use_mlx_encoder  # Use MLX encoder on Apple Silicon
                )

                # Create online processor
                self._processor = SimulStreamingOnlineProcessor(
                    self._asr,
                    logfile=None  # No logging to file
                )

                # Warm up the model with silence
                warmup_audio = np.zeros(self._sample_rate, dtype=np.float32)
                self._processor.warmup(warmup_audio)

                logger.info("SimulStreaming backend initialized and warmed up")

            else:
                # Fallback to standard TranscriptionEngine
                from whisperlivekit import TranscriptionEngine
                self._asr = TranscriptionEngine(
                    model=self._model,
                    language=self._language,
                    backend=self._backend
                )
                logger.info(f"Standard {self._backend} backend initialized")

        except Exception as e:
            logger.error(f"Failed to initialize WhisperLiveKit backend: {e}")
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Process audio in streaming chunks using WhisperLiveKit.

        Args:
            audio: Raw audio bytes (16-bit PCM)

        Yields:
            TranscriptionFrame or InterimTranscriptionFrame
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer
            self._audio_buffer.extend(audio_np)

            # Process in chunks
            async with self._processing_lock:
                while len(self._audio_buffer) >= self._chunk_samples:
                    # Extract chunk
                    chunk = np.array(self._audio_buffer[:self._chunk_samples])
                    self._audio_buffer = self._audio_buffer[self._chunk_samples:]

                    # Process chunk with SimulStreaming
                    if self._processor:
                        # Insert audio chunk
                        audio_time = len(chunk) / self._sample_rate
                        self._processor.insert_audio_chunk(chunk, audio_time)

                        # Process and get results
                        tokens, processed_time = self._processor.process_iter(is_last=False)

                        # Convert tokens to text
                        if tokens:
                            text = "".join([token.word for token in tokens]).strip()

                            if text:
                                # Check if this is a complete utterance
                                is_final = any(token.is_final for token in tokens)

                                if is_final:
                                    # Final transcription
                                    yield TranscriptionFrame(
                                        text=text,
                                        timestamp=processed_time
                                    )
                                else:
                                    # Interim transcription
                                    yield InterimTranscriptionFrame(
                                        text=text,
                                        timestamp=processed_time
                                    )

                    # Small yield to prevent blocking
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"WhisperLiveKit processing error: {e}")
            yield ErrorFrame(error=str(e))

    async def process_frame(self, frame: Frame, direction=None):
        """
        Process incoming frames, handling audio data.
        """
        await super().process_frame(frame, direction)

        # Handle raw audio frames
        if isinstance(frame, AudioRawFrame):
            # Process audio through streaming STT
            async for result_frame in self.run_stt(frame.audio):
                await self.push_frame(result_frame)

    async def flush(self) -> AsyncGenerator[Frame, None]:
        """
        Flush any remaining audio in the buffer.
        """
        if self._audio_buffer and self._processor:
            try:
                # Process remaining audio as final
                remaining = np.array(self._audio_buffer)
                if len(remaining) > 0:
                    audio_time = len(remaining) / self._sample_rate
                    self._processor.insert_audio_chunk(remaining, audio_time)

                    # Process as final chunk
                    tokens, processed_time = self._processor.process_iter(is_last=True)

                    if tokens:
                        text = "".join([token.word for token in tokens]).strip()
                        if text:
                            yield TranscriptionFrame(
                                text=text,
                                timestamp=processed_time
                            )

                # Clear buffer
                self._audio_buffer = []

            except Exception as e:
                logger.error(f"Error flushing WhisperLiveKit buffer: {e}")
                yield ErrorFrame(error=str(e))

    async def cancel(self):
        """Cancel any ongoing processing."""
        self._audio_buffer = []
        if self._processor:
            # Reset the processor state
            self._processor.load_new_backend()

    def __del__(self):
        """Cleanup when service is destroyed."""
        if hasattr(self, '_processor') and self._processor:
            try:
                del self._processor
            except:
                pass
        if hasattr(self, '_asr') and self._asr:
            try:
                del self._asr
            except:
                pass