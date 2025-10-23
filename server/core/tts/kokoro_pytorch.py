"""
PyTorch-based Kokoro TTS Service using official hexgrad/Kokoro package.
Matches the minirepo implementation exactly.
"""

import asyncio
import concurrent.futures
import threading
import time
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from tools.text_formatter import split_text_for_kokoro_streaming
from tools.audio_utils import convert_to_pcm16

# CRITICAL: Import global Metal lock to serialize PyTorch initialization with MLX operations
from core.utils.mlx_lock import MLX_GLOBAL_LOCK


class KokoroPyTorchTTSService(TTSService):
    """
    PyTorch-based Kokoro TTS service using official package.
    Matches minirepo implementation for stability.
    """

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000,
        repo_id: str = "hexgrad/Kokoro-82M",
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,
            **kwargs
        )

        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate
        self._repo_id = repo_id

        # Thread pool for non-blocking TTS generation
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="kokoro-pytorch"
        )

        # CRITICAL: Lock to prevent concurrent access (same as minirepo)
        self._generation_lock = threading.Lock()

        # Initialize Kokoro pipeline
        self._pipeline = None
        self._initialize_pipeline()

        logger.debug(f"✅ Kokoro PyTorch TTS initialized with voice: {self._voice}")

    def can_generate_metrics(self) -> bool:
        return True

    def _initialize_pipeline(self):
        """Initialize the Kokoro pipeline with global Metal lock to prevent concurrent access.

        CRITICAL: Uses MLX_GLOBAL_LOCK to serialize PyTorch Metal initialization with MLX operations
        (Parakeet STT). Without this, macOS Sequoia kills the process due to concurrent Metal access.
        """
        try:
            from kokoro import KPipeline

            logger.info(f"🔒 Acquiring global Metal lock for Kokoro PyTorch initialization...")

            # CRITICAL: Acquire the same lock used by Parakeet MLX STT
            # This serializes: Parakeet load → Kokoro load (never concurrent)
            with MLX_GLOBAL_LOCK:
                logger.debug(f"🚀 Initializing Kokoro PyTorch TTS with voice: {self._voice}")

                # Create pipeline (same as minirepo)
                self._pipeline = KPipeline(lang_code='a', repo_id=self._repo_id)

                logger.debug(f"✅ Kokoro PyTorch pipeline loaded successfully")
                logger.info("Kokoro PyTorch TTS ready")

        except ImportError as e:
            logger.error(f"Kokoro package not available: {e}")
            logger.error("Install with: pip install kokoro>=0.9.2")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro PyTorch: {e}")
            self._pipeline = None
            raise

    def _generate_audio_sync(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synchronous audio generation using PyTorch Kokoro - runs in thread pool"""
        if not self._pipeline:
            logger.error("Kokoro PyTorch pipeline not initialized")
            return None

        try:
            start_time = time.time()

            # CRITICAL: Run generation under lock (same as minirepo)
            with self._generation_lock:
                # Generate audio using KPipeline (same as minirepo)
                audio_chunks = []

                for result in self._pipeline(text, voice=self._voice, speed=self._speed):
                    if result.audio is not None:
                        audio_chunks.append(result.audio)

                if not audio_chunks:
                    logger.warning("No audio data generated from Kokoro pipeline")
                    return None

                # Concatenate chunks (same as minirepo with torch.cat)
                if len(audio_chunks) == 1:
                    audio_tensor = audio_chunks[0]
                else:
                    audio_tensor = torch.cat(audio_chunks, dim=0)

                # Convert to numpy
                audio_np = audio_tensor.numpy()

            generation_time = time.time() - start_time

            # Log performance
            chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
            logger.debug(f"Kokoro PyTorch generated {len(text)} chars in {generation_time:.3f}s ({chars_per_sec:.1f} chars/s)")

            return audio_np, self._sample_rate

        except Exception as e:
            logger.error(f"Kokoro PyTorch TTS generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro PyTorch"""

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # Split text into optimal chunks
        sentences = split_text_for_kokoro_streaming(text, min_length=50, max_length=120)

        if not sentences:
            logger.debug(f"🔇 Skipping TTS for empty text: '{text}'")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        logger.debug(f"🎤 Kokoro PyTorch TTS for {len(sentences)} chunks")

        overall_start_time = time.time()
        first_audio_sent = False

        # Start metrics tracking
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        yield TTSStartedFrame()

        try:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                chunk_start_time = time.time()

                # Generate audio
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._generate_audio_sync,
                    sentence
                )

                if result is not None:
                    audio_data, actual_sample_rate = result

                    if audio_data is not None and len(audio_data) > 0:
                        # Calculate TTFB for first chunk only
                        if not first_audio_sent:
                            ttfb = (time.time() - overall_start_time) * 1000
                            logger.debug(f"🚀 Kokoro PyTorch TTFB: {ttfb:.1f}ms")
                            await self.stop_ttfb_metrics()
                            first_audio_sent = True

                        chunk_latency = (time.time() - chunk_start_time) * 1000
                        logger.debug(f"✅ Chunk {i+1}/{len(sentences)}: {len(sentence)} chars → {chunk_latency:.1f}ms")

                        # Convert to int16 for Pipecat (PyTorch audio needs clipping for safety)
                        audio_int16 = convert_to_pcm16(audio_data, clip=True)

                        frame = TTSAudioRawFrame(
                            audio=audio_int16.tobytes(),
                            sample_rate=actual_sample_rate,
                            num_channels=1
                        )

                        yield frame
                    else:
                        logger.warning(f"Empty audio data for chunk: '{sentence}'")
                else:
                    logger.warning(f"No audio generated for chunk: '{sentence}'")

        except asyncio.CancelledError:
            logger.debug("Kokoro PyTorch TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Kokoro PyTorch TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            # Stop metrics tracking
            await self.stop_processing_metrics()
            if not first_audio_sent:
                await self.stop_ttfb_metrics()

            # Send usage metrics
            await self.start_tts_usage_metrics(text)

            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)
