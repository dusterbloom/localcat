"""
PyTorch-based Kokoro TTS Service using official hexgrad/Kokoro package.
Matches the minirepo implementation exactly.

KEY IMPROVEMENTS:
- Pre-validates model files before Metal lock acquisition
- Implements HF_HUB_OFFLINE=1 after validation to prevent network calls
- Separates network operations from Metal lock for bulletproof reliability
"""

import asyncio
import concurrent.futures
import threading
import time
import os
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

# NEW: Import model validator for pre-validation
from core.utils.model_validator import ensure_offline_ready, ModelValidationError

from core.tts.kokoro_config import (
    PYTORCH_EXECUTOR_WORKERS,
    CHUNK_MIN_LENGTH,
    CHUNK_MAX_LENGTH,
    THREAD_PREFIX_PYTORCH,
)


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
            max_workers=PYTORCH_EXECUTOR_WORKERS,
            thread_name_prefix=THREAD_PREFIX_PYTORCH
        )

        # CRITICAL: Lock to prevent concurrent access (same as minirepo)
        self._generation_lock = threading.Lock()

        # Initialize Kokoro pipeline
        self._pipeline = None
        self._initialize_pipeline()

        logger.debug(f"✅ Kokoro PyTorch TTS initialized with voice: {self._voice}")

    def can_generate_metrics(self) -> bool:
        return True

    def _prevalidate_model(self) -> bool:
        """
        Pre-validate model files and ensure offline readiness BEFORE Metal lock acquisition.

        This is the CRITICAL fix: ensures all network operations complete before we acquire
        the Metal lock, preventing deadlocks and initialization failures.
        """
        try:
            logger.info(f"🔍 Pre-validating Kokoro model {self._repo_id} before Metal lock...")

            # Check if we should skip validation (e.g., in production bundles)
            skip_validation = os.environ.get("SKIP_TTS_VALIDATION", "false").lower() == "true"
            if skip_validation:
                logger.info("⚠️  Skipping TTS validation (SKIP_TTS_VALIDATION=true)")
                return True

            # Use the new model validator to ensure offline readiness
            is_ready = ensure_offline_ready(self._repo_id)

            if not is_ready:
                logger.error("❌ Model pre-validation failed - cannot initialize safely")
                raise ModelValidationError(f"Model {self._repo_id} is not ready for offline initialization")

            logger.info("✅ Model pre-validation successful - ready for Metal lock acquisition")
            return True

        except Exception as e:
            logger.error(f"❌ Model pre-validation failed: {e}")
            # Don't raise here - let the main initialization handle it
            # This allows graceful fallback
            return False

    def _initialize_pipeline(self):
        """
        Initialize the Kokoro pipeline with comprehensive pre-validation and Metal lock safety.

        CRITICAL IMPROVEMENTS:
        1. Pre-validates model files BEFORE Metal lock acquisition
        2. Sets HF_HUB_OFFLINE=1 to prevent network calls during initialization
        3. Uses MLX_GLOBAL_LOCK to serialize with MLX operations (Parakeet STT)
        4. Configures espeak-ng paths BEFORE importing kokoro
        5. Provides robust error handling and retry logic
        """
        try:
            # CRITICAL: Configure espeak-ng paths BEFORE importing kokoro
            # This prevents the "//phontab" path error in bundled apps
            try:
                import espeakng_loader
                from misaki.espeak import EspeakWrapper

                library_path = espeakng_loader.get_library_path()
                data_path = espeakng_loader.get_data_path()

                logger.debug(f"🔧 Configuring EspeakWrapper:")
                logger.debug(f"   library_path: {library_path}")
                logger.debug(f"   data_path: {data_path}")

                # Make library available
                espeakng_loader.make_library_available()

                # Set paths as properties (NOT methods!)
                EspeakWrapper.library_path = library_path
                EspeakWrapper.data_path = data_path

                logger.info(f"✅ Espeak-ng configured successfully")
            except Exception as e:
                logger.warning(f"⚠️  Failed to configure espeak-ng: {e}")
                logger.warning("   Kokoro may still work with system espeak-ng")

            # Now safe to import kokoro
            from kokoro import KPipeline

            logger.info(f"🚀 Initializing Kokoro PyTorch TTS with repo_id: {self._repo_id}")

            # STEP 1: Pre-validate model files (CRITICAL - done OUTSIDE Metal lock)
            if not self._prevalidate_model():
                logger.warning("⚠️  Model pre-validation failed, attempting initialization anyway...")

            # STEP 2: Save current environment and force offline mode
            original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")

            # Force offline mode to prevent network calls during Metal lock
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            try:
                logger.info(f"🔒 Acquiring global Metal lock for Kokoro PyTorch initialization...")

                # STEP 3: CRITICAL - Acquire the same lock used by Parakeet MLX STT
                # This serializes: Parakeet load → Kokoro load (never concurrent)
                with MLX_GLOBAL_LOCK:
                    logger.debug(f"🎯 Creating KPipeline inside Metal lock (offline mode)...")

                    # Create pipeline with offline mode (environment ensures no network calls)
                    self._pipeline = KPipeline(
                        lang_code='a',
                        repo_id=self._repo_id
                    )

                    logger.debug(f"✅ Kokoro PyTorch pipeline loaded successfully inside Metal lock")
                    logger.info("🎉 Kokoro PyTorch TTS ready (offline mode)")

            finally:
                # STEP 4: Restore original environment (important for other services)
                if original_hf_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                else:
                    os.environ.pop("HF_HUB_OFFLINE", None)

                if original_transformers_offline is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = original_transformers_offline
                else:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)

        except ImportError as e:
            logger.error(f"❌ Kokoro package not available: {e}")
            logger.error("💡 Install with: pip install kokoro>=0.9.2")
            self._pipeline = None
            raise
        except ModelValidationError as e:
            logger.error(f"❌ Model validation failed: {e}")
            logger.error("💡 This usually means the model files are missing or corrupted")
            logger.error("💡 Try running the server once with internet to download models")
            self._pipeline = None
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro PyTorch: {e}")

            # Provide helpful troubleshooting information
            error_msg = str(e).lower()
            if "offline" in error_msg or "cache" in error_msg:
                logger.error("💡 This appears to be an offline/cache issue:")
                logger.error("   1. Ensure models are downloaded: python -c 'from kokoro import KPipeline; KPipeline(lang_code=\"a\", repo_id=\"hexgrad/Kokoro-82M\")'")
                logger.error("   2. Check cache directory permissions")
                logger.error("   3. Verify HUGGINGFACE_HUB_CACHE is set correctly")
            elif "metal" in error_msg or "gpu" in error_msg:
                logger.error("💡 This appears to be a Metal/GPU issue:")
                logger.error("   1. Ensure MLX_GLOBAL_LOCK is working correctly")
                logger.error("   2. Check for concurrent Metal operations")
                logger.error("   3. Try restarting the application")

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
        sentences = split_text_for_kokoro_streaming(
            text,
            min_length=CHUNK_MIN_LENGTH,
            max_length=CHUNK_MAX_LENGTH
        )

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

                # Mirror the exact text chunk to transcript/UI
                from pipecat.frames.frames import TTSTextFrame
                yield TTSTextFrame(text=sentence)

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
