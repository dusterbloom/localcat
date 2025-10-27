"""
MLX-based Kokoro TTS Service for Apple Silicon optimization.
Replaces ONNX implementation with native MLX for 5-10x performance improvement.
"""

import asyncio
import concurrent.futures
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from tools.text_formatter import (
    split_text_for_kokoro_streaming,
    chunk_for_kokoro_ultra_low_latency,
)
from tools.audio_utils import convert_to_pcm16
from core.utils.mlx_lock import MLX_GLOBAL_LOCK
from core.tts.kokoro_config import (
    MLX_EXECUTOR_WORKERS,
    CHUNK_MIN_LENGTH,
    CHUNK_MAX_LENGTH,
    THREAD_PREFIX_MLX,
)


class MLXKokoroTTSService(TTSService):
    """
    MLX-based Kokoro TTS service optimized for Apple Silicon.

    DESIGN (fixes battery drain):
    - SINGLETON PATTERN: Shared pipeline across all instances (no multiple models)
    - SERIALIZED GENERATION: Lock during TTS to prevent concurrent STT+TTS (MLX requirement)
    - LOCKED INITIALIZATION: Serializes model loading with STT

    MLX is not thread-safe for concurrent operations. Lock is required.
    """

    # Shared MLX pipeline across all instances (singleton pattern)
    _shared_pipeline = None
    _pipeline_init_lock = None

    def __init__(
        self,
        *,
        voice: str = "af_bella",
        speed: float = 1.0,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,  # Use sentence aggregation for smooth flow
            **kwargs
        )

        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate

        # Cancellation flag for graceful interruption handling
        self._cancelled = False

        # Thread pool for non-blocking TTS generation
        # Single worker since MLX operations must be serialized via lock
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=MLX_EXECUTOR_WORKERS,
            thread_name_prefix=THREAD_PREFIX_MLX
        )

        # Initialize shared MLX Kokoro pipeline (only once across all instances)
        self._initialize_mlx_pipeline()

        logger.info(f"✅ MLX Kokoro TTS initialized (SINGLETON) with voice: {self._voice}")

    def _initialize_mlx_pipeline(self):
        """Initialize the SHARED MLX Kokoro pipeline with singleton pattern.

        DESIGN:
        - SINGLETON: Only ONE model instance across all service instances
        - LOCKED INIT: Serializes model loading with STT (prevents concurrent Metal access)
        - SERIALIZED GENERATION: Lock also used during generation (MLX not thread-safe)

        This prevents battery drain (singleton fixes 6x model instances).
        """
        # Initialize lock reference on first instantiation
        if MLXKokoroTTSService._pipeline_init_lock is None:
            MLXKokoroTTSService._pipeline_init_lock = MLX_GLOBAL_LOCK  # Share with STT for serialized init

        # Check if pipeline is already initialized (singleton fast path)
        if MLXKokoroTTSService._shared_pipeline is not None:
            logger.debug("✅ Using existing shared MLX Kokoro pipeline (singleton)")
            return

        try:
            logger.info(f"🔒 Acquiring MLX lock for Kokoro initialization (singleton, generation also locked)...")
            with MLXKokoroTTSService._pipeline_init_lock:
                # Double-check after acquiring lock (thread-safe singleton pattern)
                if MLXKokoroTTSService._shared_pipeline is not None:
                    logger.debug("Pipeline already initialized by another instance")
                    return

                # CRITICAL: Disable espeak-ng initialization to use Misaki-only G2P
                # mlx-audio Kokoro uses Misaki as primary G2P (handles 99% of English)
                # Espeak is only a fallback for rare out-of-vocabulary words - we can skip it
                # This fixes phontab error by preventing espeak initialization at import time
                #
                # STRATEGY: Patch espeakng_loader BEFORE misaki.espeak imports it
                # misaki.espeak calls these functions at module import time
                try:
                    import espeakng_loader
                    # Return dummy safe paths to prevent espeak-ng from loading data files
                    espeakng_loader.get_library_path = lambda: ""
                    espeakng_loader.get_data_path = lambda: "/tmp"
                    logger.debug("✅ Patched espeakng_loader to return dummy paths (using Misaki-only G2P)")
                except ImportError:
                    logger.debug("⚠️  espeakng_loader not available, espeak disabled by default")

                from mlx_audio.tts.utils import load_model
                from mlx_audio.tts.models.kokoro import KokoroPipeline

                logger.debug(f"🚀 Loading SHARED MLX Kokoro model: mlx-community/Kokoro-82M-bf16")

                # Load model
                model = load_model("mlx-community/Kokoro-82M-bf16")

                # CRITICAL: Pre-create KokoroPipeline during initialization (under lock)
                # This prevents lazy creation during first .generate() call (which would be after Parakeet loads)
                # Lazy creation allocates Metal resources without lock protection → macOS Sequoia kills process
                logger.debug("Pre-creating KokoroPipeline to prevent lazy Metal allocation during first TTS...")
                pipeline = KokoroPipeline(model=model, repo_id="mlx-community/Kokoro-82M-bf16", lang_code='a')

                # Store model AND pipeline as tuple
                MLXKokoroTTSService._shared_pipeline = (model, pipeline)

                logger.debug(f"✅ Shared MLX Kokoro model and pipeline loaded successfully")
                logger.info("🎯 MLX Kokoro TTS ready (SINGLETON, serialized generation)")

        except ImportError as e:
            logger.error(f"MLX Audio not available: {e}")
            logger.error("Install with: pip install mlx-audio")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize MLX Kokoro: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _generate_audio_sync(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synchronous audio generation using SHARED MLX pipeline

        CRITICAL: Uses MLX_GLOBAL_LOCK to prevent concurrent Metal access during generation.
        macOS Sequoia kills processes when STT and TTS access Metal simultaneously.
        """
        if not MLXKokoroTTSService._shared_pipeline:
            logger.error("Shared MLX Kokoro pipeline not initialized")
            return None

        try:
            start_time = time.time()

            logger.debug(f"Generating audio for: {text}")

            # CRITICAL: Acquire MLX lock to prevent concurrent STT + TTS access
            # This is REQUIRED - MLX is not thread-safe for concurrent operations
            # Singleton prevents multiple models, lock prevents concurrent access
            with MLX_GLOBAL_LOCK:
                # Unpack shared model and pre-created pipeline
                model, pipeline = MLXKokoroTTSService._shared_pipeline

                # Collect ALL audio segments (not just first one!)
                audio_segments = []

                # Call pipeline directly (not model.generate which creates pipeline lazily)
                # pipeline() returns (graphemes, phonemes, audio) tuples
                for graphemes, phonemes, audio in pipeline(
                    text,
                    voice=self._voice,
                    speed=self._speed,
                    split_pattern=r'\n+'
                ):
                    # Check cancellation flag during generation
                    if self._cancelled:
                        logger.debug("MLX TTS generation cancelled during pipeline iteration")
                        # Ensure Metal operations complete before exiting lock
                        import mlx.core as mx
                        if audio_segments:
                            mx.eval(audio_segments)  # Force evaluation of any pending MLX ops
                        return None

                    # pipeline() returns audio directly in the tuple, not as an attribute
                    if audio is not None:
                        audio_segments.append(audio)

                # Concatenate all audio segments (INSIDE LOCK - these are MLX operations!)
                if len(audio_segments) == 0:
                    logger.warning("No audio generated")
                    return None
                elif len(audio_segments) == 1:
                    audio_array = audio_segments[0]
                else:
                    # Concatenate multiple segments using MLX
                    import mlx.core as mx
                    audio_array = mx.concatenate(audio_segments, axis=0)

                # Convert MLX array to NumPy array (INSIDE LOCK - accesses Metal!)
                audio_np = np.array(audio_array, copy=False)

            generation_time = time.time() - start_time

            # Log performance
            chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
            logger.debug(f"MLX generated {len(text)} chars in {generation_time:.3f}s ({chars_per_sec:.1f} chars/s)")

            # Force garbage collection after generation (macOS Sequoia workaround)
            import gc
            gc.collect()

            return audio_np, self._sample_rate

        except Exception as e:
            logger.error(f"MLX Kokoro TTS generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using MLX Kokoro with optimal chunking"""

        # Reset cancellation flag at start of new TTS request
        self._cancelled = False

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # Sentence-first chunking for natural prosody (disables 25-char micro-chunks)
        # Env: TTS_CHUNK_SIZE_CHARS optionally caps sentence grouping (default to CHUNK_MAX_LENGTH)
        try:
            max_chars = int(os.getenv("TTS_CHUNK_SIZE_CHARS", str(CHUNK_MAX_LENGTH)))
        except Exception:
            max_chars = CHUNK_MAX_LENGTH

        # Apply Kokoro-specific sanitization before chunking
        from tools.text_formatter import sanitize_for_kokoro
        clean_text = sanitize_for_kokoro(text)

        sentences = split_text_for_kokoro_streaming(
            clean_text,
            min_length=max(30, min(CHUNK_MIN_LENGTH, max_chars // 2)),
            max_length=max_chars,
        )

        if not sentences:
            logger.debug(f"🔇 Skipping TTS for empty text: '{text}'")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        logger.debug(f"🎤 MLX TTS for {len(sentences)} chunks: {sentences[0][:40]}{'...' if len(sentences[0]) > 40 else ''}")

        overall_start_time = time.time()
        first_audio_sent = False

        yield TTSStartedFrame()

        try:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                chunk_start_time = time.time()

                # Generate audio using MLX (should be much faster than ONNX)
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
                            logger.debug(f"🚀 MLX KOKORO TTFB: {ttfb:.1f}ms")
                            first_audio_sent = True

                        chunk_latency = (time.time() - chunk_start_time) * 1000
                        logger.debug(f"✅ MLX Chunk {i+1}/{len(sentences)}: {len(sentence)} chars → {chunk_latency:.1f}ms")

                        # Convert to int16 for Pipecat (MLX audio is already normalized, no clipping needed)
                        audio_int16 = convert_to_pcm16(audio_data, clip=False)

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
            # Set cancellation flag to stop any in-progress generation
            self._cancelled = True
            logger.debug("MLX TTS generation cancelled - flag set to abort ongoing Metal operations")
            raise
        except Exception as e:
            logger.error(f"MLX Kokoro TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup executor on deletion (fallback for non-context-manager usage)"""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore errors during cleanup
