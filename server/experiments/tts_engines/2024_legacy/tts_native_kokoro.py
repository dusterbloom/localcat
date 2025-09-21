"""
Native Kokoro ONNX TTS Service for ultra-low latency streaming.
Based on the optimized approach from maestrocat using real Kokoro ONNX.
"""

import asyncio
import concurrent.futures
import threading
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

from tools.text_formatter import chunk_for_kokoro_ultra_low_latency

# Force single-threaded execution to prevent Metal framework conflicts and heating
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


class NativeKokoroTTSService(TTSService):
    """
    Native Kokoro ONNX TTS service for ultra-low latency streaming.
    Uses the real Kokoro ONNX implementation, not MLX.
    """

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
            aggregate_sentences=True,  # Accumulate tokens into proper sentences
            **kwargs
        )

        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate

        # Single-threaded executor to prevent Metal framework conflicts and heating
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kokoro-onnx"
        )

        # Lock to ensure only one TTS generation at a time
        self._tts_lock = threading.Lock()

        # Initialize Kokoro ONNX pipeline
        self._pipeline = None
        self._initialize_pipeline()

        logger.debug(f"✅ Native Kokoro ONNX TTS initialized with voice: {self._voice}")

    def _ensure_models_downloaded(self):
        """Ensure the correct model and voices files are available"""
        cache_dir = Path.home() / ".cache" / "kokoro"
        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"

        # Check if files exist (follow symlinks)
        try:
            model_exists = model_path.exists() and model_path.resolve().stat().st_size > 300_000_000  # >300MB
        except:
            model_exists = False

        try:
            voices_exists = voices_path.exists() and voices_path.resolve().stat().st_size > 25_000_000  # >25MB
        except:
            voices_exists = False

        if model_exists and voices_exists:
            return str(model_path.resolve()), str(voices_path.resolve())

        # Files are missing - download them
        logger.debug("📥 Downloading Kokoro ONNX model files...")

        import urllib.request

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model file
        if not model_exists:
            logger.debug("Downloading kokoro-v1.0.onnx...")
            # Remove any existing symlink first
            if model_path.is_symlink():
                model_path.unlink()
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                model_path
            )

        # Download voices file
        if not voices_exists:
            logger.debug("Downloading voices-v1.0.bin...")
            # Remove any existing symlink first
            if voices_path.is_symlink():
                voices_path.unlink()
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                voices_path
            )

        logger.debug("✅ Kokoro ONNX model files downloaded")
        return str(model_path.resolve()), str(voices_path.resolve())

    def _initialize_pipeline(self):
        """Initialize the Kokoro ONNX pipeline"""
        try:
            from kokoro_onnx import Kokoro

            logger.debug(f"🚀 Initializing native Kokoro ONNX TTS with voice: {self._voice}")

            # Ensure models are available
            model_path, voices_path = self._ensure_models_downloaded()

            logger.debug(f"Using model: {model_path}")
            logger.debug(f"Using voices: {voices_path}")

            # Initialize Kokoro with the correct files
            self._pipeline = Kokoro(
                model_path=model_path,
                voices_path=voices_path,
                espeak_config=None  # Use default
            )

            logger.debug(f"✅ Kokoro ONNX pipeline loaded successfully")

            # Test the voice
            try:
                test_audio, test_sr = self._pipeline.create("Hello", voice=self._voice, speed=self._speed)
                logger.debug(f"✅ Voice '{self._voice}' verified - generated {len(test_audio)} samples at {test_sr}Hz")
            except Exception as voice_error:
                logger.error(f"❌ Voice '{self._voice}' failed: {voice_error}")
                # Try safe fallback voice
                try:
                    test_audio, test_sr = self._pipeline.create("Hello", voice="af_bella", speed=self._speed)
                    logger.debug(f"✅ Fallback to af_bella - generated {len(test_audio)} samples")
                    self._voice = "af_bella"
                except:
                    # Last resort: try voice 0 as string
                    test_audio, test_sr = self._pipeline.create("Hello", voice="0", speed=self._speed)
                    logger.debug(f"✅ Ultimate fallback to voice 0 - generated {len(test_audio)} samples")
                    self._voice = "0"

            logger.info("Native Kokoro ONNX TTS ready")

        except ImportError:
            logger.error("kokoro-onnx not installed. Install with: pip install kokoro-onnx")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro ONNX: {e}")
            self._pipeline = None
            raise

    def _generate_audio_sync(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synchronous audio generation - runs in thread pool with single-threaded lock"""
        if not self._pipeline:
            logger.error("Kokoro pipeline not initialized")
            return None

        # Ensure single-threaded execution to prevent Metal conflicts
        with self._tts_lock:
            try:
                start_time = time.time()

                # Generate audio using Kokoro ONNX
                audio_data, sample_rate = self._pipeline.create(
                    text,
                    voice=self._voice,
                    speed=self._speed
                )

                generation_time = time.time() - start_time

                # Log performance
                chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
                logger.debug(f"Generated {len(text)} chars in {generation_time:.3f}s ({chars_per_sec:.1f} chars/s)")

                return audio_data, sample_rate

            except Exception as e:
                logger.error(f"Kokoro ONNX TTS generation failed: {e}")
                return None

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from complete sentences using optimal Kokoro chunking"""

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # Use ultra-low latency chunking for optimal TTS performance
        # Based on benchmarking: 25 chars = 487ms TTFT, 150 chars = 3,556ms TTFT
        sentences = chunk_for_kokoro_ultra_low_latency(text, max_chars=25)

        if not sentences:
            logger.debug(f"🔇 Skipping TTS for empty text: '{text}'")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        logger.debug(f"🎤 Streaming {len(sentences)} optimized chunks: {sentences[0][:40]}{'...' if len(sentences[0]) > 40 else ''}")

        overall_start_time = time.time()
        first_audio_sent = False

        yield TTSStartedFrame()

        try:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                chunk_start_time = time.time()

                # Generate audio for this optimally-sized chunk
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
                            logger.debug(f"🚀 OPTIMIZED KOKORO TTFB: {ttfb:.1f}ms")
                            first_audio_sent = True

                        chunk_latency = (time.time() - chunk_start_time) * 1000
                        logger.debug(f"✅ Chunk {i+1}/{len(sentences)}: {len(sentence)} chars → {chunk_latency:.1f}ms")

                        # Convert to int16 for Pipecat
                        if audio_data.dtype != np.int16:
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                        else:
                            audio_int16 = audio_data

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
            logger.debug("TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Optimized Kokoro TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)
