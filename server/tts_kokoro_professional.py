"""
Professional Kokoro TTS Service with artifact-free audio processing.
Drop-in replacement for tts_native_kokoro.py with superior audio quality.
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
from audio_processor import AudioProcessor, create_clean_audio_frame

# Force single-threaded execution to prevent Metal framework conflicts
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


class ProfessionalKokoroTTSService(TTSService):
    """
    Professional-grade Kokoro TTS service with artifact-free audio processing.

    Key improvements over NativeKokoroTTSService:
    - Eliminates sentence-ending artifacts through proper audio processing
    - Smart limiting prevents clipping while preserving dynamics
    - Automatic fade-out for natural sentence endings
    - DC offset removal for cleaner audio
    - Comprehensive quality validation and logging
    - Maintains same API compatibility for drop-in replacement
    """

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000,
        fade_duration_ms: float = 50.0,
        target_peak_db: float = -3.0,
        enable_quality_logging: bool = True,
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
        self._enable_quality_logging = enable_quality_logging

        # Initialize professional audio processor
        self._audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            fade_duration_ms=fade_duration_ms,
            target_peak_db=target_peak_db
        )

        # Single-threaded executor to prevent Metal framework conflicts
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kokoro-pro"
        )

        # Lock to ensure only one TTS generation at a time
        self._tts_lock = threading.Lock()

        # Initialize Kokoro ONNX pipeline
        self._pipeline = None
        self._initialize_pipeline()

        logger.info(f"✨ Professional Kokoro TTS initialized: voice={voice}, fade={fade_duration_ms}ms")

    def _ensure_models_downloaded(self):
        """Ensure the correct model and voices files are available"""
        cache_dir = Path.home() / ".cache" / "kokoro"
        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"

        # Check if files exist (follow symlinks)
        try:
            model_exists = model_path.exists() and model_path.resolve().stat().st_size > 300_000_000
        except:
            model_exists = False

        try:
            voices_exists = voices_path.exists() and voices_path.resolve().stat().st_size > 25_000_000
        except:
            voices_exists = False

        if model_exists and voices_exists:
            return str(model_path.resolve()), str(voices_path.resolve())

        # Files are missing - download them
        logger.info("📥 Downloading Kokoro ONNX model files...")

        import urllib.request

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model file
        if not model_exists:
            logger.info("Downloading kokoro-v1.0.onnx...")
            if model_path.is_symlink():
                model_path.unlink()
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                model_path
            )

        # Download voices file
        if not voices_exists:
            logger.info("Downloading voices-v1.0.bin...")
            if voices_path.is_symlink():
                voices_path.unlink()
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                voices_path
            )

        logger.info("✅ Kokoro ONNX model files ready")
        return str(model_path.resolve()), str(voices_path.resolve())

    def _initialize_pipeline(self):
        """Initialize the Kokoro ONNX pipeline"""
        try:
            from kokoro_onnx import Kokoro

            logger.debug(f"🚀 Initializing Professional Kokoro ONNX with voice: {self._voice}")

            # Ensure models are available
            model_path, voices_path = self._ensure_models_downloaded()

            # Initialize Kokoro with the correct files
            self._pipeline = Kokoro(
                model_path=model_path,
                voices_path=voices_path,
                espeak_config=None
            )

            logger.debug(f"✅ Professional Kokoro pipeline loaded")

            # Test the voice with quality validation
            try:
                test_audio, test_sr = self._pipeline.create("Hello", voice=self._voice, speed=self._speed)

                # Process through our audio pipeline to validate
                clean_frame = create_clean_audio_frame(
                    test_audio,
                    sample_rate=test_sr,
                    log_quality=True
                )

                logger.info(f"✅ Voice '{self._voice}' validated with professional processing")

            except Exception as voice_error:
                logger.error(f"❌ Voice '{self._voice}' failed: {voice_error}")
                # Try safe fallback voice
                try:
                    test_audio, test_sr = self._pipeline.create("Hello", voice="af_bella", speed=self._speed)
                    logger.info(f"✅ Fallback to af_bella successful")
                    self._voice = "af_bella"
                except:
                    # Last resort: try voice 0
                    test_audio, test_sr = self._pipeline.create("Hello", voice="0", speed=self._speed)
                    logger.info(f"✅ Ultimate fallback to voice 0")
                    self._voice = "0"

            logger.info("🎭 Professional Kokoro TTS ready")

        except ImportError:
            logger.error("kokoro-onnx not installed. Install with: pip install kokoro-onnx")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Professional Kokoro: {e}")
            self._pipeline = None
            raise

    def _generate_audio_sync(self, text: str) -> Optional[tuple[np.ndarray, int, dict, dict]]:
        """
        Synchronous audio generation with professional processing.

        Returns:
            Tuple of (clean_audio_int16, sample_rate, processing_stats, quality_metrics)
        """
        if not self._pipeline:
            logger.error("Kokoro pipeline not initialized")
            return None

        # Ensure single-threaded execution
        with self._tts_lock:
            try:
                start_time = time.time()

                # Generate raw audio using Kokoro ONNX
                raw_audio, sample_rate = self._pipeline.create(
                    text,
                    voice=self._voice,
                    speed=self._speed
                )

                generation_time = time.time() - start_time

                # Process through professional audio pipeline
                processing_start = time.time()

                clean_audio, processing_stats = self._audio_processor.process_tts_audio(
                    raw_audio,
                    apply_fade=True,
                    remove_dc=True
                )

                quality_metrics = self._audio_processor.validate_audio_quality(clean_audio)

                processing_time = time.time() - processing_start

                # Enhanced performance logging
                chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
                logger.debug(
                    f"🎵 Professional TTS: {len(text)} chars in {generation_time:.3f}s "
                    f"({chars_per_sec:.1f} chars/s) + {processing_time*1000:.1f}ms processing"
                )

                # Log quality if enabled
                if self._enable_quality_logging and quality_metrics["status"] != "clean":
                    warnings = quality_metrics.get("warnings", [])
                    logger.warning(f"Audio quality warnings: {warnings}")

                return clean_audio, sample_rate, processing_stats, quality_metrics

            except Exception as e:
                logger.error(f"Professional Kokoro TTS generation failed: {e}")
                return None

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate high-quality speech with professional audio processing"""

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # Use ultra-low latency chunking for optimal performance
        sentences = chunk_for_kokoro_ultra_low_latency(text, max_chars=25)

        if not sentences:
            logger.debug(f"🔇 Skipping TTS for empty text: '{text}'")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        logger.debug(f"🎤 Professional streaming {len(sentences)} chunks: {sentences[0][:40]}{'...' if len(sentences[0]) > 40 else ''}")

        overall_start_time = time.time()
        first_audio_sent = False
        total_quality_warnings = 0

        yield TTSStartedFrame()

        try:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                chunk_start_time = time.time()

                # Generate audio with professional processing
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._generate_audio_sync,
                    sentence
                )

                if result is not None:
                    clean_audio, actual_sample_rate, processing_stats, quality_metrics = result

                    if clean_audio is not None and len(clean_audio) > 0:
                        # Calculate TTFB for first chunk only
                        if not first_audio_sent:
                            ttfb = (time.time() - overall_start_time) * 1000
                            logger.debug(f"🚀 PROFESSIONAL KOKORO TTFB: {ttfb:.1f}ms")
                            first_audio_sent = True

                        chunk_latency = (time.time() - chunk_start_time) * 1000

                        # Enhanced chunk logging with quality info
                        quality_status = quality_metrics.get("status", "unknown")
                        warnings_count = len(quality_metrics.get("warnings", []))
                        total_quality_warnings += warnings_count

                        logger.debug(
                            f"✨ Professional Chunk {i+1}/{len(sentences)}: {len(sentence)} chars → "
                            f"{chunk_latency:.1f}ms | {quality_status} | warnings: {warnings_count}"
                        )

                        # Create standard Pipecat frame
                        frame = TTSAudioRawFrame(
                            audio=clean_audio.tobytes(),
                            sample_rate=actual_sample_rate,
                            num_channels=1
                        )

                        yield frame
                    else:
                        logger.warning(f"Empty audio data for chunk: '{sentence}'")
                else:
                    logger.warning(f"No audio generated for chunk: '{sentence}'")

        except asyncio.CancelledError:
            logger.debug("Professional TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Professional Kokoro TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            # Final quality summary
            total_time = time.time() - overall_start_time
            logger.info(
                f"🎭 Professional TTS complete: {len(sentences)} chunks, {total_time:.2f}s, "
                f"{total_quality_warnings} quality warnings"
            )
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)


# Backward compatibility alias
class EnhancedKokoroTTSService(ProfessionalKokoroTTSService):
    """Alias for backward compatibility."""
    pass