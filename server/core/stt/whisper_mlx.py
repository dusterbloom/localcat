"""
Whisper-MLX STT Service - Direct MLX Whisper implementation for ultra-low latency.
Uses MLX-optimized Whisper with direct transcribe() calls, bypassing Pipecat's batching.

Performance:
- DirectMLXWhisperSTT: ~150ms latency (36x faster than Pipecat's batch wrapper)
- Uses quantized models for Apple Silicon optimization
- Global MLX lock prevents Metal concurrency issues
"""

import asyncio
import time
import numpy as np
from typing import AsyncGenerator, Union

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    ErrorFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.services.ai_services import SegmentedSTTService
from core.utils.mlx_lock import MLX_GLOBAL_LOCK
from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame


class DirectMLXWhisperSTT(SegmentedSTTService):
    """
    Ultra-low latency Whisper STT using direct mlx_whisper.transcribe() calls.
    Bypasses Pipecat's batching wrapper for 36x speed improvement.

    Key optimizations:
    - Direct transcribe() calls (no batching overhead)
    - Quantized model support
    - MLX global lock (prevents Metal conflicts)
    - Warmup during init (eliminates first-call penalty)
    - Optimized transcribe parameters (temperature=0.0, fp16=False)
    """

    def __init__(
        self,
        *,
        model: str = "mlx-community/whisper-small.en-mlx-q4",
        language: str = "en",
        temperature: float = 0.0,
        no_speech_threshold: float = 0.6,
        hallucination_silence_threshold: float = 0.3,
        _preloaded_whisper_module=None,  # NEW: Accept preloaded module
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._model = model
        self._language = language
        self._temperature = temperature
        self._preloaded_whisper = _preloaded_whisper_module  # Store for later use
        # Allow env overrides for anti-hallucination tuning
        import os as _os
        try:
            self._no_speech_threshold = float(_os.getenv("WHISPER_NO_SPEECH_THRESHOLD", str(no_speech_threshold)))
        except Exception:
            self._no_speech_threshold = no_speech_threshold
        try:
            self._hallucination_silence_threshold = float(
                _os.getenv("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", str(hallucination_silence_threshold))
            )
        except Exception:
            self._hallucination_silence_threshold = hallucination_silence_threshold
        try:
            self._logprob_threshold = float(_os.getenv("WHISPER_LOGPROB_THRESHOLD", "-0.25"))
        except Exception:
            self._logprob_threshold = -0.25
        try:
            self._volume_threshold = float(_os.getenv("WHISPER_VOLUME_THRESHOLD", "0.01"))
        except Exception:
            self._volume_threshold = 0.01

        # Optional phrase suppression for common UI-action hallucinations
        suppress_from_env = _os.getenv("WHISPER_SUPPRESS_PHRASES", "")
        base_suppress = {
            "im going to start the video",
            "i'm going to start the video",
            "im going to start video",
            "i'm going to start video",
            "start the video",
            "start video",
            "start recording",
            "i'm going to start recording",
            "im going to start recording",
            # Broader activity phrases frequently mis-fired on silence/background
            "i'm going to be doing a video",
            "im going to be doing a video",
            "going to be doing a video",
            "i am going to be doing a video",
            "i'm going to be doing",
            "im going to be doing",
            "going to start",
        }
        if suppress_from_env.strip():
            extra = {s.strip().lower() for s in suppress_from_env.split(",") if s.strip()}
            base_suppress |= extra
        self._suppress_phrases = base_suppress

        # Track VAD state; only transcribe when actively speaking
        self._vad_active = False
        # Accept both WHISPER_MIN_SEGMENT_SEC and legacy WHISPER_MIN_SEGMENT_DURATION
        try:
            min_seg = _os.getenv("WHISPER_MIN_SEGMENT_SEC") or _os.getenv("WHISPER_MIN_SEGMENT_DURATION")
            self._min_segment_sec = float(min_seg) if min_seg is not None else 0.3
        except Exception:
            self._min_segment_sec = 0.3
        # Max audio duration override
        try:
            self._max_chunk_sec = float(_os.getenv("WHISPER_MAX_AUDIO_DURATION", "10.0"))
        except Exception:
            self._max_chunk_sec = 10.0

        # Use preloaded module if available, otherwise import fresh
        if self._preloaded_whisper:
            self._mlx_whisper = self._preloaded_whisper
            logger.info(f"✅ DirectMLXWhisperSTT using PRELOADED module: {model} (instant!)")
            # Skip warmup - already done during preload
        else:
            # Fallback: Load from scratch (slower)
            try:
                import mlx_whisper
                self._mlx_whisper = mlx_whisper
                logger.info(f"🚀 DirectMLXWhisperSTT: {model}")
                # Warmup: Load model and eliminate cold start penalty
                self._warmup()
                logger.warning("⚠️  Loaded Whisper from scratch (slow) - preloading failed?")
            except ImportError:
                logger.error("mlx-whisper not available. Install with: pip install mlx-whisper")
                raise

    def _warmup(self):
        """Warmup the model to eliminate first-call latency penalty."""
        try:
            logger.info("🔥 Warming up Whisper model...")
            with MLX_GLOBAL_LOCK:
                # Create 1 second of silence for warmup
                dummy_audio = np.zeros(16000, dtype=np.float32)

                # Trigger model loading
                result = self._mlx_whisper.transcribe(
                    dummy_audio,
                    path_or_hf_repo=self._model,
                    verbose=False,
                    temperature=self._temperature,
                    fp16=False,
                    no_speech_threshold=self._no_speech_threshold
                )

                logger.info(f"✅ Whisper warmup complete - model loaded and ready")
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed (non-critical): {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Process audio and generate transcription using direct mlx_whisper calls.

        Args:
            audio: Raw audio bytes (float32 PCM, 16kHz)

        Yields:
            TranscriptionFrame with transcribed text
        """
        try:
            start_time = time.time()
            logger.debug(f"🎤 DirectMLXWhisperSTT.run_stt called with {len(audio)} bytes")

            # Min segment duration check (tunable): reject very short chunks
            # Prevents hallucinations on noise/background audio
            duration_seconds = len(audio) / (16000 * 2)  # 16kHz, 2 bytes per sample
            if duration_seconds < self._min_segment_sec:
                logger.debug(f"⏭️  Skipping chunk: {duration_seconds:.2f}s < 0.3s minimum")
                return

            # Max audio chunk size limiting: cap at 10 seconds
            # Prevents extreme latency from processing 30-60 second utterances
            MAX_CHUNK_SECONDS = max(1.0, self._max_chunk_sec)
            max_bytes = int(MAX_CHUNK_SECONDS * 16000 * 2)

            if len(audio) > max_bytes:
                duration = len(audio) / (16000 * 2)
                logger.warning(
                    f"⚠️  Audio chunk too large: {len(audio)/1024:.1f}KB ({duration:.1f}s), "
                    f"truncating to {MAX_CHUNK_SECONDS}s"
                )
                # Keep most recent audio (end of utterance is most important)
                audio = audio[-max_bytes:]

            # Optimize NumPy conversion: reduce allocations and copies
            audio_int16 = np.frombuffer(audio, dtype=np.int16)
            audio_np = audio_int16.astype(np.float32)
            audio_np /= 32768.0  # In-place division

            # Lightweight logging (removed expensive array scan)
            logger.debug(f"📊 Audio: {len(audio)} bytes ({len(audio)/(16000*2):.1f}s)")

            # Skip RMS check when VAD active (trust VAD's decision)
            # Only compute RMS when VAD is inactive to save processing time
            if not self._vad_active:
                rms = float(np.sqrt(np.mean(audio_np ** 2))) if audio_np.size > 0 else 0.0
                if rms < self._volume_threshold:
                    logger.debug(f"⏭️  Skipping low-volume audio (rms={rms:.4f})")
                    return
            else:
                # VAD confirmed speech, skip expensive RMS calculation
                rms = 0.0  # Placeholder for hallucination suppression check

            # Direct transcription with MLX lock
            with MLX_GLOBAL_LOCK:
                result = self._mlx_whisper.transcribe(
                    audio_np,
                    path_or_hf_repo=self._model,
                    verbose=False,
                    language=self._language,
                    temperature=self._temperature,
                    fp16=False,
                    no_speech_threshold=self._no_speech_threshold,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=self._logprob_threshold,
                    hallucination_silence_threshold=self._hallucination_silence_threshold
                )

            elapsed_ms = (time.time() - start_time) * 1000

            if result and "text" in result:
                text = result["text"].strip()

                if text:
                    # Normalize and suppress known UI-action hallucinations
                    norm = text.strip().lower()
                    # Broad keyword suppression for common misfires on silence/noise
                    if norm in self._suppress_phrases and (not self._vad_active or rms < self._volume_threshold * 1.5):
                        logger.info(f"🧯 Whisper hallucination suppressed: '{text}' (rms={rms:.4f})")
                        return
                    # Heuristic suppression: short utterances about starting/recording videos
                    if ("video" in norm or "record" in norm) and (
                        norm.startswith("i'm going to") or norm.startswith("im going to") or "going to" in norm
                    ):
                        if (not self._vad_active and duration_seconds < max(0.8, self._min_segment_sec * 2)) or rms < self._volume_threshold * 1.2:
                            logger.info(f"🧯 Whisper heuristic-suppressed: '{text}' (dur={duration_seconds:.2f}s, rms={rms:.4f}, vad={self._vad_active})")
                            return
                    logger.info(f"🎯 Direct Whisper: '{text}' ({elapsed_ms:.1f}ms)")
                    yield TranscriptionFrame(text=text, user_id="", timestamp="")
                else:
                    logger.info(f"Empty transcription ({elapsed_ms:.1f}ms)")
            else:
                logger.warning("No transcription result from Whisper")

        except Exception as e:
            logger.error(f"DirectMLXWhisper error: {e}")
            yield ErrorFrame(error=f"Whisper transcription error: {e}")

    async def process_frame(self, frame: Frame, direction=None):
        """Track VAD frames to avoid transcribing silence/background noise."""
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, UserStartedSpeakingFrame):
                self._vad_active = True
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._vad_active = False
        except Exception:
            pass


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
