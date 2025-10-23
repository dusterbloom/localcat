"""
Parakeet Batch STT Service for Pipecat
Provides batch-mode transcription for fallback scenarios and long-form audio processing.
"""

import asyncio
import numpy as np
import os
import tempfile
import time
import wave
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    ErrorFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame
)
from pipecat.services.stt_service import STTService

from core.utils.mlx_lock import MLX_GLOBAL_LOCK

try:
    from parakeet_mlx import from_pretrained
    PARAKEET_AVAILABLE = True
except ImportError:
    logger.warning("parakeet_mlx not available. Install with: pip install parakeet-mlx")
    try:
        # Fallback to old mlx_audio
        from mlx_audio.stt.utils import load_model
        PARAKEET_AVAILABLE = True
        PARAKEET_OLD_FORMAT = True
        logger.warning("Using legacy mlx_audio format - batch mode only")
    except ImportError:
        logger.warning("Neither parakeet_mlx nor mlx_audio available")
        PARAKEET_AVAILABLE = False
        PARAKEET_OLD_FORMAT = False
else:
    PARAKEET_OLD_FORMAT = False


class ParakeetBatchSTT(STTService):
    """
    Batch-mode Parakeet STT for fallback scenarios.
    Processes complete audio utterances for higher accuracy than streaming mode.
    """

    # Known Parakeet hallucination patterns (common false positives on silence/noise)
    # NOTE: These should be normalized (no punctuation) since we strip punctuation before matching
    HALLUCINATION_PATTERNS = {
        "yeah", "yep", "mmhmm", "mmhmmm", "mhm", "uhhuh",
        "im just", "thank you", "thanks",
        "uh", "um", "hmm", "ah", "oh", "Почему?","Scary."
    }

    # CRITICAL: Words that should NEVER be filtered, even if short
    # These are essential for user interactions (confirmations, names, commands)
    CRITICAL_WORDS = {
        "yes", "no", "ok", "okay", "go", "hi", "hey", "bye", "one", "two"
    }

    def __init__(
        self,
        *,
        model_path: str = "mlx-community/parakeet-tdt-0.6b-v3",
        language: str = "en",
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not PARAKEET_AVAILABLE:
            raise ImportError("parakeet_mlx is required but not installed")

        self._model_path = model_path
        self._language = language
        self._sample_rate = 16000  # Parakeet expects 16kHz
        self._temperature = temperature

        # Model initialization
        self._model = None
        self._init_parakeet_model()

        logger.info(f"✅ Parakeet Batch STT initialized: {model_path}")

        # Streaming/VAD integration state for batch mode
        self._vad_active: bool = False
        self._buffered_audio: list[np.ndarray] = []  # float32 [-1,1] segments @ 16kHz
        self._buffer_duration: float = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    def _init_parakeet_model(self):
        """Initialize the Parakeet model for batch processing"""
        try:
            logger.info(f"🔒 Acquiring MLX lock for Parakeet batch initialization...")
            with MLX_GLOBAL_LOCK:
                logger.info(f"Loading Parakeet model: {self._model_path}")

                if PARAKEET_OLD_FORMAT:
                    # Legacy mlx_audio format
                    self._model = load_model(self._model_path)
                else:
                    # New parakeet_mlx format
                    self._model = from_pretrained(self._model_path)

                logger.info("✅ Parakeet batch model loaded successfully")
                logger.debug("Keeping MLX model in memory (shares Metal heap with Kokoro MLX TTS)")

        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            raise

    def _normalize_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """Normalize audio volume to optimal levels"""
        # Calculate RMS and peak
        rms = np.sqrt(np.mean(audio_np**2))
        peak = np.max(np.abs(audio_np))

        if rms < 1e-8 or peak < 1e-8:  # Essentially silent
            return audio_np

        # Target RMS level for optimal transcription (avoid clipping)
        target_rms = 0.1

        # Calculate gain needed
        if rms > 0:
            gain = target_rms / rms
            # Limit gain to prevent over-amplification
            gain = min(gain, 0.8 / peak)  # Keep peak below 0.8 to avoid clipping
            gain = max(gain, 0.1)  # Don't reduce too much

            # Apply gain
            normalized = audio_np * gain

            # Soft clipping if needed
            normalized = np.tanh(normalized * 1.2) * 0.8

            # logger.debug(f"Audio normalized: RMS {rms:.4f} -> {np.sqrt(np.mean(normalized**2)):.4f}, gain: {gain:.2f}")
            return normalized

        return audio_np

    def _append_audio(self, audio_bytes: bytes) -> None:
        """Append incoming PCM16 bytes to the utterance buffer (assumed 16kHz mono)."""
        if not audio_bytes:
            return
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        arr = self._normalize_audio(arr)
        self._buffered_audio.append(arr)
        self._buffer_duration += len(arr) / float(self._sample_rate)

    def _get_buffer_bytes(self) -> bytes:
        if not self._buffered_audio:
            return b""
        full = (
            np.concatenate(self._buffered_audio)
            if len(self._buffered_audio) > 1
            else self._buffered_audio[0]
        )
        audio_int16 = np.clip(full * 32767.0, -32768.0, 32767.0).astype(np.int16)
        return audio_int16.tobytes()

    def _reset_buffer(self) -> None:
        self._buffered_audio = []
        self._buffer_duration = 0.0

    def _audio_bytes_to_wav(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to temporary WAV file for Parakeet with normalization"""
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Apply audio normalization
        audio_np = self._normalize_audio(audio_np)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write WAV file
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self._sample_rate)
                # Convert back to int16
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            return temp_file.name

    def _is_hallucination(self, text: str) -> bool:
        """Check if text matches known hallucination patterns"""
        if not text or not text.strip():
            return True

        # Normalize text for matching
        normalized = text.strip().lower()

        # Remove punctuation for matching
        import re
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # CRITICAL: Check whitelist first - never filter critical words
        if normalized in self.CRITICAL_WORDS:
            return False

        # Check if entire text matches a hallucination pattern
        if normalized in self.HALLUCINATION_PATTERNS:
            logger.debug(f"Blocked hallucination: '{text}'")
            return True

        # Check for very short single-word outputs (likely noise)
        # Skip this check if the word is in the critical whitelist
        words = normalized.split()
        if len(words) == 1 and len(words[0]) <= 3:
            # Double-check it's not a critical word (belt and suspenders)
            if words[0] not in self.CRITICAL_WORDS:
                logger.debug(f"Blocked short noise: '{text}'")
                return True

        return False

    def _process_audio_batch_sync(self, audio_bytes: bytes) -> tuple[str, str]:
        """Synchronous batch processing - returns (text, audio_path)

        CRITICAL: Uses MLX_GLOBAL_LOCK to prevent concurrent Metal access during transcription.
        """
        # Convert to temporary WAV file
        audio_path = self._audio_bytes_to_wav(audio_bytes)

        # CRITICAL: Acquire lock during transcription to prevent concurrent Metal access
        # This prevents process killing when Parakeet STT and Kokoro TTS run simultaneously
        with MLX_GLOBAL_LOCK:
            # Generate transcription (use model's transcribe API to avoid shape issues)
            if PARAKEET_OLD_FORMAT:
                # Legacy mlx_audio API may expect raw path for generate
                result = self._model.generate(audio_path)
            else:
                # New parakeet_mlx API exposes transcribe(path) → AlignedResult
                result = self._model.transcribe(audio_path)

        # Extract text from result
        text = ""

        if hasattr(result, 'text'):
            text = result.text.strip()
        elif hasattr(result, 'transcription'):
            text = result.transcription.strip()
        else:
            logger.warning(f"Unexpected result format from Parakeet: {type(result)}")
            return "", audio_path

        logger.debug(f"Parakeet batch transcription: '{text}'")

        # Filter known hallucinations instead of using confidence heuristics
        if self._is_hallucination(text):
            return "", audio_path

        return text, audio_path

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Buffer audio during speech; produce final on end-of-turn (see process_frame)."""
        try:
            if self._vad_active:
                self._append_audio(audio)
            # Batch mode yields no interims
        except Exception as e:
            logger.error(f"Error buffering audio for Parakeet batch STT: {e}")
            yield ErrorFrame(f"Parakeet batch STT buffering error: {e}")

    async def process_frame(self, frame: Frame, direction=None):
        """Handle VAD events to orchestrate batch transcription lifecycle."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._vad_active = True
            self._reset_buffer()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._vad_active = False
            try:
                audio_bytes = self._get_buffer_bytes()
                self._reset_buffer()
                if not audio_bytes:
                    return

                # Start metrics
                await self.start_processing_metrics()
                await self.start_ttfb_metrics()

                # Run transcription in executor
                text, audio_path = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_audio_batch_sync, audio_bytes
                )

                # Stop metrics
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()

                # Clean up temporary file
                try:
                    os.unlink(audio_path)
                except:
                    pass

                if text:
                    await self.push_frame(
                        TranscriptionFrame(
                            text=text,
                            user_id=self._user_id or "user",
                            timestamp=str(time.time())
                        )
                    )
            except Exception as e:
                logger.error(f"Parakeet batch finalize error: {e}")
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
