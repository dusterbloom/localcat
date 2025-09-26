"""
Parakeet Batch STT Service for Pipecat
Provides batch-mode transcription for fallback scenarios and long-form audio processing.
"""

import asyncio
import json
import numpy as np
import os
import tempfile
import time
import wave
from typing import AsyncGenerator, Optional

import mlx.core as mx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    ErrorFrame,
    AudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame
)
from pipecat.services.stt_service import STTService
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import load_audio
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

    def __init__(
        self,
        *,
        model_path: str = "mlx-community/parakeet-tdt-0.6b-v3",
        language: str = "en",
        confidence_threshold: float = 0.3,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not PARAKEET_AVAILABLE:
            raise ImportError("parakeet_mlx is required but not installed")

        self._model_path = model_path
        self._language = language
        self._sample_rate = 16000  # Parakeet expects 16kHz
        self._confidence_threshold = confidence_threshold
        self._temperature = temperature

        # Model initialization
        self._model = None
        self._init_parakeet_model()

        logger.info(f"✅ Parakeet Batch STT initialized: {model_path} (confidence_threshold: {confidence_threshold})")

    def _init_parakeet_model(self):
        """Initialize the Parakeet model for batch processing"""
        try:
            logger.info(f"Loading Parakeet model: {self._model_path}")

            if PARAKEET_OLD_FORMAT:
                # Legacy mlx_audio format
                self._model = load_model(self._model_path)
            else:
                # New parakeet_mlx format
                self._model = from_pretrained(self._model_path)

            logger.info("✅ Parakeet batch model loaded successfully")
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

    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence score based on text characteristics"""
        if not text or not text.strip():
            return 0.0

        text = text.strip()
        words = text.split()

        # Very short texts are likely hallucinations
        if len(words) < 2:
            return 0.2

        # Penalize very long texts (might be repetition)
        if len(words) > 50:
            return 0.1

        # Base confidence on text length and word diversity
        word_count = len(words)
        unique_words = len(set(words))
        diversity_ratio = unique_words / word_count if word_count > 0 else 0

        # Combine factors for confidence estimate
        length_score = min(word_count / 10, 1.0)  # Favor reasonable length
        diversity_score = diversity_ratio  # Favor diverse vocabulary

        confidence = (length_score * 0.7 + diversity_score * 0.3)

        # Cap at reasonable maximum
        return min(confidence, 0.9)

    def _process_audio_batch(self, audio_bytes: bytes) -> str:
        """Process complete audio utterance in batch mode"""
        try:
            # Convert to temporary WAV file
            audio_path = self._audio_bytes_to_wav(audio_bytes)

            # Prepare generation parameters
            generate_kwargs = {}

            # Add temperature if supported
            if self._temperature != 0.0:
                # Try common temperature parameter names
                for temp_param in ["temperature", "temp", "sampling_temperature"]:
                    try:
                        # Test if the parameter is accepted
                        test_kwargs = generate_kwargs.copy()
                        test_kwargs[temp_param] = self._temperature
                        if PARAKEET_OLD_FORMAT:
                            result = self._model.generate(audio_path, **test_kwargs)
                        else:
                            from parakeet_mlx.audio import load_audio
                            audio_array = load_audio(audio_path, self._model.preprocessor_config.sample_rate)
                            result = self._model.generate(audio_array, **test_kwargs)
                        generate_kwargs = test_kwargs
                        logger.debug(f"Using temperature parameter '{temp_param}' with value {self._temperature}")
                        break
                    except TypeError:
                        continue
                else:
                    logger.debug("Temperature parameter not supported by Parakeet model, using default")

            # Generate transcription
            if PARAKEET_OLD_FORMAT:
                # Legacy mlx_audio API
                result = self._model.generate(audio_path, **generate_kwargs)
            else:
                # New parakeet_mlx API
                from parakeet_mlx.audio import load_audio
                audio_array = load_audio(audio_path, self._model.preprocessor_config.sample_rate)
                result = self._model.generate(audio_array, **generate_kwargs)

            # Extract text and confidence from result
            text = ""
            confidence = 0.0

            if hasattr(result, 'text'):
                text = result.text.strip()
            elif hasattr(result, 'transcription'):
                text = result.transcription.strip()
            else:
                logger.warning(f"Unexpected result format from Parakeet: {type(result)}")
                return ""

            # Try to extract confidence score
            if hasattr(result, 'confidence'):
                confidence = float(result.confidence)
            elif hasattr(result, 'score'):
                confidence = float(result.score)
            else:
                # Estimate confidence based on text characteristics
                confidence = self._estimate_confidence(text)

            logger.debug(f"Parakeet batch transcription: '{text}' (confidence: {confidence:.2f})")

            # Filter out low-confidence transcriptions
            if confidence < self._confidence_threshold:
                logger.debug(f"Filtered low-confidence transcription: '{text}' (confidence: {confidence:.2f} < {self._confidence_threshold})")
                return ""

            return text

        except Exception as e:
            logger.error(f"Parakeet batch inference failed: {e}")
            return ""
        finally:
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio and yield transcription frames"""
        try:
            # Process the complete audio utterance
            text = await asyncio.get_event_loop().run_in_executor(None, self._process_audio_batch, audio)

            if text:
                # Yield final transcription frame
                yield TranscriptionFrame(
                    text=text,
                    user_id=self._user_id or "user",
                    timestamp=str(time.time())
                )
            else:
                # No transcription available
                logger.debug("Parakeet batch: no transcription generated")

        except Exception as e:
            logger.error(f"Error in Parakeet batch STT: {e}")
            yield ErrorFrame(f"Parakeet batch STT error: {e}")