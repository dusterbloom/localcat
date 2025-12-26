"""
Supertonic TTS Service - Lightning-fast on-device TTS.

Features:
- 30-45x faster than real-time on Apple Silicon
- Automatic handling of numbers, dates, currency, abbreviations
- 44.1kHz high-quality audio output
- ONNX-based (bundle compatible, no Metal/GPU issues)
- MIT licensed

Performance (typical):
- "Hello, world!" → 55ms synthesis, 1.6s audio (30x real-time)
- Complex text with $99.99, dates → 130ms synthesis (45x real-time)

References:
- https://github.com/supertone-inc/supertonic
- https://huggingface.co/Supertone/supertonic
"""

import asyncio
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


class SupertonicTTSService(TTSService):
    """
    Supertonic TTS with Pipecat integration.

    Supertonic is a 66M parameter TTS model that achieves 30-45x real-time
    performance on consumer hardware. It handles complex text (numbers, dates,
    currency) automatically without preprocessing.

    Args:
        voice: Voice style to use (M1-M5 for male, F1-F5 for female)
        model_dir: Custom model directory (for bundled apps)
        total_steps: Denoising steps (2=fast, 5=higher quality)
        speed: Speech rate factor (0.5-2.0, default 1.0)
        target_sample_rate: Output sample rate (default 24000 for pipeline compat)
        aggregate_sentences: Whether to aggregate sentences before synthesis
    """

    VOICES = {"M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"}
    NATIVE_SAMPLE_RATE = 44100

    def __init__(
        self,
        *,
        voice: str = "F1",
        model_dir: Optional[str] = None,
        total_steps: int = 2,
        speed: float = 1.0,
        target_sample_rate: int = 24000,
        aggregate_sentences: bool = True,
        **kwargs,
    ):
        super().__init__(
            sample_rate=target_sample_rate,
            aggregate_sentences=aggregate_sentences,
            push_text_frames=True,
            **kwargs,
        )

        if voice not in self.VOICES:
            logger.warning(f"Unknown voice '{voice}', defaulting to F1")
            voice = "F1"

        self._voice_name = voice
        self._model_dir = model_dir
        self._total_steps = max(1, min(total_steps, 10))
        self._speed = max(0.5, min(speed, 2.0))
        self._target_sample_rate = target_sample_rate

        # Lazy-loaded components
        self._tts = None
        self._voice_style = None
        self._resampler = None

        logger.info(
            f"SupertonicTTS initialized: voice={voice}, steps={self._total_steps}, "
            f"speed={self._speed}, target_sr={target_sample_rate}"
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def _ensure_loaded(self):
        """Lazy-load Supertonic TTS model."""
        if self._tts is not None:
            return

        logger.info(f"Loading Supertonic TTS (voice={self._voice_name})...")
        start = time.perf_counter()

        try:
            from supertonic import TTS

            # Load TTS with optional custom model directory
            if self._model_dir:
                self._tts = TTS(model_dir=self._model_dir, auto_download=False)
                logger.info(f"Loaded Supertonic from custom path: {self._model_dir}")
            else:
                self._tts = TTS(auto_download=True)

            # Pre-load voice style
            self._voice_style = self._tts.get_voice_style(self._voice_name)

            elapsed = time.perf_counter() - start
            logger.info(f"Supertonic TTS loaded in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load Supertonic TTS: {e}")
            raise

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample from 44.1kHz to target sample rate."""
        if self._target_sample_rate == self.NATIVE_SAMPLE_RATE:
            return audio

        try:
            from scipy import signal

            ratio = self._target_sample_rate / self.NATIVE_SAMPLE_RATE
            new_length = int(audio.shape[-1] * ratio)

            # Handle both (1, N) and (N,) shapes
            if len(audio.shape) > 1:
                audio = audio.squeeze()

            resampled = signal.resample(audio, new_length)
            return resampled.astype(np.float32)

        except ImportError:
            # Fallback: simple decimation (lower quality but no scipy needed)
            logger.warning("scipy not available, using simple decimation for resampling")
            step = self.NATIVE_SAMPLE_RATE // self._target_sample_rate
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            return audio[::step].astype(np.float32)

    def _to_int16(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to int16 bytes."""
        # Ensure audio is float32 in range [-1, 1]
        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Synthesize text to speech."""
        await self._ensure_loaded()

        text = text.strip()
        if not text:
            return

        yield TTSStartedFrame()

        try:
            start_time = time.perf_counter()

            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            wav, _ = await loop.run_in_executor(
                None,
                lambda: self._tts.synthesize(
                    text,
                    self._voice_style,
                    total_steps=self._total_steps,
                    speed=self._speed,
                    verbose=False,
                ),
            )

            synthesis_time = time.perf_counter() - start_time

            # Resample if needed
            if self._target_sample_rate != self.NATIVE_SAMPLE_RATE:
                wav = self._resample(wav)

            # Calculate metrics
            num_samples = wav.shape[-1] if len(wav.shape) > 1 else len(wav)
            audio_duration = num_samples / self._target_sample_rate
            rtf = synthesis_time / audio_duration if audio_duration > 0 else 0

            logger.debug(
                f"Supertonic synthesis: {len(text)} chars → "
                f"{synthesis_time*1000:.1f}ms, {audio_duration:.2f}s audio, "
                f"RTF={rtf:.4f} ({1/rtf:.0f}x real-time)"
            )

            # Convert to int16 bytes
            audio_bytes = self._to_int16(wav)

            # Yield in chunks for streaming behavior
            chunk_size = 4096 * 2  # 4096 samples * 2 bytes per sample
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._target_sample_rate,
                    num_channels=1,
                )

        except Exception as e:
            logger.error(f"Supertonic TTS error: {e}")
            yield ErrorFrame(error=str(e))

        finally:
            yield TTSStoppedFrame()

    async def set_voice(self, voice: str):
        """Change the voice style."""
        if voice not in self.VOICES:
            logger.warning(f"Unknown voice '{voice}', keeping {self._voice_name}")
            return

        if self._tts is not None:
            self._voice_style = self._tts.get_voice_style(voice)
        self._voice_name = voice
        logger.info(f"Supertonic voice changed to: {voice}")
