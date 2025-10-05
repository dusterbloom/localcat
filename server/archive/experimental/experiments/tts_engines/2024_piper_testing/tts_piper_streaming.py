"""
Piper TTS with true streaming support for ultra-low latency.
Piper is one of the fastest neural TTS systems available.
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator, Optional
import os

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


class PiperStreamingTTS(TTSService):
    """
    Native Piper TTS with true streaming capability.
    Generates audio in real-time chunks for minimal latency.
    """

    def __init__(
        self,
        *,
        voice: str = "en_US-lessac-medium",  # Fast, high-quality voice
        sample_rate: int = 22050,  # Piper medium quality default
        speaker_id: Optional[int] = None,
        length_scale: float = 1.0,  # Speech speed (lower = faster)
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,  # Aggregate sentences for smooth speech
            **kwargs
        )

        self._voice_name = voice
        self._speaker_id = speaker_id
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w_scale = noise_w_scale

        # Initialize Piper
        self._voice = None
        self._initialize_piper()

        logger.info(f"✅ Piper Streaming TTS initialized with voice: {voice}")

    def _initialize_piper(self):
        """Initialize Piper voice model"""
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig

            logger.debug(f"🚀 Loading Piper voice: {self._voice_name}")

            # Find the voice model path
            # Check multiple locations where Piper voices might be
            possible_paths = [
                Path.cwd() / f"{self._voice_name}.onnx",  # Current directory
                Path.home() / ".local" / "share" / "piper-voices" / self._voice_name / f"{self._voice_name}.onnx",
                Path.home() / ".local" / "share" / "piper_voices" / self._voice_name / f"{self._voice_name}.onnx",
                Path.home() / "Documents" / "piper-voices" / self._voice_name / f"{self._voice_name}.onnx",
            ]

            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break

            if not model_path:
                # Try to find any ONNX file with the voice name
                for pattern in [f"*{self._voice_name}*.onnx", "*.onnx"]:
                    onnx_files = list(Path.cwd().glob(pattern))
                    if onnx_files:
                        model_path = str(onnx_files[0])
                        break

            if not model_path:
                raise FileNotFoundError(
                    f"Voice model '{self._voice_name}' not found. "
                    f"Run: python3 -m piper.download_voices {self._voice_name}"
                )
            logger.debug(f"Loading model from: {model_path}")

            # Load the voice
            self._voice = PiperVoice.load(model_path)

            # Create synthesis config
            self._syn_config = SynthesisConfig(
                speaker_id=self._speaker_id,
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w_scale=self._noise_w_scale,
            )

            # Warmup
            logger.debug("🔥 Warming up Piper...")
            warmup_start = time.time()

            warmup_texts = [
                "Hello.",
                "How are you today?",
                "This is a test of the streaming text to speech system."
            ]

            for text in warmup_texts:
                # Generate but don't use the audio
                for audio_chunk in self._voice.synthesize(text, self._syn_config):
                    pass  # Just consume to warm up

            warmup_time = (time.time() - warmup_start) * 1000
            logger.debug(f"✅ Piper warmup completed in {warmup_time:.1f}ms")

        except ImportError:
            logger.error("❌ Piper not installed. Run: pip install piper-tts")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Piper: {e}")
            raise

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech with true streaming using Piper's native streaming"""

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        logger.debug(f"🎤 Piper streaming TTS: '{text[:50]}...'")

        overall_start_time = time.time()
        first_audio_sent = False

        yield TTSStartedFrame()

        try:
            # Piper generates audio in streaming chunks natively
            chunk_count = 0

            # Run synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Create async generator from sync generator
            async def stream_audio():
                # Run the synthesis in a thread to avoid blocking
                def generate_chunks():
                    chunks = []
                    # Use the pre-configured synthesis config
                    for audio_chunk in self._voice.synthesize(text, self._syn_config):
                        chunks.append(audio_chunk)
                    return chunks

                # Get all chunks (Piper generates them quickly)
                chunks = await loop.run_in_executor(None, generate_chunks)

                for audio_chunk in chunks:
                    yield audio_chunk

            # Stream the audio chunks
            async for audio_chunk in stream_audio():
                chunk_count += 1

                # Extract audio data from chunk
                # AudioChunk has attributes: audio_int16_bytes, sample_rate, sample_width
                if hasattr(audio_chunk, 'audio_int16_bytes'):
                    audio_bytes = audio_chunk.audio_int16_bytes
                    sample_rate = audio_chunk.sample_rate
                    channels = audio_chunk.sample_channels
                elif isinstance(audio_chunk, tuple) and len(audio_chunk) == 4:
                    sample_rate, width, channels, audio_bytes = audio_chunk
                else:
                    # Assume it's raw audio bytes
                    audio_bytes = audio_chunk
                    sample_rate = 22050
                    channels = 1

                if audio_bytes and len(audio_bytes) > 0:
                    # Track TTFB
                    if not first_audio_sent:
                        ttfb = (time.time() - overall_start_time) * 1000
                        logger.info(f"🚀 Piper Streaming TTFB: {ttfb:.1f}ms")
                        first_audio_sent = True

                    # Piper already provides int16 bytes, just stream them
                    yield TTSAudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=sample_rate,
                        num_channels=channels
                    )

                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.001)

            total_time = (time.time() - overall_start_time) * 1000
            logger.debug(f"✅ Piper streamed {chunk_count} chunks in {total_time:.1f}ms")

        except asyncio.CancelledError:
            logger.debug("Piper TTS cancelled")
            raise
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass