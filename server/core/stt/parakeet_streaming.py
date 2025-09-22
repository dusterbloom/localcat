"""
Parakeet Streaming STT Service for Pipecat
Provides ultra-low latency streaming transcription with VAD and smart chunking.
"""

import asyncio
import numpy as np
import os
import tempfile
import time
import wave
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    ErrorFrame
)
from pipecat.services.ai_services import STTService

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


class ParakeetStreamingSTT(STTService):
    """
    Ultra-low latency streaming STT using Parakeet MLX model.
    Optimized for real-time voice agent conversations.
    """

    def __init__(self,
                 model_path: str = "mlx-community/parakeet-tdt-0.6b-v3",
                 language: str = "en",
                 chunk_duration: float = 1.0,
                 enable_vad: bool = True,
                 temperature: float = 0.0,
                 confidence_threshold: float = 0.2,
                 sentence_pause_threshold: float = 1.2,
                 max_chunk_duration: float = 4.0,
                 context_size: tuple = (256, 256),
                 depth: int = 3,
                 volume_threshold: float = 0.001,
                 **kwargs):
        super().__init__(**kwargs)

        self.model_path = model_path
        self.language = language
        self.chunk_duration = chunk_duration
        self.enable_vad = enable_vad
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.sentence_pause_threshold = sentence_pause_threshold
        self.max_chunk_duration = max_chunk_duration
        self.context_size = context_size
        self.depth = depth
        self.volume_threshold = volume_threshold

        # Model and processing state
        self._model = None
        self._processor = None
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.last_transcription_time = 0

        # Initialize model
        self._init_parakeet_model()

    def _init_parakeet_model(self):
        """Initialize Parakeet model and processor"""
        if not PARAKEET_AVAILABLE:
            raise ImportError("Parakeet MLX not available")

        try:
            logger.info(f"Loading Parakeet model: {self.model_path}")

            if PARAKEET_OLD_FORMAT:
                # Legacy mlx_audio format
                self._model = load_model(self.model_path)
                self._processor = None
            else:
                # New parakeet_mlx format
                self._model, self._processor = from_pretrained(self.model_path)

            logger.info("✅ Parakeet model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            raise

    def _audio_bytes_to_wav(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to WAV file for processing"""
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())

            return temp_file.name

    def _process_audio_file(self, audio_path: str) -> str:
        """Process audio file and return transcription"""
        try:
            # Generate transcription
            if PARAKEET_OLD_FORMAT:
                # Legacy mlx_audio API
                result = self._model.generate(audio_path)
            else:
                # New parakeet_mlx API
                audio_array = load_audio(audio_path, self._model.preprocessor_config.sample_rate)
                result = self._model.generate(audio_array)

            # Extract text from result
            text = ""
            if hasattr(result, 'text'):
                text = result.text.strip()
            elif hasattr(result, 'transcription'):
                text = result.transcription.strip()
            else:
                logger.warning(f"Unexpected result format from Parakeet: {type(result)}")
                return ""

            return text

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return ""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process streaming audio and yield transcription frames"""
        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer
            self.audio_buffer.append(audio_array)
            self.buffer_duration += len(audio_array) / 16000.0  # Assuming 16kHz

            # Process if we have enough audio
            current_time = time.time()
            if self.buffer_duration >= self.chunk_duration:
                # Concatenate buffered audio
                combined_audio = np.concatenate(self.audio_buffer)

                # Process audio
                transcription = await self._transcribe_audio(combined_audio)

                if transcription.strip():
                    # Create transcription frame
                    frame = TranscriptionFrame(text=transcription)
                    yield frame

                    # Reset buffer
                    self.audio_buffer = []
                    self.buffer_duration = 0.0
                    self.last_transcription_time = current_time

        except Exception as e:
            logger.error(f"Streaming STT error: {e}")
            yield ErrorFrame(error=f"STT processing failed: {e}")

    async def _transcribe_audio(self, audio_array: np.ndarray) -> str:
        """Transcribe audio array to text"""
        try:
            # Create temporary WAV file
            temp_wav = self._audio_bytes_to_wav(audio_array.tobytes())

            try:
                # Process the audio file
                transcription = self._process_audio_file(temp_wav)

                # Clean up
                os.unlink(temp_wav)

                return transcription.strip()

            finally:
                # Ensure cleanup
                try:
                    os.unlink(temp_wav)
                except:
                    pass

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""