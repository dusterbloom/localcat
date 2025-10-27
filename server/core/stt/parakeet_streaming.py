"""
Parakeet Streaming STT Service for Pipecat
Provides ultra-low latency streaming transcription with VAD and smart chunking.

Enhancements:
- Hallucination suppression (short/noise-like outputs are filtered)
- Env-tunable thresholds for interim/final text length and volume gating
"""

import numpy as np
import time
import os
import re
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    ErrorFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame
)
from pipecat.services.ai_services import STTService

from core.utils.mlx_lock import MLX_GLOBAL_LOCK

try:
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import load_audio
    import mlx.core as mx
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
                 chunk_duration: float = 1.0,  # Reduced for more responsive transcription
                 enable_vad: bool = False,
                 temperature: float = 0.1,
                 sentence_pause_threshold: float = 1.2,
                 max_chunk_duration: float = 2.0,
                 context_size: tuple = (256, 256),
                 depth: int = 3,  # Default to 1 for streaming
                 volume_threshold: float = 0.0005,  # Less sensitive for quiet speech
                 **kwargs):
        super().__init__(**kwargs)

        self.model_path = model_path
        self.language = language
        self.chunk_duration = chunk_duration
        self.enable_vad = enable_vad
        self.temperature = temperature
        self.sentence_pause_threshold = sentence_pause_threshold
        self.max_chunk_duration = max_chunk_duration
        self.context_size = context_size
        self.depth = depth
        self.volume_threshold = volume_threshold

        # Warn if internal VAD enabled — external Silero VAD is already gating frames
        # Running double VAD can cause premature cutoffs or missed speech in noisy rooms.
        if self.enable_vad:
            logger.warning("[Parakeet STT] Internal VAD enabled while pipeline typically uses external Silero VAD."
                           " Consider disabling PARakeet internal VAD to avoid conflicting gating.")

        # Model and processing state
        self._model = None
        self._processor = None
        self._transcriber = None  # Streaming transcriber context
        self._transcriber_context = None
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.last_transcription_time = 0
        self._last_sent_length = 0  # Track how much text we've already sent
        self._current_turn_text = ""  # Track complete text for current turn

        # VAD state tracking
        self._vad_active = False  # Track if we're between start/stop speaking frames
        self._last_finalized_time = 0.0

        # Hallucination filtering configuration (env-tunable)
        self._filter_hallucinations: bool = os.getenv("PARAKEET_FILTER_HALLUCINATIONS", "true").lower() in ("1", "true", "yes")
        try:
            self._interim_min_chars: int = int(os.getenv("PARAKEET_INTERIM_MIN_CHARS", "5"))
        except Exception:
            self._interim_min_chars = 5
        try:
            self._final_min_chars: int = int(os.getenv("PARAKEET_FINAL_MIN_CHARS", "6"))
        except Exception:
            self._final_min_chars = 6
        # Known Parakeet hallucination patterns (align with batch service)
        self._hallucination_patterns = {
            "yeah", "yep", "yes", "mmhmm", "mmhmmm", "mhm", "uhhuh",
            "im just", "thank you", "thanks", "okay", "ok",
            "uh", "um", "hmm", "ah", "oh", "почему", "scary"
        }

        # Initialize model
        self._init_parakeet_model()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    def _normalize_text_for_filter(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip().lower()
        # Remove punctuation for robust matching
        t = re.sub(r'[^\w\s]', '', t)
        return t

    def _is_hallucination_like(self, text: str, *, allow_short_word: bool = False) -> bool:
        if not self._filter_hallucinations:
            return False
        if not text:
            return True
        norm = self._normalize_text_for_filter(text)
        if not norm:
            return True
        # Exact match against known short noise-like tokens
        if norm in self._hallucination_patterns:
            return True
        words = norm.split()
        # Very short single-word snippets are often noise; allow override
        if not allow_short_word and len(words) == 1 and len(words[0]) <= 3:
            return True
        return False

    def _normalize_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """Normalize audio volume to optimal levels for transcription"""
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

            return normalized

        return audio_np

    def _init_parakeet_model(self):
        """Initialize Parakeet model and streaming transcriber"""
        if not PARAKEET_AVAILABLE:
            raise ImportError("Parakeet MLX not available")

        try:
            logger.info(f"🔒 Acquiring MLX lock for Parakeet streaming initialization...")
            with MLX_GLOBAL_LOCK:
                logger.info(f"Loading Parakeet model: {self.model_path}")

                if PARAKEET_OLD_FORMAT:
                    # Legacy mlx_audio format - not supported for streaming
                    raise ImportError("Legacy Parakeet format not supported for streaming")
                else:
                    # New parakeet_mlx format - handle variable return values
                    result = from_pretrained(self.model_path)
                    if isinstance(result, tuple):
                        if len(result) >= 2:
                            self._model, self._processor = result[0], result[1]
                        elif len(result) == 1:
                            self._model = result[0]
                            self._processor = None
                        else:
                            raise ValueError(f"Unexpected return from from_pretrained: {result}")
                    else:
                        # Single return value
                        self._model = result
                        self._processor = None

                # Create streaming transcriber context and enter it
                self._transcriber_context = self._model.transcribe_stream(
                    context_size=self.context_size,
                    depth=self.depth,
                    keep_original_attention=False  # Use local attention for streaming
                )

                # Enter the context manager to get the actual transcriber
                self._transcriber = self._transcriber_context.__enter__()

                logger.info("✅ Parakeet streaming model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            raise





    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process streaming audio and yield transcription frames"""
        try:
            # Convert audio bytes to numpy array and normalize to float32 [-1, 1]
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Apply volume normalization for better transcription quality
            audio_array = self._normalize_audio(audio_array)

            # Only process audio when VAD is active (when using external VAD)
            # or when internal VAD is enabled
            should_process = False

            if self.enable_vad:
                # Use internal volume thresholding when VAD is enabled
                if len(audio_array) > 0:
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    should_process = rms > self.volume_threshold
            else:
                # Use external VAD state when internal VAD is disabled
                should_process = self._vad_active

            if should_process:
                # Add audio to buffer
                self.audio_buffer.append(audio_array)
                self.buffer_duration += len(audio_array) / 16000.0  # Track at 16kHz rate

            # Only process if we have accumulated enough audio
            if not PARAKEET_OLD_FORMAT and self.buffer_duration >= self.chunk_duration:
                # Concatenate all buffered audio
                full_audio = np.concatenate(self.audio_buffer)

                # Convert to MLX array and feed to transcriber
                audio_mlx = mx.array(full_audio)
                self._transcriber.add_audio(audio_mlx)

                # Clear buffer after processing
                self.audio_buffer = []
                self.buffer_duration = 0.0

                # Get current transcription result
                result = self._transcriber.result

                # Extract text and yield appropriate frame type
                if hasattr(result, 'text'):
                    full_text = result.text.strip()

                    if full_text:
                        # Safety check: if text seems to be accumulating from previous turns,
                        # only take the new portion that makes sense
                        if self._current_turn_text and full_text.startswith(self._current_turn_text):
                            # Text is building on current turn - this is normal
                            pass
                        elif self._current_turn_text and len(full_text) > len(self._current_turn_text) + 50:
                            # Text is much longer than expected - might be accumulating from previous turns
                            logger.warning(f"Text accumulation detected: current='{self._current_turn_text}', got='{full_text[:100]}...'")
                            # Reset and start fresh
                            self.reset_transcriber()
                            return

                        # Calculate the new portion of text (avoid duplicates)
                        if len(full_text) > self._last_sent_length:
                            # Get only the new text that hasn't been sent yet
                            new_text = full_text[self._last_sent_length:]

                            if new_text:  # Only send if there's actual new content
                                # Filter obvious hallucinations and trivial noise
                                if (len(new_text.strip()) < self._interim_min_chars) or self._is_hallucination_like(new_text):
                                    # Do not update _last_sent_length here so we can accumulate more before sending
                                    pass
                                else:
                                    # Send as interim transcription during speech
                                    frame = InterimTranscriptionFrame(
                                        text=new_text,
                                        user_id=getattr(self, '_user_id', None) or "user",
                                        timestamp=str(time.time())
                                    )

                                    # Only log interim frames occasionally to reduce log spam
                                    if len(new_text.strip()) > 5:  # Log meaningful chunks
                                        logger.debug(f"[Parakeet STT] Interim: {new_text.strip()}")
                                    yield frame

                                    # Update tracking for next iteration
                                    self._last_sent_length = len(full_text)
                                    self._current_turn_text = full_text  # Track complete turn text

        except Exception as e:
            logger.error(f"Streaming STT error: {e}")
            yield ErrorFrame(error=f"STT processing failed: {e}")

    async def process_frame(self, frame: Frame, direction=None):
        """Handle VAD frames to gate transcription processing"""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Clear ALL state when user starts speaking to prevent carryover
            self._vad_active = True
            self.reset_transcriber()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Finalize any pending transcription when user stops speaking
            now = time.time()
            since_last = now - self._last_finalized_time
            if since_last < 0.15:
                return
            self._vad_active = False
            # Flush any remaining buffered audio and transcription
            await self._finalize_pending_transcription()

    async def _finalize_pending_transcription(self):
        """Finalize any buffered transcription and audio"""
        try:
            # Process any remaining buffered audio first
            if self.audio_buffer and self.buffer_duration > 0:
                full_audio = np.concatenate(self.audio_buffer)
                audio_mlx = mx.array(full_audio)
                self._transcriber.add_audio(audio_mlx)

            # Always get the final transcription result
            result = self._transcriber.result
            if hasattr(result, 'text'):
                full_text = result.text.strip()
                if full_text:
                    # Skip final output if it looks like a hallucination or too short to be useful
                    if (len(self._normalize_text_for_filter(full_text)) < self._final_min_chars) or self._is_hallucination_like(full_text, allow_short_word=True):
                        # Treat as no meaningful speech captured
                        self._last_sent_length = 0
                        self.audio_buffer = []
                        self.buffer_duration = 0.0
                        self._last_finalized_time = time.time()
                        self._current_turn_text = ""
                    else:
                        # Yield final transcription with complete accumulated text
                        frame = TranscriptionFrame(
                            text=full_text,
                            user_id=getattr(self, '_user_id', None) or "user",
                            timestamp=str(time.time())
                        )
                        await self.push_frame(frame)
                        self._last_sent_length = 0  # Reset for next utterance
                        logger.info(f"[Parakeet STT] Final: {full_text}")

            # Clear buffers
            self.audio_buffer = []
            self.buffer_duration = 0.0
            self._last_finalized_time = time.time()
            self._current_turn_text = ""

        except Exception as e:
            logger.error(f"Error finalizing transcription: {e}")

    async def flush(self) -> AsyncGenerator[Frame, None]:
        """Flush any remaining transcription when speech ends"""
        if hasattr(self, '_transcriber') and self._transcriber:
            # Get final result
            result = self._transcriber.result

            if hasattr(result, 'text'):
                full_text = result.text.strip()

                # Send any remaining unsent text as final
                if full_text and len(full_text) > self._last_sent_length:
                    final_text = full_text[self._last_sent_length:]

                    if final_text.strip():
                        frame = TranscriptionFrame(
                            text=final_text.strip(),
                            user_id=getattr(self, '_user_id', None) or "user",
                            timestamp=str(time.time())
                        )
                        logger.debug(f"[Parakeet STT] Flush final: {final_text.strip()}")
                        yield frame

            # Reset for next utterance
            self._last_sent_length = 0
            self._current_turn_text = ""
            self.audio_buffer = []
            self.buffer_duration = 0.0

            # Reset the transcriber state
            if hasattr(self._transcriber, 'reset'):
                self._transcriber.reset()

    def _cleanup(self):
        """Clean up streaming transcriber"""
        if hasattr(self, '_transcriber_context') and self._transcriber_context:
            try:
                self._transcriber_context.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing transcriber: {e}")
        self._transcriber = None
        self._transcriber_context = None

    def reset_transcriber(self):
        """Reset transcriber state for new conversation turn"""
        logger.debug("Resetting Parakeet transcriber for new turn")

        # Clear all tracking state
        self._last_sent_length = 0
        self._current_turn_text = ""
        self.audio_buffer = []
        self.buffer_duration = 0.0

        # Reset existing transcriber if available
        if hasattr(self, '_transcriber') and self._transcriber:
            if hasattr(self._transcriber, 'reset'):
                try:
                    self._transcriber.reset()
                    logger.debug("Transcriber reset successful")
                except Exception as e:
                    logger.warning(f"Failed to reset transcriber: {e}")

        # Recreate streaming context for completely fresh state
        if hasattr(self, '_transcriber_context') and self._transcriber_context:
            try:
                self._transcriber_context.__exit__(None, None, None)
                logger.debug("Previous transcriber context closed")
            except Exception as e:
                logger.warning(f"Error closing transcriber context: {e}")

        # Create fresh transcriber context
        if self._model:
            try:
                self._transcriber_context = self._model.transcribe_stream(
                    context_size=self.context_size,
                    depth=self.depth,
                    keep_original_attention=False
                )
                self._transcriber = self._transcriber_context.__enter__()
                logger.debug("New transcriber context created successfully")
            except Exception as e:
                logger.error(f"Failed to create new transcriber context: {e}")
