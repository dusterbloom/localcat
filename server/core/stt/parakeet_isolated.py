"""
Parakeet Isolated STT Service - Process-isolated transcription.
Communicates with parakeet_worker.py to avoid Metal threading conflicts.

Architecture:
- Main process (bot.py) runs this service
- Worker process (parakeet_worker.py) owns Parakeet model + Metal context
- Communication via JSON over stdin/stdout pipes
- Eliminates STT/TTS Metal conflicts through process boundaries
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService


class ParakeetIsolatedSTT(STTService):
    """
    Process-isolated Parakeet STT service for Metal conflict avoidance.

    Benefits:
    - Complete Metal context isolation from TTS
    - No locks needed (OS handles process isolation)
    - No deadlocks possible
    - Consistent with existing TTS worker architecture
    """

    def __init__(
        self,
        *,
        model_path: str = "mlx-community/parakeet-tdt-0.6b-v3",
        streaming: bool = True,
        context_size: tuple = (256, 256),
        depth: int = 3,
        chunk_duration: float = 3.0,  # Audio buffering duration (optimal: 2-4s)
        volume_threshold: float = 0.0005,  # Volume gating threshold
        beam_width: int = 8,  # Beam search width (optimal: 8-10)
        temperature: float = 0.0,  # Sampling temperature
        sentence_pause_threshold: float = 1.2,  # Sentence boundary threshold
        max_chunk_duration: float = 4.0,  # Maximum chunk duration
        enable_vad: bool = False,  # Internal VAD
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._model_path = model_path
        self._streaming = streaming
        self._context_size = context_size
        self._depth = depth
        self._chunk_duration = chunk_duration
        self._volume_threshold = volume_threshold
        self._beam_width = beam_width
        self._temperature = temperature
        self._sentence_pause_threshold = sentence_pause_threshold
        self._max_chunk_duration = max_chunk_duration
        self._enable_vad = enable_vad
        self._sample_rate = 16000  # Parakeet expects 16kHz

        # Worker process management
        self._process: Optional[subprocess.Popen] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._initialized = False

        # Audio buffering state (from streaming version)
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self._last_sent_length = 0  # Track how much text we've already sent
        self._current_turn_text = ""  # Track complete text for current turn

        # VAD state tracking
        self._vad_active = False
        self._last_finalized_time = 0.0

        # Hallucination filtering configuration (from streaming version)
        self._filter_hallucinations: bool = os.getenv("PARAKEET_FILTER_HALLUCINATIONS", "true").lower() in ("1", "true", "yes")
        try:
            self._interim_min_chars: int = int(os.getenv("PARAKEET_INTERIM_MIN_CHARS", "5"))
        except Exception:
            self._interim_min_chars = 5
        try:
            self._final_min_chars: int = int(os.getenv("PARAKEET_FINAL_MIN_CHARS", "6"))
        except Exception:
            self._final_min_chars = 6
        # Known Parakeet hallucination patterns
        self._hallucination_patterns = {
            "yeah", "yep", "yes", "mmhmm", "mmhmmm", "mhm", "uhhuh",
            "im just", "thank you", "thanks", "okay", "ok",
            "uh", "um", "hmm", "ah", "oh", "почему", "scary"
        }

        # Start worker process
        self._start_worker()

        logger.info(
            f"✅ Parakeet Isolated STT initialized (process-isolated, Metal conflict free)"
        )

    def _start_worker(self):
        """Launch the isolated worker process."""
        try:
            # Find worker script
            worker_path = Path(__file__).parent / "parakeet_worker.py"
            if not worker_path.exists():
                raise FileNotFoundError(f"Worker script not found: {worker_path}")

            # Launch worker process
            self._process = subprocess.Popen(
                [sys.executable, str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Binary mode for audio data
                bufsize=0,  # Unbuffered for low latency
            )

            logger.info(f"🚀 Started Parakeet worker process (PID: {self._process.pid})")

            # Start async reader task
            self._reader_task = asyncio.create_task(self._read_worker_output())

            # Initialize worker (command sent, will complete asynchronously)
            self._send_command(
                {
                    "cmd": "init",
                    "model_path": self._model_path,
                    "streaming": self._streaming,
                    "context_size": list(self._context_size),
                    "depth": self._depth,
                    "beam_width": self._beam_width,
                    "temperature": self._temperature,
                }
            )

            logger.info("⏳ Worker initialization started (will complete async)...")

        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            raise

    def _send_command(self, cmd: dict):
        """Send JSON command to worker process."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Worker process not running")

        try:
            cmd_json = json.dumps(cmd) + "\n"
            self._process.stdin.write(cmd_json.encode("utf-8"))
            self._process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send command to worker: {e}")
            raise

    def _normalize_text_for_filter(self, text: str) -> str:
        """Normalize text for hallucination filtering."""
        if not text:
            return ""
        t = text.strip().lower()
        # Remove punctuation for robust matching
        t = re.sub(r'[^\w\s]', '', t)
        return t

    def _is_hallucination_like(self, text: str, *, allow_short_word: bool = False) -> bool:
        """Check if text appears to be a hallucination."""
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
        """Normalize audio volume to optimal levels for transcription."""
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

    async def _read_worker_output(self):
        """Read and parse worker output asynchronously."""
        try:
            loop = asyncio.get_event_loop()

            while self._process and self._process.poll() is None:
                # Read line from worker stdout (blocking call in executor)
                line = await loop.run_in_executor(
                    None, self._process.stdout.readline
                )

                if not line:
                    break

                try:
                    msg = json.loads(line.decode("utf-8").strip())

                    # Handle different message types
                    if "status" in msg:
                        logger.debug(f"Worker: {msg['status']}")
                        if "initialized" in msg['status'].lower() or "loaded" in msg['status'].lower():
                            self._initialized = True

                    elif "success" in msg:
                        logger.debug(f"Worker initialized successfully")
                        self._initialized = True

                    elif "text" in msg:
                        # Transcription result
                        await self._output_queue.put(msg)

                    elif "error" in msg:
                        logger.error(f"Worker error: {msg['error']}")
                        await self._output_queue.put(msg)

                    elif "warning" in msg:
                        logger.warning(f"Worker warning: {msg['warning']}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse worker output: {e}")

        except Exception as e:
            logger.error(f"Worker output reader error: {e}")
        finally:
            logger.info("Worker output reader stopped")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio through isolated worker and yield transcription frames."""
        # Wait for initialization (non-blocking async wait)
        if not self._initialized:
            max_wait = 30.0  # 30 second timeout
            wait_start = time.time()
            while not self._initialized and (time.time() - wait_start) < max_wait:
                await asyncio.sleep(0.1)  # Non-blocking async sleep

            if not self._initialized:
                logger.error("Worker initialization timeout")
                yield ErrorFrame(error="STT worker initialization timeout")
                return

        try:
            # Convert audio bytes to numpy array and normalize to float32 [-1, 1]
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Apply volume normalization for better transcription quality
            audio_array = self._normalize_audio(audio_array)

            # Only process audio when VAD is active or check volume threshold
            should_process = self._vad_active
            if not should_process:
                # Check volume threshold when VAD not active
                if len(audio_array) > 0:
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    should_process = rms > self._volume_threshold

            if should_process:
                # Add audio to buffer
                self.audio_buffer.append(audio_array)
                self.buffer_duration += len(audio_array) / self._sample_rate

            # Only process if we have accumulated enough audio
            if self.buffer_duration >= self._chunk_duration:
                # Concatenate all buffered audio
                full_audio = np.concatenate(self.audio_buffer)

                # Convert back to int16 for transport
                audio_int16 = (full_audio * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Encode audio as base64 for JSON transport
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

                # Send transcribe command to worker
                self._send_command({"cmd": "transcribe", "audio": audio_b64})

                # Clear buffer after processing
                self.audio_buffer = []
                self.buffer_duration = 0.0

                # Wait for transcription result (with timeout)
                try:
                    result = await asyncio.wait_for(
                        self._output_queue.get(), timeout=2.0
                    )

                    if "error" in result:
                        logger.error(f"Worker transcription error: {result['error']}")
                        yield ErrorFrame(error=result["error"])
                        return

                    if "text" in result:
                        full_text = result["text"].strip()
                        # Note: is_final from worker - we handle as interim until finalization

                        if full_text:
                            # Safety check: if text seems to be accumulating from previous turns
                            if self._current_turn_text and full_text.startswith(self._current_turn_text):
                                # Text is building on current turn - this is normal
                                pass
                            elif self._current_turn_text and len(full_text) > len(self._current_turn_text) + 50:
                                # Text is much longer than expected - might be accumulating
                                logger.warning(f"Text accumulation detected, resetting...")
                                self._reset_transcriber_state()
                                return

                            # Calculate the new portion of text (avoid duplicates)
                            if len(full_text) > self._last_sent_length:
                                # Get only the new text that hasn't been sent yet
                                new_text = full_text[self._last_sent_length:]

                                if new_text:  # Only send if there's actual new content
                                    # Filter obvious hallucinations and trivial noise
                                    if (len(new_text.strip()) < self._interim_min_chars) or self._is_hallucination_like(new_text):
                                        # Skip sending, but don't update _last_sent_length
                                        pass
                                    else:
                                        # Send as interim transcription during speech
                                        frame = InterimTranscriptionFrame(
                                            text=new_text,
                                            user_id="",
                                            timestamp=str(time.time())
                                        )

                                        # Only log interim frames occasionally to reduce log spam
                                        if len(new_text.strip()) > 5:  # Log meaningful chunks
                                            logger.debug(f"[Parakeet Isolated STT] Interim: {new_text.strip()}")
                                        yield frame

                                        # Update tracking for next iteration
                                        self._last_sent_length = len(full_text)
                                        self._current_turn_text = full_text

                except asyncio.TimeoutError:
                    logger.warning("Worker transcription timeout")

        except Exception as e:
            logger.error(f"STT processing error: {e}")
            yield ErrorFrame(error=str(e))

    async def process_frame(self, frame: Frame, direction=None):
        """Handle VAD frames to gate transcription processing."""
        await super().process_frame(frame, direction)

        # Track VAD state
        if isinstance(frame, UserStartedSpeakingFrame):
            # Clear ALL state when user starts speaking to prevent carryover
            self._vad_active = True
            self._reset_transcriber_state()
            logger.debug("VAD: User started speaking (state reset)")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Finalize any pending transcription when user stops speaking
            now = time.time()
            since_last = now - self._last_finalized_time
            if since_last < 0.15:  # Debounce rapid stop events
                return

            self._vad_active = False
            logger.debug("VAD: User stopped speaking")

            # Flush any remaining buffered audio and transcription
            await self._finalize_pending_transcription()

    def _reset_transcriber_state(self):
        """Reset transcriber state for new conversation turn."""
        logger.debug("Resetting Parakeet transcriber for new turn")

        # Clear all tracking state
        self._last_sent_length = 0
        self._current_turn_text = ""
        self.audio_buffer = []
        self.buffer_duration = 0.0

        # Reset worker streaming context
        if self._streaming:
            try:
                self._send_command({"cmd": "reset"})
                logger.debug("Worker reset command sent")
            except Exception as e:
                logger.warning(f"Failed to reset worker: {e}")

    async def _finalize_pending_transcription(self):
        """Finalize any buffered transcription and audio."""
        try:
            # Process any remaining buffered audio first
            if self.audio_buffer and self.buffer_duration > 0:
                full_audio = np.concatenate(self.audio_buffer)

                # Convert back to int16 for transport
                audio_int16 = (full_audio * 32768.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Encode and send to worker
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                self._send_command({"cmd": "transcribe", "audio": audio_b64})

            # Send finalize command to get the complete transcription
            self._send_command({"cmd": "finalize"})

            # Wait for final result
            try:
                result = await asyncio.wait_for(
                    self._output_queue.get(), timeout=3.0
                )

                if "text" in result:
                    full_text = result["text"].strip()
                    if full_text:
                        # Skip final output if it looks like a hallucination or too short
                        if (len(self._normalize_text_for_filter(full_text)) < self._final_min_chars) or \
                           self._is_hallucination_like(full_text, allow_short_word=True):
                            # Treat as no meaningful speech captured
                            logger.debug("Final text filtered as hallucination")
                        else:
                            # Yield final transcription with complete accumulated text
                            frame = TranscriptionFrame(
                                text=full_text,
                                user_id="",
                                timestamp=str(time.time())
                            )
                            await self.push_frame(frame)
                            logger.info(f"[Parakeet Isolated STT] Final: {full_text}")
                    else:
                        logger.debug("Worker returned empty final transcription")

                elif "error" in result:
                    logger.error(f"Worker finalization error: {result['error']}")

            except asyncio.TimeoutError:
                logger.warning("Finalization timeout - no final transcription received")

            # Clear buffers and state
            self.audio_buffer = []
            self.buffer_duration = 0.0
            self._last_sent_length = 0
            self._current_turn_text = ""
            self._last_finalized_time = time.time()

        except Exception as e:
            logger.error(f"Error finalizing transcription: {e}")

    async def cleanup(self):
        """Cleanup worker process on shutdown."""
        logger.info("Shutting down Parakeet worker...")

        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        # Terminate worker process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Worker didn't terminate gracefully, killing...")
                self._process.kill()
            except Exception as e:
                logger.error(f"Error terminating worker: {e}")

        logger.info("Parakeet worker shut down")

    def __del__(self):
        """Ensure worker is cleaned up on deletion."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass
