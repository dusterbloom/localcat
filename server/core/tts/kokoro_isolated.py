"""
Kokoro Isolated TTS Service - Process-isolated text-to-speech.
Communicates with kokoro_worker.py to avoid Metal threading conflicts.

Architecture:
- Main process (bot.py) runs this service
- Worker process (kokoro_worker.py) owns Kokoro model + Metal context
- Communication via JSON over stdin/stdout pipes
- Eliminates STT/TTS Metal conflicts through process boundaries

This mirrors the proven parakeet_isolated.py pattern for consistency.
"""

import asyncio
import base64
import json
import os
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
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from tools.text_formatter import split_text_for_kokoro_streaming
from tools.audio_utils import convert_to_pcm16
from core.tts.kokoro_config import (
    CHUNK_MIN_LENGTH,
    CHUNK_MAX_LENGTH,
)


class KokoroIsolatedTTS(TTSService):
    """
    Process-isolated Kokoro TTS service for Metal conflict avoidance.

    Benefits:
    - Complete Metal context isolation from STT
    - No locks needed (OS handles process isolation)
    - No deadlocks possible
    - Graceful cancellation via subprocess termination
    - Consistent with parakeet_isolated STT architecture
    """

    def __init__(
        self,
        *,
        model_name: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_bella",
        speed: float = 1.0,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,  # Use sentence aggregation for smooth flow
            **kwargs
        )

        self._model_name = model_name
        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate

        # Worker process management
        self._process: Optional[subprocess.Popen] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._initialized = False

        # Cancellation flag for graceful interruption handling
        self._cancelled = False
        self._current_generation_id: Optional[str] = None

        # Start worker process
        self._start_worker()

        logger.info(
            f"✅ Kokoro Isolated TTS initialized (process-isolated, Metal conflict free) with voice: {self._voice}"
        )

    def _start_worker(self):
        """Launch the isolated worker process."""
        try:
            # Find worker script
            worker_path = Path(__file__).parent / "kokoro_worker.py"
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

            logger.info(f"🚀 Started Kokoro worker process (PID: {self._process.pid})")

            # Start async reader task
            self._reader_task = asyncio.create_task(self._read_worker_output())

            # Initialize worker (command sent, will complete asynchronously)
            self._send_command(
                {
                    "cmd": "init",
                    "model": self._model_name,
                    "voice": self._voice,
                }
            )

            logger.info("⏳ Worker initialization started (will complete async)...")

        except Exception as e:
            logger.error(f"Failed to start Kokoro worker: {e}")
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
                        # Get sample rate from worker config
                        if "config" in msg and "sample_rate" in msg["config"]:
                            self._sample_rate = msg["config"]["sample_rate"]

                    elif "chunk" in msg:
                        # Audio chunk result
                        await self._output_queue.put(msg)

                    elif "done" in msg:
                        # Generation complete marker
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

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using isolated Kokoro worker with optimal chunking."""

        # Reset cancellation flag at start of new TTS request
        self._cancelled = False
        self._current_generation_id = f"gen_{time.time()}"

        # Wait for initialization (non-blocking async wait)
        if not self._initialized:
            max_wait = 30.0  # 30 second timeout
            wait_start = time.time()
            while not self._initialized and (time.time() - wait_start) < max_wait:
                await asyncio.sleep(0.1)  # Non-blocking async sleep

            if not self._initialized:
                logger.error("Worker initialization timeout")
                yield ErrorFrame(error="TTS worker initialization timeout")
                return

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # Split text into optimal chunks for Kokoro
        sentences = split_text_for_kokoro_streaming(
            text,
            min_length=CHUNK_MIN_LENGTH,
            max_length=CHUNK_MAX_LENGTH
        )

        if not sentences:
            logger.debug(f"🔇 Skipping TTS for empty text: '{text}'")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        logger.debug(f"🎤 Isolated TTS for {len(sentences)} chunks: {sentences[0][:40]}{'...' if len(sentences[0]) > 40 else ''}")

        overall_start_time = time.time()
        first_audio_sent = False

        yield TTSStartedFrame()

        try:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                # Check cancellation flag
                if self._cancelled:
                    logger.debug("TTS generation cancelled by user interruption")
                    break

                chunk_start_time = time.time()

                # Mirror the exact text chunk to transcript/UI
                from pipecat.frames.frames import TTSTextFrame
                yield TTSTextFrame(text=sentence)

                # Send generation command to worker
                self._send_command({
                    "cmd": "generate",
                    "text": sentence,
                    "speed": self._speed,
                    "generation_id": self._current_generation_id,
                })

                # Collect audio chunks for this sentence
                sentence_audio_chunks = []
                generation_done = False

                while not generation_done and not self._cancelled:
                    try:
                        # Wait for worker output with timeout
                        msg = await asyncio.wait_for(
                            self._output_queue.get(),
                            timeout=10.0  # 10 second timeout per chunk
                        )

                        if "chunk" in msg:
                            # Decode base64 audio chunk
                            chunk_b64 = msg["chunk"]
                            audio_bytes = base64.b64decode(chunk_b64)
                            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                            sentence_audio_chunks.append(audio_data)

                        elif "done" in msg:
                            generation_done = True

                        elif "error" in msg:
                            logger.error(f"Worker generation error: {msg['error']}")
                            yield ErrorFrame(error=msg['error'])
                            return

                    except asyncio.TimeoutError:
                        logger.error(f"Timeout waiting for worker audio chunk {i+1}")
                        break

                # Concatenate and yield audio for this sentence
                if sentence_audio_chunks:
                    # Calculate TTFB for first chunk only
                    if not first_audio_sent:
                        ttfb = (time.time() - overall_start_time) * 1000
                        logger.debug(f"🚀 KOKORO ISOLATED TTFB: {ttfb:.1f}ms")
                        first_audio_sent = True

                    chunk_latency = (time.time() - chunk_start_time) * 1000
                    logger.debug(f"✅ Isolated Chunk {i+1}/{len(sentences)}: {len(sentence)} chars → {chunk_latency:.1f}ms")

                    # Concatenate all audio chunks for this sentence
                    audio_int16 = np.concatenate(sentence_audio_chunks)

                    frame = TTSAudioRawFrame(
                        audio=audio_int16.tobytes(),
                        sample_rate=self._sample_rate,
                        num_channels=1
                    )

                    yield frame
                else:
                    logger.warning(f"No audio generated for chunk: '{sentence}'")

        except asyncio.CancelledError:
            # Set cancellation flag to stop any in-progress generation
            self._cancelled = False
            logger.debug("Isolated TTS generation cancelled - terminating worker task")
            raise
        except Exception as e:
            logger.error(f"Kokoro Isolated TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()

    async def cancel(self, frame):
        """Cancel ongoing TTS generation."""
        self._cancelled = True
        logger.debug(f"Cancellation requested for Kokoro Isolated TTS: {frame.id}")

        # Terminate worker process immediately for clean cancellation
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()

        # Clear any pending messages in queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except:
                break

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup worker process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception as e:
                logger.warning(f"Error terminating worker: {e}")

    def __del__(self):
        """Cleanup worker process on deletion (fallback for non-context-manager usage)."""
        try:
            if hasattr(self, '_process') and self._process:
                if self._process.poll() is None:  # Process still running
                    self._process.terminate()
        except Exception:
            pass  # Ignore errors during cleanup
