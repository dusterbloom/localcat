"""Ultra-low latency Kokoro TTS service with optimized streaming."""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from tools.text_formatter import sanitize_for_voice, chunk_for_kokoro_ultra_low_latency


class TTSMLXUltraLowLatency(TTSService):
    """Ultra-low latency Kokoro TTS with 40-80ms time-to-first-byte."""

    def __init__(
        self,
        *,
        model: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        device: Optional[str] = None,
        sample_rate: int = 24000,
        speed: float = 1.0,
        use_boundaries: bool = True,  # Use sentence boundary detection
        buffer_ms: int = 40,  # Reduced buffer size for lower latency
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._device = device
        self._speed = speed
        self._use_boundaries = use_boundaries
        self._buffer_ms = buffer_ms

        self._process: Optional[subprocess.Popen[str]] = None
        self._initialized = False
        self._worker_script = self._get_worker_script_path()

        # Metrics tracking
        self._ttfb_ms = None
        self._total_chunks = 0

        # Barge-in cancellation support
        self._cancel_event: asyncio.Event = asyncio.Event()

        # Interruption state tracking
        self._interrupted: bool = False

    async def request_cancel(self) -> None:
        """Signal the TTS stream to stop as soon as possible (barge-in)."""
        try:
            self._cancel_event.set()
        except Exception:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with interruption handling.

        Handles UserStartedSpeakingFrame to enter interrupted state and cancel TTS,
        UserStoppedSpeakingFrame to exit interrupted state, and filters TextFrames
        during interruption to prevent queued text from being spoken.
        """
        # Handle interruption lifecycle
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("TTS: User started speaking - entering interrupted state")
            self._interrupted = True
            await self.request_cancel()  # Cancel current generation immediately
            await self._handle_interruption(frame, direction)  # Clear text aggregator
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("TTS: User stopped speaking - exiting interrupted state")
            self._interrupted = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            logger.debug("TTS: Received InterruptionFrame - clearing text aggregator")
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            # Drop TextFrames during interruption to prevent queued speech
            if self._interrupted:
                logger.debug(f"TTS: Dropping TextFrame during interruption: '{frame.text[:50]}...'")
                return  # Silently drop the frame
            # Normal processing
            await super().process_frame(frame, direction)

        else:
            # Pass all other frames through normally
            await super().process_frame(frame, direction)

    async def _handle_interruption(self, frame: Frame, direction: FrameDirection):
        """Handle interruption by clearing text aggregator and canceling generation."""
        self._processing_text = False
        if hasattr(self, '_text_aggregator'):
            await self._text_aggregator.handle_interruption()
        logger.debug("TTS: Interruption handled - text aggregator cleared")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    def _get_worker_script_path(self) -> str:
        current_dir = Path(__file__).parent
        # Use optimized worker
        worker_path = current_dir / "kokoro_worker_optimized.py"
        if not worker_path.exists():
            # Fallback to regular worker
            worker_path = current_dir / "kokoro_worker.py"
            logger.warning(f"Optimized worker not found, using regular worker")
        return str(worker_path)

    def _start_worker(self) -> bool:
        try:
            # Pass through environment variables - respects .env configuration
            env = os.environ.copy()
            # Only override buffer_ms if explicitly set in __init__
            env["TTS_BUFFER_MS"] = str(self._buffer_ms)
            # Ensure prewarming and lazy allocation for optimal performance
            env["TTS_PREWARM"] = os.getenv("TTS_PREWARM", "true")
            env["MLX_GPU_ALLOCATOR"] = "lazy"

            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for lowest latency
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,  # Process group isolation
            )
            logger.info(f"🚀 Started optimized Kokoro worker (pid={self._process.pid}, buffer={self._buffer_ms}ms)")
            return True
        except Exception as exc:
            logger.error(f"Failed to start Kokoro worker: {exc}")
            return False

    def _ensure_worker(self) -> bool:
        if self._process and self._process.poll() is None:
            return True
        return self._start_worker()

    async def _initialize_if_needed(self) -> bool:
        if self._initialized:
            return True
        if not self._ensure_worker():
            return False

        loop = asyncio.get_event_loop()
        command = json.dumps({"cmd": "init", "model": self._model_name, "voice": self._voice}) + "\n"

        # Use non-blocking write for lower latency
        await loop.run_in_executor(None, self._process.stdin.write, command)
        await loop.run_in_executor(None, self._process.stdin.flush)

        # Read response
        line = await loop.run_in_executor(None, self._process.stdout.readline)
        if not line:
            logger.error("Kokoro worker did not respond to init")
            return False

        resp = json.loads(line.strip())
        if resp.get("success"):
            config = resp.get("config", {})
            logger.info(f"Kokoro initialized with ultra-low latency config: {config}")
            self._initialized = True
            return True

        logger.error(f"Kokoro init failed: {resp.get('error')}")
        return False

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech with ultra-low latency streaming."""
        cleaned = sanitize_for_voice(text)

        # CRITICAL: Pre-chunk text for ultra-low latency (25 char chunks)
        # This prevents worker from buffering large audio chunks
        chunks = chunk_for_kokoro_ultra_low_latency(cleaned, max_chars=250)

        if not chunks:
            logger.debug(f"Skipping TTS for empty text: '{text}'")
            return

        logger.debug(f"TTS input: {len(chunks)} chunks from '{cleaned[:80]}...'")

        try:
            # Initialize variables that may be accessed in finally block
            ttfb_stopped = False

            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro worker")

            # Clear previous cancel signal and start metrics
            self._cancel_event.clear()
            overall_start_time = time.time()
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            yield TTSStartedFrame()

            loop = asyncio.get_event_loop()

            # Track metrics across all chunks
            first_audio = True
            total_audio_bytes = 0
            total_chunk_count = 0

            # Process each text chunk separately for ultra-low latency
            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Check for cancellation before sending next chunk
                if self._cancel_event.is_set():
                    logger.debug("TTS canceled before processing all chunks")
                    break

                logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: '{chunk_text}'")

                # Send generation command for this chunk
                command = json.dumps({
                    "cmd": "generate",
                    "text": chunk_text,
                    "speed": self._speed,
                    "use_boundaries": False  # Disable for small chunks
                }) + "\n"

                # Non-blocking write for lowest latency
                await loop.run_in_executor(None, self._process.stdin.write, command)
                await loop.run_in_executor(None, self._process.stdin.flush)

                # Stream audio chunks from this text chunk
                chunk_done = False
                while not chunk_done:
                    # Read response line
                    line = await loop.run_in_executor(None, self._process.stdout.readline)
                    if not line:
                        raise RuntimeError("Kokoro worker stopped unexpectedly")

                    payload = json.loads(line.strip())

                    # Check for cancellation
                    if self._cancel_event.is_set():
                        logger.debug("TTS cancel requested; draining current chunk")
                        if payload.get("done"):
                            chunk_done = True
                        continue

                    if "chunk" in payload:
                        audio_bytes = base64.b64decode(payload["chunk"])
                        total_chunk_count += 1
                        total_audio_bytes += len(audio_bytes)

                        # Log TTFB for first audio across all chunks
                        if first_audio:
                            ttfb = (time.time() - overall_start_time) * 1000
                            self._ttfb_ms = ttfb
                            logger.info(f"⚡ Ultra-low latency TTFB: {ttfb:.1f}ms (chunk {chunk_idx + 1}, {len(audio_bytes)} bytes)")
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                            first_audio = False

                        # Stream audio immediately
                        if audio_bytes:
                            yield TTSAudioRawFrame(audio_bytes, self.sample_rate, 1)

                        # Minimal delay for smooth streaming
                        chunk_duration = len(audio_bytes) / (self.sample_rate * 2)
                        delay = min(chunk_duration * 0.02, 0.005)
                        await asyncio.sleep(delay)

                    elif payload.get("done"):
                        # This text chunk is complete
                        chunk_done = True
                        logger.debug(f"Chunk {chunk_idx + 1}/{len(chunks)} complete")

                    elif "error" in payload:
                        raise RuntimeError(f"Worker error on chunk {chunk_idx + 1}: {payload['error']}")

            # Log final metrics
            total_time = (time.time() - overall_start_time) * 1000
            logger.info(f"TTS completed: {len(chunks)} text chunks, {total_chunk_count} audio chunks, "
                       f"{total_audio_bytes} bytes, {total_time:.1f}ms total, TTFB: {self._ttfb_ms:.1f}ms")

        except Exception as exc:
            logger.error(f"Error in ultra-low latency TTS: {exc}")
            yield ErrorFrame(error=str(exc))
        finally:
            await self.stop_processing_metrics()
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _cleanup(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._initialized = False

    async def __aenter__(self):
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
