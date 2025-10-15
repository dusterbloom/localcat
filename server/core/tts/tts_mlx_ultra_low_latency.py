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
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from tools.text_formatter import sanitize_for_voice


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

    async def request_cancel(self) -> None:
        """Signal the TTS stream to stop as soon as possible (barge-in)."""
        try:
            self._cancel_event.set()
        except Exception:
            pass

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
            # Optimized environment variables for consistent latency
            env = os.environ.copy()
            env["KOKORO_MIN_TOKENS"] = "150"  # Reduced for faster first chunk
            env["KOKORO_MAX_TOKENS"] = "200"  # Reduced for consistency
            env["KOKORO_BUFFER_MS"] = str(self._buffer_ms)
            env["TTS_PREWARM"] = "true"  # Enable enhanced prewarming
            env["MLX_GPU_ALLOCATOR"] = "lazy"  # Optimized memory allocation

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
        logger.debug(f"TTS input: {cleaned[:100]}...")

        if not cleaned:
            logger.debug("Skipping TTS for empty text")
            return

        try:
            # Initialize variables that may be accessed in finally block
            ttfb_stopped = False

            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro worker")

            # Clear previous cancel signal and start metrics
            self._cancel_event.clear()
            start_time = time.time()
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            yield TTSStartedFrame()

            # Send generation command
            loop = asyncio.get_event_loop()
            command = json.dumps({
                "cmd": "generate",
                "text": cleaned,
                "speed": self._speed,
                "use_boundaries": self._use_boundaries
            }) + "\n"

            # Non-blocking write for lowest latency
            await loop.run_in_executor(None, self._process.stdin.write, command)
            await loop.run_in_executor(None, self._process.stdin.flush)

            # Stream audio chunks with minimal delay
            first_chunk = True
            chunk_count = 0
            total_audio_bytes = 0

            while True:
                # Read response line
                line = await loop.run_in_executor(None, self._process.stdout.readline)
                if not line:
                    raise RuntimeError("Kokoro worker stopped unexpectedly")

                payload = json.loads(line.strip())

                # Check for cancellation before handling payload
                if self._cancel_event.is_set():
                    logger.debug("TTS cancel requested; stopping stream mid-generation")
                    # Drain until done without yielding audio
                    if payload.get("done"):
                        metrics = {
                            "chunks": payload.get("chunks", 0),
                            "ttfb_ms": payload.get("ttfb_ms", self._ttfb_ms),
                            "total_ms": payload.get("total_ms"),
                            "audio_bytes": 0
                        }
                        logger.info(f"TTS completed (canceled): {metrics}")
                        break
                    else:
                        # Skip any audio chunks after cancellation
                        continue

                if "chunk" in payload:
                    audio_bytes = base64.b64decode(payload["chunk"])
                    chunk_count += 1
                    total_audio_bytes += len(audio_bytes)

                    # Log metrics for first chunk
                    if first_chunk:
                        ttfb = (time.time() - start_time) * 1000
                        self._ttfb_ms = ttfb
                        logger.info(f"Ultra-low latency TTFB: {ttfb:.1f}ms (chunk {payload.get('bytes')} bytes)")
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True
                        first_chunk = False

                    # Stream audio immediately without sub-chunking
                    if audio_bytes:
                        yield TTSAudioRawFrame(audio_bytes, self.sample_rate, 1)

                    # Optimized delay for consistent streaming performance
                    chunk_duration = len(audio_bytes) / (self.sample_rate * 2)  # seconds
                    # Reduced delay for lower latency, but maintain stability
                    delay = min(chunk_duration * 0.02, 0.005)  # 2% of duration or 5ms max
                    
                    # Adaptive delay based on performance metrics
                    if hasattr(self, '_ttfb_ms') and self._ttfb_ms:
                        if self._ttfb_ms > 800:  # High latency case
                            delay = max(delay * 0.5, 0.001)  # Reduce delay further
                        elif self._ttfb_ms < 400:  # Good performance case
                            delay = min(delay * 1.2, 0.008)  # Slightly more delay for stability
                    
                    await asyncio.sleep(delay)

                elif payload.get("boundary") == "sentence":
                    # Handle sentence boundaries if needed
                    logger.debug("Sentence boundary detected")

                elif payload.get("done"):
                    metrics = {
                        "chunks": payload.get("chunks", chunk_count),
                        "ttfb_ms": payload.get("ttfb_ms", self._ttfb_ms),
                        "total_ms": payload.get("total_ms"),
                        "audio_bytes": total_audio_bytes
                    }
                    if total_audio_bytes == 0:
                        logger.warning(f"TTS completed with zero audio; skipping playback: {metrics}")
                    else:
                        logger.info(f"TTS completed: {metrics}")
                    break

                elif "error" in payload:
                    raise RuntimeError(payload["error"])

                else:
                    logger.debug(f"Unknown payload: {payload}")

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
