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
        buffer_ms: int = 50,  # Target buffer size in milliseconds
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._device = device
        self._speed = speed
        self._use_boundaries = use_boundaries
        self._buffer_ms = buffer_ms

        # Pre‑roll configuration (accelerate first audio without harming onboarding)
        import os as _os
        self._preroll_enabled = _os.getenv("KOKORO_PREROLL_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        try:
            self._preroll_chars = int(_os.getenv("KOKORO_PREROLL_CHARS", "25"))
        except Exception:
            self._preroll_chars = 25
        try:
            self._preroll_min_total_chars = int(_os.getenv("KOKORO_PREROLL_MIN_TOTAL_CHARS", "60"))
        except Exception:
            self._preroll_min_total_chars = 60

        self._process: Optional[subprocess.Popen[str]] = None
        self._initialized = False
        self._worker_script = self._get_worker_script_path()

        # Metrics tracking
        self._ttfb_ms = None
        self._total_chunks = 0

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
            # Set environment variables for worker configuration
            env = os.environ.copy()
            # Keep legacy KOKORO_* for backward compat, and set TTS_* for worker
            env["KOKORO_MIN_TOKENS"] = env.get("KOKORO_MIN_TOKENS", "175")
            env["KOKORO_MAX_TOKENS"] = env.get("KOKORO_MAX_TOKENS", "250")
            env["KOKORO_BUFFER_MS"] = env.get("KOKORO_BUFFER_MS", str(self._buffer_ms))

            # Align with worker expectations
            env["TTS_MIN_TOKENS"] = env.get("TTS_MIN_TOKENS", env["KOKORO_MIN_TOKENS"])  # 175
            env["TTS_MAX_TOKENS"] = env.get("TTS_MAX_TOKENS", env["KOKORO_MAX_TOKENS"])  # 250
            env["TTS_BUFFER_MS"] = env.get("TTS_BUFFER_MS", env["KOKORO_BUFFER_MS"])    # 50

            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for lowest latency
                env=env,
            )
            logger.info(f"Started ultra-low latency Kokoro worker (pid={self._process.pid}, buffer={self._buffer_ms}ms)")
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
        """Generate speech with ultra‑low latency streaming.

        Implements a conversation‑only pre‑roll: a tiny first chunk to guarantee
        fast time‑to‑first‑byte, then normal sentence‑level streaming.
        """
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

            # Start metrics once for the whole utterance
            start_time = time.time()
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            yield TTSStartedFrame()

            loop = asyncio.get_event_loop()

            async def _stream_once(payload_text: str, use_boundaries: bool,
                                   first_chunk_flag: bool) -> bool:
                nonlocal ttfb_stopped
                # Send generation command
                command = json.dumps({
                    "cmd": "generate",
                    "text": payload_text,
                    "speed": self._speed,
                    "use_boundaries": use_boundaries
                }) + "\n"

                await loop.run_in_executor(None, self._process.stdin.write, command)
                await loop.run_in_executor(None, self._process.stdin.flush)

                # Stream audio
                while True:
                    line = await loop.run_in_executor(None, self._process.stdout.readline)
                    if not line:
                        raise RuntimeError("Kokoro worker stopped unexpectedly")

                    payload = json.loads(line.strip())

                    if "chunk" in payload:
                        audio_bytes = base64.b64decode(payload["chunk"]) if isinstance(payload["chunk"], str) else payload["chunk"]

                        if first_chunk_flag:
                            ttfb = (time.time() - start_time) * 1000
                            self._ttfb_ms = ttfb
                            logger.info(f"Ultra-low latency TTFB: {ttfb:.1f}ms (chunk {payload.get('bytes')} bytes)")
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                            first_chunk_flag = False

                        if audio_bytes:
                            yield TTSAudioRawFrame(audio_bytes, self.sample_rate, 1)

                        # Gentle pacing for smooth playback
                        chunk_duration = len(audio_bytes) / (self.sample_rate * 2)  # seconds
                        await asyncio.sleep(min(chunk_duration * 0.05, 0.01))

                    elif payload.get("boundary") == "sentence":
                        logger.debug("Sentence boundary detected")
                    elif payload.get("done"):
                        # Done with this sub‑utterance
                        break
                    elif "error" in payload:
                        raise RuntimeError(payload["error"])
                return first_chunk_flag

            # Decide whether to use pre‑roll (conversation only)
            use_preroll = (
                self._use_boundaries and self._preroll_enabled and len(cleaned) >= self._preroll_min_total_chars
            )

            if use_preroll:
                # Compute a tiny first chunk (~25 chars) for instant TTFB
                chunks = chunk_for_kokoro_ultra_low_latency(cleaned, max_chars=self._preroll_chars)
                if chunks:
                    pre_text = chunks[0].strip()
                    remainder = cleaned
                    if cleaned.startswith(pre_text):
                        remainder = cleaned[len(pre_text):].lstrip()
                    else:
                        # Fallback: find first occurrence; if not found, skip preroll gracefully
                        idx = cleaned.find(pre_text)
                        if idx != -1:
                            remainder = cleaned[idx + len(pre_text):].lstrip()
                        else:
                            pre_text = ""

                    if pre_text:
                        # 1) Pre‑roll without boundaries for fastest first audio
                        first_chunk = True
                        async for frame in _stream_once(pre_text, use_boundaries=False, first_chunk_flag=first_chunk):
                            # Unused: generator returns TTSAudioRawFrame only
                            pass
                        # 2) Stream the remainder with sentence boundaries
                        if remainder:
                            first_chunk = False  # already emitted first chunk
                            async for frame in _stream_once(remainder, use_boundaries=True, first_chunk_flag=first_chunk):
                                pass
                        # Finish
                        return

            # Fallback: single‑pass streaming
            first_chunk = True
            async for frame in _stream_once(cleaned, use_boundaries=self._use_boundaries, first_chunk_flag=first_chunk):
                pass

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
