"""
Qwen3 TTS Service - High-quality emotional TTS with voice cloning support.

Uses process isolation via subprocess worker to avoid Metal threading conflicts
on Apple Silicon and provide memory isolation for the 1.7B parameter model.

Features:
- Emotional control via natural language instructions
- Voice cloning from 3-second reference audio
- 10 language support
- 9 built-in speaker voices
- ~97ms TTFB with streaming

Usage:
    tts = Qwen3TTSService(
        voice="Ryan",  # Built-in speaker
        model_type="custom_voice",  # or "base" for cloning, "voice_design"
    )

    # With emotional control
    async for frame in tts.run_tts("Hello!", instruct="Speak excitedly"):
        yield frame

    # With voice cloning
    tts.set_voice_clone(ref_audio="path/to/audio.wav", ref_text="transcript")
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

from tools.text_formatter import sanitize_for_voice


# Qwen3 built-in speakers with their characteristics
QWEN3_SPEAKERS = {
    # English speakers
    "Ryan": {"lang": "English", "desc": "Dynamic male, strong rhythmic drive"},
    "Aiden": {"lang": "English", "desc": "Sunny American male, clear midrange"},
    # Chinese speakers
    "Vivian": {"lang": "Chinese", "desc": "Bright, slightly edgy young female"},
    "Serena": {"lang": "Chinese", "desc": "Warm, gentle young female"},
    "Uncle_Fu": {"lang": "Chinese", "desc": "Seasoned male, low mellow timbre"},
    "Dylan": {"lang": "Chinese", "desc": "Youthful Beijing male, clear natural"},
    "Eric": {"lang": "Chinese", "desc": "Lively Chengdu male, slightly husky"},
    # Japanese speaker
    "Ono_Anna": {"lang": "Japanese", "desc": "Playful Japanese female, light nimble"},
    # Korean speaker
    "Sohee": {"lang": "Korean", "desc": "Warm Korean female, rich emotion"},
}

# Language code mapping
QWEN3_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


class Qwen3TTSService(TTSService):
    """
    Qwen3 TTS service with emotional control and voice cloning.

    Uses subprocess isolation for the 1.7B parameter model to avoid
    Metal threading conflicts on Apple Silicon.
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        voice: str = "Ryan",
        model_type: str = "custom_voice",  # custom_voice, base, voice_design
        language: str = "English",
        sample_rate: int = 24000,
        instruct: Optional[str] = None,  # Default emotional instruction
        # Voice cloning settings
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._model_type = model_type
        self._language = language
        self._default_instruct = instruct
        self._sample_rate = sample_rate

        # Voice cloning configuration
        self._ref_audio = ref_audio
        self._ref_text = ref_text

        # Subprocess management
        self._process: Optional[subprocess.Popen] = None
        self._initialized = False
        self._worker_script = self._get_worker_script_path()

        # Metrics
        self._ttfb_ms = None

        # Barge-in support
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._interrupted: bool = False

        # Log configuration
        speaker_info = QWEN3_SPEAKERS.get(voice, {"desc": "Custom voice"})
        logger.info(f"Qwen3 TTS configured: voice={voice} ({speaker_info['desc']}), "
                   f"model_type={model_type}, language={language}")

    @property
    def voice(self) -> str:
        """Current voice/speaker."""
        return self._voice

    @voice.setter
    def voice(self, value: str):
        """Set voice (requires re-initialization for some changes)."""
        self._voice = value

    @property
    def instruct(self) -> Optional[str]:
        """Default emotional instruction."""
        return self._default_instruct

    @instruct.setter
    def instruct(self, value: Optional[str]):
        """Set default emotional instruction."""
        self._default_instruct = value

    def set_voice_clone(self, ref_audio: str, ref_text: str):
        """
        Configure voice cloning from reference audio.

        Args:
            ref_audio: Path to reference audio file, URL, or base64 string
            ref_text: Transcript of the reference audio
        """
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._model_type = "base"  # Base model is used for voice cloning
        logger.info(f"Voice cloning configured with ref_audio: {ref_audio[:50]}...")

    async def request_cancel(self) -> None:
        """Signal TTS to stop (barge-in support)."""
        try:
            self._cancel_event.set()
        except Exception:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with interruption handling."""
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("Qwen3 TTS: User started speaking - entering interrupted state")
            self._interrupted = True
            await self.request_cancel()
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("Qwen3 TTS: User stopped speaking - exiting interrupted state")
            self._interrupted = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            logger.debug("Qwen3 TTS: Received InterruptionFrame")
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            if self._interrupted:
                logger.debug(f"Qwen3 TTS: Dropping TextFrame during interruption")
                return
            await super().process_frame(frame, direction)

        else:
            await super().process_frame(frame, direction)

    async def _handle_interruption(self, frame: Frame, direction: FrameDirection):
        """Handle interruption by clearing state."""
        self._processing_text = False
        if hasattr(self, '_text_aggregator'):
            await self._text_aggregator.handle_interruption()

    def can_generate_metrics(self) -> bool:
        """This service supports TTFB and processing metrics."""
        return True

    def _get_worker_script_path(self) -> str:
        """Get path to the Qwen3 worker script."""
        return str(Path(__file__).parent / "qwen3_worker.py")

    def _start_worker(self) -> bool:
        """Start the Qwen3 worker subprocess."""
        try:
            env = os.environ.copy()
            # Ensure lazy GPU allocation for MPS
            env["MLX_GPU_ALLOCATOR"] = "lazy"

            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )
            logger.info(f"Started Qwen3 worker (pid={self._process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start Qwen3 worker: {e}")
            return False

    def _ensure_worker(self) -> bool:
        """Ensure worker is running."""
        if self._process and self._process.poll() is None:
            return True
        return self._start_worker()

    async def _initialize_if_needed(self) -> bool:
        """Initialize the worker subprocess if not already done."""
        if self._initialized:
            return True
        if not self._ensure_worker():
            return False

        loop = asyncio.get_event_loop()
        command = json.dumps({
            "cmd": "init",
            "model": self._model_name,
            "voice": self._voice,
            "model_type": self._model_type,
        }) + "\n"

        await loop.run_in_executor(None, self._process.stdin.write, command)
        await loop.run_in_executor(None, self._process.stdin.flush)

        line = await loop.run_in_executor(None, self._process.stdout.readline)
        if not line:
            logger.error("Qwen3 worker did not respond to init")
            return False

        resp = json.loads(line.strip())
        if resp.get("success"):
            config = resp.get("config", {})
            self._sample_rate = config.get("sample_rate", self._sample_rate)
            logger.info(f"Qwen3 TTS initialized: {config}")
            self._initialized = True
            return True

        logger.error(f"Qwen3 init failed: {resp.get('error')}")
        return False

    @traced_tts
    async def run_tts(self, text: str, instruct: Optional[str] = None) -> AsyncGenerator[Frame, None]:
        """
        Generate speech from text with optional emotional control.

        Args:
            text: Text to synthesize
            instruct: Emotional/prosody instruction (e.g., "Speak angrily", "Warm and friendly")
                     Falls back to default_instruct if not provided.
        """
        cleaned = sanitize_for_voice(text)

        if not cleaned.strip():
            logger.debug("Skipping TTS for empty text")
            return

        # Use provided instruct or fall back to default
        effective_instruct = instruct or self._default_instruct

        logger.debug(f"Qwen3 TTS: '{cleaned[:80]}...' instruct={effective_instruct}")

        try:
            ttfb_stopped = False

            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Qwen3 worker")

            self._cancel_event.clear()
            start_time = time.time()
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            yield TTSStartedFrame()

            loop = asyncio.get_event_loop()

            # Determine command type based on voice cloning config
            if self._ref_audio and self._ref_text and self._model_type == "base":
                command = json.dumps({
                    "cmd": "clone",
                    "text": cleaned,
                    "ref_audio": self._ref_audio,
                    "ref_text": self._ref_text,
                    "language": self._language,
                }) + "\n"
            else:
                command = json.dumps({
                    "cmd": "generate",
                    "text": cleaned,
                    "language": self._language,
                    "instruct": effective_instruct,
                }) + "\n"

            await loop.run_in_executor(None, self._process.stdin.write, command)
            await loop.run_in_executor(None, self._process.stdin.flush)

            first_audio = True
            total_bytes = 0
            chunk_count = 0

            while True:
                line = await loop.run_in_executor(None, self._process.stdout.readline)
                if not line:
                    raise RuntimeError("Qwen3 worker stopped unexpectedly")

                payload = json.loads(line.strip())

                if self._cancel_event.is_set():
                    if payload.get("done"):
                        break
                    continue

                if "chunk" in payload:
                    audio_bytes = base64.b64decode(payload["chunk"])
                    chunk_count += 1
                    total_bytes += len(audio_bytes)

                    if first_audio:
                        ttfb = (time.time() - start_time) * 1000
                        self._ttfb_ms = ttfb
                        logger.info(f"⚡ Qwen3 TTS TTFB: {ttfb:.1f}ms ({len(audio_bytes)} bytes)")
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True
                        first_audio = False

                    if audio_bytes:
                        yield TTSAudioRawFrame(audio_bytes, self._sample_rate, 1)

                    # Small delay for smooth streaming
                    await asyncio.sleep(0.005)

                elif payload.get("done"):
                    total_ms = payload.get("total_ms", 0)
                    worker_sr = payload.get("sample_rate", self._sample_rate)
                    if worker_sr != self._sample_rate:
                        self._sample_rate = worker_sr
                        logger.debug(f"Updated sample rate to {worker_sr}Hz")
                    logger.info(f"Qwen3 TTS completed: {chunk_count} chunks, "
                               f"{total_bytes} bytes, {total_ms:.1f}ms")
                    break

                elif "error" in payload:
                    raise RuntimeError(f"Qwen3 worker error: {payload['error']}")

        except Exception as e:
            logger.error(f"Qwen3 TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_processing_metrics()
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _cleanup(self):
        """Cleanup worker subprocess."""
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
        """Async context manager entry."""
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._cleanup()
