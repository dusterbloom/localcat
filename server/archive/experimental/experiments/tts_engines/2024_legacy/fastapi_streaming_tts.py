#!/usr/bin/env python3
"""
FastAPI Streaming TTS Service for Pipecat
Uses HTTP connection pooling to FastAPI server for ultra-low latency
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

import httpx
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


class FastAPIStreamingTTS(TTSService):
    """
    FastAPI Streaming TTS Service with HTTP connection pooling
    Provides real streaming TTS via FastAPI server with Unix socket
    """

    def __init__(
        self,
        *,
        voice: str = "af_bella",
        speed: float = 1.0,
        sample_rate: int = 24000,
        socket_path: str = "/tmp/fastapi-tts.sock",
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,
            **kwargs
        )

        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate
        self._socket_path = socket_path

        # HTTP client with connection pooling optimized for local Unix socket
        self._client: Optional[httpx.AsyncClient] = None
        self._setup_http_client()

        logger.debug(f"✅ FastAPI Streaming TTS initialized with voice: {self._voice}")

    def _setup_http_client(self):
        """Setup HTTP client with connection pooling for Unix socket"""
        transport = httpx.AsyncHTTPTransport(
            uds=self._socket_path,
            limits=httpx.Limits(
                max_keepalive_connections=5,  # Pool size
                max_connections=10,           # Max concurrent
                keepalive_expiry=300.0        # 5min expiry
            )
        )

        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(5.0, connect=0.1)  # Fast local timeouts
        )

    async def _ensure_client(self):
        """Ensure HTTP client is available"""
        if self._client is None:
            self._setup_http_client()

    async def _synthesize_audio(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize audio via FastAPI server"""
        await self._ensure_client()

        start_time = time.time()
        assert self._client is not None
        response = await self._client.post(
            "http://localhost/synthesize",
            json={
                "text": text,
                "voice": self._voice,
                "speed": self._speed
            }
        )

        http_time = time.time() - start_time
        # logger.debug(f"HTTP request completed in {http_time:.3f}s")

        if response.status_code != 200:
            raise RuntimeError(f"FastAPI TTS failed: {response.status_code} - {response.text}")

        data = response.json()
        import base64
        audio_bytes = base64.b64decode(data["audio"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        generation_time = time.time() - start_time
        chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
        # logger.debug(f"FastAPI TTS: {len(text)} chars in {generation_time:.3f}s ({chars_per_sec:.1f} chars/s)")

        return audio_array, data["sample_rate"]

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from complete text without chunking - true streaming"""

        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        # logger.debug(f"🌐 FastAPI Streaming full text: {text[:50]}{'...' if len(text) > 50 else ''}")

        start_time = time.time()

        yield TTSStartedFrame()

        try:
            # Generate audio for the complete text via FastAPI (no chunking)
            audio_data, actual_sample_rate = await self._synthesize_audio(text)

            if audio_data is not None and len(audio_data) > 0:
                ttfb = (time.time() - start_time) * 1000
                # logger.debug(f"🚀 FASTAPI TTFB: {ttfb:.1f}ms for {len(text)} chars")

                frame = TTSAudioRawFrame(
                    audio=audio_data.tobytes(),
                    sample_rate=actual_sample_rate,
                    num_channels=1
                )

                yield frame
            else:
                logger.warning(f"Empty audio data for text: '{text}'")

        except asyncio.CancelledError:
            logger.debug("FastAPI TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"FastAPI Streaming TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()