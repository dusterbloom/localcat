"""
HTTP sidecar-backed Kokoro TTS service.

Delegates audio generation to an external service (MLX on macOS or sherpa-rs on
other platforms) via a streaming HTTP interface.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

import httpx
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


CHUNK_SAMPLES = 480  # 20ms @ 24kHz
FRAME_BYTES = CHUNK_SAMPLES * 2  # mono int16


class SidecarKokoroTTSService(TTSService):
    """TTS service that streams audio from an HTTP sidecar."""

    def __init__(
        self,
        *,
        base_url: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000,
        lang: str = "en",
        chunk_size_chars: int = 120,
        max_concurrency: int = 1,
        request_timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,
            **kwargs,
        )

        self._voice = voice
        self._speed = speed
        self._lang = lang
        self._sample_rate = sample_rate
        self._chunk_size_chars = max(40, chunk_size_chars)
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(request_timeout, connect=5.0) if request_timeout else httpx.Timeout(None),
        )
        self._concurrency = asyncio.Semaphore(max(1, max_concurrency))
        self._request_lock = asyncio.Lock()
        logger.debug(f"✅ Kokoro sidecar TTS configured @ {self._base_url}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    async def _stream_from_sidecar(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio bytes for the provided text from the sidecar."""
        payload = {
            "text": text,
            "voice": self._voice,
            "speed": self._speed,
            "lang": self._lang,
        }

        buffer = bytearray()
        async with self._request_lock:
            async with self._client.stream("POST", "/tts", json=payload) as response:
                if response.status_code != 200:
                    detail = await response.aread()
                    raise RuntimeError(
                        f"Sidecar error ({response.status_code}): {detail.decode(errors='ignore')}"
                    )

                sample_rate_header = response.headers.get("X-Sample-Rate")
                if sample_rate_header:
                    try:
                        self._sample_rate = int(sample_rate_header)
                    except ValueError:
                        logger.debug(f"Invalid X-Sample-Rate header: {sample_rate_header}")

                async for data in response.aiter_bytes():
                    if not data:
                        continue
                    buffer.extend(data)
                    while len(buffer) >= FRAME_BYTES:
                        chunk = buffer[:FRAME_BYTES]
                        del buffer[:FRAME_BYTES]
                        yield bytes(chunk)

        if buffer:
            yield bytes(buffer)

    def _split_sentences(self, text: str):
        max_length = max(80, self._chunk_size_chars)
        return split_text_for_kokoro_streaming(
            text,
            min_length=50,
            max_length=max_length,
        )

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech by delegating to the sidecar."""
        if not text.strip():
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        sentences = self._split_sentences(text)
        if not sentences:
            logger.debug(f"🔇 Skipping TTS for empty text: '{text}'")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        yield TTSStartedFrame()

        first_audio_sent = False
        overall_start_time = time.time()

        try:
            async with self._concurrency:
                for idx, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    chunk_start = time.time()
                    pcm_logged = False
                    try:
                        async for audio_bytes in self._stream_from_sidecar(sentence):
                            if not audio_bytes:
                                continue

                            if not first_audio_sent:
                                ttfb_ms = (time.time() - overall_start_time) * 1000
                                logger.debug(f"🚀 Sidecar Kokoro TTFB: {ttfb_ms:.1f}ms")
                                await self.stop_ttfb_metrics()
                                first_audio_sent = True

                            chunk_latency = (time.time() - chunk_start) * 1000
                            logger.debug(
                                f"✨ Sidecar chunk {idx + 1}/{len(sentences)}: "
                                f"{len(sentence)} chars → {chunk_latency:.1f}ms"
                            )

                            # First-chunk PCM stats for quick debugging
                            if not pcm_logged:
                                try:
                                    import numpy as _np
                                    arr = _np.frombuffer(audio_bytes, dtype=_np.int16)
                                    # Avoid excessive logs: inspect first 2000 samples
                                    sl = arr[:2000]
                                    if sl.size:
                                        rms = float(_np.sqrt(_np.mean(sl.astype(_np.float32) ** 2)))
                                        pcm_min = int(sl.min())
                                        pcm_max = int(sl.max())
                                        logger.debug(
                                            f"[PCM DEBUG] sr={self._sample_rate}Hz, bytes={len(audio_bytes)}, "
                                            f"rms={rms:.1f}, min={pcm_min}, max={pcm_max}"
                                        )
                                        pcm_logged = True
                                except Exception as _e:
                                    logger.debug(f"[PCM DEBUG] failed to parse chunk: {_e}")

                            yield TTSAudioRawFrame(
                                audio=audio_bytes,
                                sample_rate=self._sample_rate,
                                num_channels=1,
                            )
                    except Exception as exc:
                        logger.error(f"Kokoro sidecar request failed: {exc}")
                        raise

        except asyncio.CancelledError:
            logger.debug("Sidecar TTS generation cancelled")
            raise
        except Exception as exc:
            logger.error(f"Kokoro sidecar TTS error: {exc}")
            yield ErrorFrame(error=str(exc))
        finally:
            await self.stop_processing_metrics()
            if not first_audio_sent:
                await self.stop_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()
