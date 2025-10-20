#!/usr/bin/env python3
"""
MLX Kokoro TTS sidecar.

Exposes a lightweight HTTP streaming API so the Tauri host can delegate TTS work
to a dedicated process. The sidecar keeps a single MLX Kokoro instance warm and
streams 24 kHz PCM audio in 20 ms frames.

This daemon runs independently and communicates with the main app via HTTP.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import numpy as np
from loguru import logger

# Configure persistent logging for daemon mode
LOG_DIR = Path.home() / "Library" / "Logs" / "LocalCat"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "tts-daemon.log"

logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Console logging
logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="DEBUG")  # File logging

# PID file for daemon management
PID_FILE = Path("/tmp/localcat-tts-daemon.pid")

SERVER_ROOT = Path(__file__).resolve().parents[1]
PIPECAT_SRC = SERVER_ROOT / "pipecat" / "src"
for path in (SERVER_ROOT, PIPECAT_SRC):
    if path.exists():
        sys.path.insert(0, str(path))

from core.tts.kokoro_mlx import MLXKokoroTTSService
from pipecat.frames.frames import TTSAudioRawFrame


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.getenv("LOCALCAT_TTS_PORT", "8770"))
DEFAULT_VOICE = os.getenv("LOCALCAT_TTS_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("LOCALCAT_TTS_SPEED", "1.0"))
DEFAULT_LANG = os.getenv("LOCALCAT_TTS_LANG", "en")
DEFAULT_GEN_SR = 24000  # MLX Kokoro native sample rate
# Default to 24 kHz to match prior Kokoro paths; can override to 48000 via env
DEFAULT_OUT_SR = int(os.getenv("LOCALCAT_TTS_OUT_SAMPLERATE", "24000"))

# Global TTS service - initialized in lifespan
tts_service: MLXKokoroTTSService | None = None
_generation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global tts_service

    # Startup
    logger.info("🚀 Starting MLX Kokoro TTS daemon")
    logger.info(f"📝 Logging to: {LOG_FILE}")
    logger.info(f"📍 PID: {os.getpid()}, Port: {DEFAULT_PORT}, Voice: {DEFAULT_VOICE}")

    # Write PID file for daemon management
    PID_FILE.write_text(str(os.getpid()))
    logger.debug(f"✅ PID file written: {PID_FILE}")

    try:
        tts_service = MLXKokoroTTSService(
            voice=DEFAULT_VOICE,
            speed=DEFAULT_SPEED,
            sample_rate=24000,
        )
        logger.info("🔥 Warming up MLX pipeline with test generation...")
        # Pre-warm the pipeline to avoid first-request JIT compilation issues
        try:
            warmup_result = tts_service._generate_audio_sync("Hello")  # type: ignore[attr-defined]
            if warmup_result:
                logger.info("✅ MLX Kokoro TTS daemon ready (warmed up)")
            else:
                logger.warning("⚠️  Warmup returned no audio, but continuing...")
        except Exception as e:
            logger.error(f"❌ Warmup failed: {e}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS service: {e}")
        raise

    yield  # Server runs here

    # Shutdown
    logger.info("🔻 TTS daemon shutting down")
    if PID_FILE.exists():
        PID_FILE.unlink()
        logger.debug(f"🗑️  Removed PID file: {PID_FILE}")


app = FastAPI(title="LocalCat Kokoro MLX Sidecar", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None
    lang: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {
        "engine": "kokoro_mlx_sidecar",
        "voice": DEFAULT_VOICE,
        "lang": DEFAULT_LANG,
        "sample_rate": 24000,
    }


def _chunk_bytes_for_sr(sr: int) -> int:
    # 20ms chunks, int16 mono
    return int(sr * 0.02) * 2


async def _stream_sentence(text: str, voice: str, speed: float, lang: str) -> AsyncGenerator[bytes, None]:
    if tts_service is None:
        error_msg = "TTS service not initialized - daemon startup may have failed"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.debug(f"[TTS Stream] Starting synthesis: text='{text[:50]}...', voice={voice}, speed={speed}, lang={lang}")
    buffer = bytearray()
    async with _generation_lock:
        try:
            # Adjust runtime voice/speed before generation
            logger.debug(f"[TTS Stream] Setting voice={voice}, speed={speed}, lang={lang}")
            tts_service._voice = voice  # type: ignore[attr-defined]
            tts_service._speed = speed  # type: ignore[attr-defined]
            tts_service._lang = lang  # type: ignore[attr-defined]
            resolver = getattr(tts_service, "_resolve_lang_code", None)
            if callable(resolver):
                try:
                    logger.debug(f"[TTS Stream] Resolving lang code: {lang}")
                    resolver(lang)  # type: ignore[misc]
                except Exception as exc:
                    logger.warning(f"Failed to resolve lang '{lang}': {exc}")

            logger.debug("[TTS Stream] Calling _generate_audio_sync synchronously")
            # Call synchronously to avoid MLX/Metal threading issues with executors
            # CRITICAL: MLX/Metal may crash when generator iteration happens in worker threads
            # This is a dedicated daemon, so blocking the event loop briefly is acceptable
            result = tts_service._generate_audio_sync(text)  # type: ignore[attr-defined]
            logger.debug(f"[TTS Stream] Audio generation complete, result type: {type(result)}")
        except Exception as exc:
            logger.error(f"[TTS Stream] ERROR during audio generation: {exc}", exc_info=True)
            raise

        if not result:
            raise RuntimeError("Sidecar MLX returned no audio data")

        audio_data, sample_rate = result
        if sample_rate and sample_rate != DEFAULT_GEN_SR:
            logger.debug(f"Sidecar MLX reported sample rate -> {sample_rate}")

        import numpy as np

        if hasattr(audio_data, "astype"):
            audio_np = np.array(audio_data)
        else:
            audio_np = np.asarray(audio_data)

        # Flatten to mono
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)

        # Sanitize (replace NaN/inf), DC-remove, normalize, convert to int16
        audio_f = np.nan_to_num(audio_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio_f.size:
            audio_f -= float(audio_f.mean())
        peak = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0
        if peak > 1e-6:
            audio_f = audio_f / peak  # normalize to [-1, 1]
        audio_np = (np.clip(audio_f, -1.0, 1.0) * 32767).astype(np.int16)

        # If still silent, write a debug WAV to /tmp for inspection (one-shot per request)
        try:
            if audio_np.size and int(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))) == 0:
                import soundfile as sf
                debug_wav = "/tmp/localcat_sidecar_debug.wav"
                sf.write(debug_wav, audio_np.astype(np.int16), samplerate=sample_rate)
                logger.debug(f"[PCM DEBUG] Wrote silent audio to {debug_wav}")
        except Exception:
            pass

        # If still silent after sanitize+normalize, try one fallback voice
        try:
            rms_check = int(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))) if audio_np.size else 0
        except Exception:
            rms_check = 0
        if rms_check == 0 and voice != "af_bella":
            logger.debug("[PCM DEBUG] Silent audio detected, retrying with fallback voice 'af_bella'")
            # regenerate using fallback voice
            tts_service._voice = "af_bella"  # type: ignore[attr-defined]
            result_fb = await loop.run_in_executor(
                tts_service._executor,  # type: ignore[attr-defined]
                tts_service._generate_audio_sync,  # type: ignore[attr-defined]
                text,
            )
            if result_fb:
                audio_data_fb, sr_fb = result_fb
                if hasattr(audio_data_fb, "astype"):
                    audio_np = np.array(audio_data_fb).reshape(-1)
                else:
                    audio_np = np.asarray(audio_data_fb).reshape(-1)
                audio_np = np.nan_to_num(audio_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if audio_np.size:
                    audio_np -= float(audio_np.mean())
                peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
                if peak > 1e-6:
                    audio_np = (audio_np / peak * 32767).astype(np.int16)

        # Resample to output sample rate if requested (fast nearest-neighbor upsample)
        out_sr = DEFAULT_OUT_SR
        if out_sr != DEFAULT_GEN_SR:
            if out_sr % DEFAULT_GEN_SR == 0:
                factor = out_sr // DEFAULT_GEN_SR
                audio_np = np.repeat(audio_np, factor)
            else:
                # Fallback: simple linear interpolation
                import numpy as _np
                x = _np.arange(audio_np.shape[0])
                xi = _np.linspace(0, audio_np.shape[0]-1, int(audio_np.shape[0] * (out_sr/DEFAULT_GEN_SR)))
                audio_np = _np.interp(xi, x, audio_np.astype(_np.float32)).astype(_np.int16)

        pcm_bytes = audio_np.tobytes()
        buffer.extend(pcm_bytes)

        chunk_bytes = _chunk_bytes_for_sr(out_sr)
        while len(buffer) >= chunk_bytes:
            chunk = bytes(buffer[:chunk_bytes])
            del buffer[:chunk_bytes]
            yield chunk

        if buffer:
            yield bytes(buffer)


@app.post("/tts")
async def synthesize(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    voice = request.voice or DEFAULT_VOICE
    speed = request.speed or DEFAULT_SPEED
    lang = request.lang or DEFAULT_LANG

    if lang.lower() not in {"en", "a", "b", "j", "z"}:
        logger.warning(f"Unsupported lang '{lang}', falling back to default '{DEFAULT_LANG}'")

    async def audio_stream():
        try:
            async for chunk in _stream_sentence(request.text, voice, speed, lang):
                yield chunk
        except Exception as exc:
            import traceback
            logger.error(f"Sidecar synthesis failed: {exc}")
            logger.error(f"Exception type: {type(exc).__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    headers = {
        "X-Sample-Rate": str(DEFAULT_OUT_SR),
        "X-Voice": voice,
    }
    return StreamingResponse(audio_stream(), media_type="application/octet-stream", headers=headers)


@app.post("/prewarm")
async def prewarm(request: TTSRequest | None = None):
    text = request.text if request and request.text else "Hello from LocalCat."
    async for _ in _stream_sentence(text, DEFAULT_VOICE, DEFAULT_SPEED, DEFAULT_LANG):
        break
    return {"status": "ready"}


@app.post("/shutdown")
async def shutdown():
    """Gracefully shutdown the daemon (called by app on quit)"""
    logger.info("🛑 Shutdown requested via /shutdown endpoint")

    async def delayed_shutdown():
        await asyncio.sleep(0.5)  # Allow response to be sent
        logger.info("👋 Stopping TTS daemon")
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_shutdown())
    return {"status": "shutting_down"}


def _install_signal_handlers(loop: asyncio.AbstractEventLoop):
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_shutdown(loop, s)))


async def _shutdown(loop: asyncio.AbstractEventLoop, sig: signal.Signals):
    logger.info(f"🔻 Shutting down sidecar ({sig.name})")
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def main():
    import uvicorn

    loop = asyncio.get_event_loop()
    _install_signal_handlers(loop)

    uvicorn.run(
        "sidecars.tts_sidecar_mlx:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_level="info",
        reload=False,
        # Stability in bundled macOS app: avoid uvloop/httptools
        loop="asyncio",
        http="h11",
        # lifespan="on" is default, we're using lifespan context manager
    )


if __name__ == "__main__":
    # CRITICAL: Force spawn mode for multiprocessing to avoid fork/Obj-C/Metal conflicts on macOS
    # This is required for MLX to work correctly in app bundles where fork safety is compromised
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()
