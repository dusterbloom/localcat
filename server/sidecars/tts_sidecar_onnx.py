#!/usr/bin/env python3
"""
Kokoro ONNX TTS sidecar daemon.

Uses kokoro-onnx instead of MLX to avoid Metal threading issues.
Exposes a lightweight HTTP streaming API so the Tauri host can delegate TTS work
to a dedicated process. Streams 24 kHz PCM audio in 20 ms frames.
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

# Kokoro ONNX model paths
SERVER_ROOT = Path(__file__).resolve().parents[1]
KOKORO_MODEL_PATH = SERVER_ROOT / "models" / "kokoro" / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = SERVER_ROOT / "models" / "kokoro" / "voices-v1.0.bin"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.getenv("LOCALCAT_TTS_PORT", "8770"))
DEFAULT_VOICE = os.getenv("LOCALCAT_TTS_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("LOCALCAT_TTS_SPEED", "1.0"))
DEFAULT_LANG = os.getenv("LOCALCAT_TTS_LANG", "en")
DEFAULT_OUT_SR = 24000  # Kokoro native sample rate

# Global Kokoro instance
kokoro_tts = None
_generation_lock = asyncio.Lock()
_executor = None  # ThreadPoolExecutor for blocking calls


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global kokoro_tts, _executor
    from concurrent.futures import ThreadPoolExecutor

    # Startup
    logger.info("🚀 Starting Kokoro ONNX TTS daemon")
    logger.info(f"📝 Logging to: {LOG_FILE}")
    logger.info(f"📍 PID: {os.getpid()}, Port: {DEFAULT_PORT}, Voice: {DEFAULT_VOICE}")

    # Write PID file for daemon management
    PID_FILE.write_text(str(os.getpid()))
    logger.debug(f"✅ PID file written: {PID_FILE}")

    # Create executor for blocking ONNX calls (single worker for stability)
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro_worker")
    logger.debug("✅ Created single-threaded executor for ONNX runtime")

    try:
        import kokoro_onnx
        logger.info(f"📦 Loading Kokoro ONNX from: {KOKORO_MODEL_PATH}")
        logger.info(f"🎤 Loading voices from: {KOKORO_VOICES_PATH}")

        # Load Kokoro with default configuration
        # The ORT threading is controlled by environment variables set in main()
        kokoro_tts = kokoro_onnx.Kokoro(
            model_path=str(KOKORO_MODEL_PATH),
            voices_path=str(KOKORO_VOICES_PATH)
        )
        logger.info("✅ Kokoro ONNX TTS daemon ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS service: {e}", exc_info=True)
        raise

    yield  # Server runs here

    # Shutdown
    logger.info("🔻 TTS daemon shutting down")
    if _executor:
        _executor.shutdown(wait=True)
        logger.debug("✅ Shut down thread pool executor")
    if PID_FILE.exists():
        PID_FILE.unlink()
        logger.debug(f"🗑️  Removed PID file: {PID_FILE}")


app = FastAPI(title="LocalCat Kokoro ONNX Sidecar", lifespan=lifespan)
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
        "engine": "kokoro_onnx_sidecar",
        "voice": DEFAULT_VOICE,
        "lang": DEFAULT_LANG,
        "sample_rate": DEFAULT_OUT_SR,
    }


def _chunk_bytes_for_sr(sr: int) -> int:
    # 20ms chunks, int16 mono
    return int(sr * 0.02) * 2


async def _stream_sentence(text: str, voice: str, speed: float, lang: str) -> AsyncGenerator[bytes, None]:
    if kokoro_tts is None:
        error_msg = "TTS service not initialized - daemon startup may have failed"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.debug(f"[TTS Stream] Starting synthesis: text='{text[:50]}...', voice={voice}, speed={speed}, lang={lang}")

    # CRITICAL: Generate ALL audio first, then stream it
    # Kokoro has known streaming issues - must generate complete audio before streaming bytes
    # See: https://github.com/remsky/Kokoro-FastAPI/issues/57

    try:
        async with _generation_lock:
            logger.debug("[TTS Stream] Calling kokoro_tts.create (full generation)")

            # Generate complete audio in executor (kokoro-onnx is synchronous/blocking)
            # This must complete fully before we start streaming
            loop = asyncio.get_running_loop()  # FIXED: use get_running_loop() in async context
            samples, sample_rate = await loop.run_in_executor(
                _executor,
                lambda: kokoro_tts.create(text, voice=voice, speed=speed, lang=lang)
            )

            logger.debug(f"[TTS Stream] Audio generation complete: {len(samples) if hasattr(samples, '__len__') else 'unknown'} samples, sr={sample_rate}")

        # NOW that generation is complete, convert and stream
        # Convert to int16 PCM
        audio_np = np.asarray(samples, dtype=np.float32)

        # Flatten to mono if needed
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)

        # Sanitize (replace NaN/inf), DC-remove, normalize
        audio_f = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
        if audio_f.size:
            audio_f -= float(audio_f.mean())
        peak = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0
        if peak > 1e-6:
            audio_f = audio_f / peak

        # Convert to int16
        audio_int16 = (np.clip(audio_f, -1.0, 1.0) * 32767).astype(np.int16)

        logger.debug(f"[TTS Stream] Converted to int16, size={audio_int16.size} samples, streaming in chunks...")

        # Stream in 20ms chunks
        pcm_bytes = audio_int16.tobytes()
        chunk_size = _chunk_bytes_for_sr(sample_rate)

        offset = 0
        chunk_count = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + chunk_size]
            offset += chunk_size
            chunk_count += 1
            yield chunk

        logger.debug(f"[TTS Stream] Streaming complete: {chunk_count} chunks sent")

    except Exception as exc:
        import traceback
        logger.error(f"[TTS Stream] ERROR during audio generation: {exc}", exc_info=True)
        logger.error(f"Exception type: {type(exc).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


@app.post("/tts")
async def synthesize(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    voice = request.voice or DEFAULT_VOICE
    speed = request.speed or DEFAULT_SPEED
    lang = request.lang or DEFAULT_LANG

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

    # FIXED: Create new event loop instead of deprecated get_event_loop()
    # This avoids "There is no current event loop" deprecation warning
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)

    # Set ORT threading environment variables before uvicorn starts
    # These must be set BEFORE onnxruntime is imported
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # OpenMP single thread
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")  # Only errors

    uvicorn.run(
        app,  # Pass app object directly instead of string
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        log_level="info",
        reload=False,
        loop="asyncio",
        http="h11",
    )


if __name__ == "__main__":
    main()
