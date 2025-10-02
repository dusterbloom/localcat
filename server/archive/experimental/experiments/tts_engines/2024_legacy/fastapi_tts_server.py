#!/usr/bin/env python3
"""
FastAPI TTS Server with Real Streaming Support
Provides Kokoro TTS via HTTP with connection pooling optimization
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from kokoro_onnx import Kokoro
from loguru import logger
from pydantic import BaseModel
import uvicorn


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella"
    speed: float = 1.0
    stream: bool = True


class FastAPITTSService:
    """FastAPI TTS Service with Kokoro ONNX"""

    def __init__(self):
        self.pipeline: Optional[Kokoro] = None
        self._initialize_pipeline()

    def _ensure_models_downloaded(self):
        """Ensure the correct model and voices files are available"""
        cache_dir = Path.home() / ".cache" / "kokoro"
        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"

        # Check if files exist (follow symlinks)
        try:
            model_exists = model_path.exists() and model_path.resolve().stat().st_size > 300_000_000  # >300MB
        except:
            model_exists = False

        try:
            voices_exists = voices_path.exists() and voices_path.resolve().stat().st_size > 25_000_000  # >25MB
        except:
            voices_exists = False

        if model_exists and voices_exists:
            return str(model_path.resolve()), str(voices_path.resolve())

        # Files are missing - download them
        logger.debug("📥 Downloading Kokoro ONNX model files...")

        import urllib.request

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model file
        if not model_exists:
            logger.debug("Downloading kokoro-v1.0.onnx...")
            # Remove any existing symlink first
            if model_path.is_symlink():
                model_path.unlink()
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                model_path
            )

        # Download voices file
        if not voices_exists:
            logger.debug("Downloading voices-v1.0.bin...")
            # Remove any existing symlink first
            if voices_path.is_symlink():
                voices_path.unlink()
            urllib.request.urlretrieve(
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                voices_path
            )

        logger.debug("✅ Kokoro ONNX model files downloaded")
        return str(model_path.resolve()), str(voices_path.resolve())

    def _initialize_pipeline(self):
        """Initialize the Kokoro ONNX pipeline"""
        try:
            logger.debug("🚀 Initializing FastAPI Kokoro ONNX TTS")

            # Ensure models are available
            model_path, voices_path = self._ensure_models_downloaded()

            logger.debug(f"Using model: {model_path}")
            logger.debug(f"Using voices: {voices_path}")

            # Initialize Kokoro with the correct files
            self.pipeline = Kokoro(
                model_path=model_path,
                voices_path=voices_path,
                espeak_config=None  # Use default
            )

            logger.debug("✅ Kokoro ONNX pipeline loaded successfully")

            # Test the voice
            try:
                test_audio, test_sr = self.pipeline.create("Hello", voice="af_bella", speed=1.0)
                logger.debug(f"✅ Voice af_bella verified - generated {len(test_audio)} samples at {test_sr}Hz")
            except Exception as voice_error:
                logger.error(f"❌ Voice test failed: {voice_error}")
                raise

            logger.info("FastAPI Kokoro ONNX TTS ready")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro ONNX: {e}")
            self.pipeline = None
            raise

    def generate_audio(self, text: str, voice: str = "af_bella", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Generate audio synchronously"""
        if not self.pipeline:
            raise RuntimeError("Kokoro pipeline not initialized")

        start_time = time.time()
        audio_data, sample_rate = self.pipeline.create(
            text=text,
            voice=voice,
            speed=speed
        )
        generation_time = time.time() - start_time

        chars_per_sec = len(text) / generation_time if generation_time > 0 else 0
        # logger.debug(f"Generated {len(text)} chars in {generation_time:.3f}s ({chars_per_sec:.1f} chars/s)")

        return audio_data, sample_rate


# Global TTS service instance
tts_service = FastAPITTSService()

# FastAPI app
app = FastAPI(title="FastAPI TTS Server", version="1.0.0")


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """Synthesize text to speech and return audio data"""
    try:
        audio_data, sample_rate = tts_service.generate_audio(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )

        # Convert to int16 for compatibility
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        import base64
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')

        return {
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "duration": len(audio_int16) / sample_rate,
            "format": "base64"
        }

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/stream")
async def synthesize_stream(request: TTSRequest):
    """Stream audio data as it's generated (real-time streaming)"""
    try:
        # For now, generate all at once and stream chunks
        # TODO: Implement true real-time streaming as audio is generated
        audio_data, sample_rate = tts_service.generate_audio(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )

        # Convert to int16
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        # Stream audio in chunks
        async def audio_generator():
            chunk_size = 4096  # 4KB chunks
            audio_bytes = audio_int16.tobytes()

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                yield chunk
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.01)

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Channels": "1",
                "X-Bit-Depth": "16"
            }
        )

    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FastAPI TTS Server",
        "model_loaded": tts_service.pipeline is not None
    }


@app.get("/voices")
async def list_voices():
    """List available voices"""
    # This is a simplified list - in production you'd extract from the model
    return {
        "voices": [
            "af_bella", "af_sarah", "am_adam", "am_michael",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
            "ef_rebecca", "ef_serena", "em_david", "em_james",
            "ff_siwis", "hf_alpha", "hm_omega", "if_sara",
            "im_nicola", "jf_alpha", "jf_gongitsune", "jm_kumo",
            "pf_dora", "pm_alan", "zf_xiaobei", "zf_xiaoni",
            "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
            "zm_yunxia", "zm_yunyang"
        ]
    }


if __name__ == "__main__":
    # Force single-threaded execution to prevent Metal conflicts
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    # Use Unix socket for local communication
    socket_path = "/tmp/fastapi-tts.sock"

    # Remove existing socket if it exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    logger.info("🚀 Starting FastAPI TTS Server with Unix socket")
    uvicorn.run(
        app,
        uds=socket_path,
        workers=1,  # Single worker to avoid Metal conflicts
        log_level="info"
    )