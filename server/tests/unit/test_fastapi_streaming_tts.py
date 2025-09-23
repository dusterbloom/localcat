#!/usr/bin/env python3
"""
Test FastAPI Streaming TTS Service integration
"""

import asyncio
import time
import sys
import os

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.tts.kokoro_mlx import MLXKokoroTTSService as FastAPIStreamingTTS


async def test_fastapi_streaming_tts():
    """Test the FastAPI streaming TTS service"""

    print("🧪 Testing FastAPI Streaming TTS Service")
    print("=" * 50)

    async with FastAPIStreamingTTS(
        voice="af_bella",
        speed=1.0,
        sample_rate=24000
    ) as tts:

        test_texts = [
            "Hello, world!",
            "This is a test of the FastAPI streaming TTS service.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {len(text)} chars")
            print(f"Text: {text}")

            start_time = time.time()
            frame_count = 0
            total_audio_bytes = 0

            async for frame in tts.run_tts(text):
                frame_count += 1
                if hasattr(frame, 'audio'):
                    total_audio_bytes += len(frame.audio)

            total_time = time.time() - start_time

            print(".2f")
            print(f"Frames generated: {frame_count}")
            print(f"Total audio bytes: {total_audio_bytes}")

    print("\n✅ FastAPI Streaming TTS Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_fastapi_streaming_tts())