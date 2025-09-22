#!/usr/bin/env python3
"""
Test FastAPI Streaming TTS Service integration
"""

import asyncio
import time
# Skipped - module doesn't exist:  FastAPIStreamingTTS


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