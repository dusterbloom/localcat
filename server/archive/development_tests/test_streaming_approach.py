#!/usr/bin/env python3
"""Test the new streaming MLX approach vs the old worker approach."""

import asyncio
import sys
import os
import time
import logging

# Disable logging for cleaner output
logging.disable(logging.CRITICAL)

sys.path.append(os.path.dirname(__file__))

from tts_mlx_streaming import TTSMLXStreaming
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

async def test_streaming_approach():
    """Test the new direct MLX streaming approach."""

    test_cases = [
        ("Short", "Hello there!"),
        ("Medium", "That's a great question! I can help you with that."),
        ("Long", "Hello there! How are you doing today? I hope you're having a wonderful day so far."),
        ("Complex", "The implementation of streaming TTS represents a significant advancement in speech synthesis technology.")
    ]

    print("🚀 Testing NEW STREAMING MLX Approach")
    print("Target: 40-80ms TTFB (Kokoro FastAPI best practices)")
    print("=" * 60)

    # Initialize TTS with new streaming approach
    tts = TTSMLXStreaming(
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_heart",
        sample_rate=24000
    )

    print("✅ Streaming TTS initialized")

    for name, text in test_cases:
        print(f"\n🧪 Test {name}: '{text}'")

        start_time = time.time()
        chunk_count = 0
        total_bytes = 0
        ttfb = None

        async for frame in tts.run_tts(text):
            if isinstance(frame, TTSAudioRawFrame):
                chunk_count += 1
                total_bytes += len(frame.audio)
                if ttfb is None:
                    ttfb = (time.time() - start_time) * 1000
                    print(f"🚀 TTFB: {ttfb:.1f}ms, chunk: {len(frame.audio)} bytes")

        total_time = (time.time() - start_time) * 1000
        print(f"📊 Total: {total_time:.1f}ms, {chunk_count} chunks, {total_bytes:,} bytes")

        # Check against target
        if ttfb and ttfb <= 80:
            print("✅ EXCELLENT: Within 40-80ms target!")
        elif ttfb and ttfb <= 200:
            print("🟡 GOOD: Under 200ms")
        else:
            print("❌ TOO SLOW: Far from 40-80ms target")

        await asyncio.sleep(0.5)  # Brief pause between tests

if __name__ == "__main__":
    asyncio.run(test_streaming_approach())