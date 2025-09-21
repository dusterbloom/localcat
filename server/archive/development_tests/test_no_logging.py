#!/usr/bin/env python3
"""Test TTS with minimal logging to check if logging is causing latency."""

import asyncio
import sys
import os
import time
import logging

# Disable all logging for this test
logging.disable(logging.CRITICAL)

sys.path.append(os.path.dirname(__file__))

from tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

async def test_no_logging():
    """Test simple cases with all logging disabled."""

    test_cases = [
        "Hello there!",
        "That's a great question!",
        "Hello there! How are you doing today?",
        "That's a great question! I can help you with that."
    ]

    print("🧪 Testing TTS with NO LOGGING")
    print("Target: 40-80ms TTFB (Kokoro FastAPI best practices)")
    print("=" * 60)

    # Initialize TTS with minimal settings
    tts = TTSMLXUltraLowLatency(
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_heart",
        use_boundaries=False,
        buffer_ms=50
    )

    if not await tts._initialize_if_needed():
        print("❌ Failed to initialize TTS")
        return

    print("✅ TTS initialized (no logging)")

    for i, text in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: '{text}'")

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

if __name__ == "__main__":
    asyncio.run(test_no_logging())