#!/usr/bin/env python3
"""Test the ultra-low latency TTS approach."""

import asyncio
import sys
import os
import time
import logging

# Disable logging for cleaner output
logging.disable(logging.CRITICAL)

sys.path.append(os.path.dirname(__file__))

from tts_ultra_low_latency import UltraLowLatencyTTS
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

async def test_ultra_low_latency():
    """Test the ultra-low latency phrase streaming approach."""

    test_cases = [
        ("Short", "Hello there!"),
        ("Medium", "That's a great question! I can help you with that."),
        ("Long", "Hello there! How are you doing today? I hope you're having a wonderful day so far."),
        ("Complex", "The implementation of streaming TTS represents a significant advancement in speech synthesis technology that enables real-time voice applications.")
    ]

    print("🚀 Testing ULTRA-LOW LATENCY TTS (Phrase Streaming)")
    print("Target: 40-80ms TTFB through phrase-level processing")
    print("=" * 60)

    try:
        # Initialize TTS with ultra-low latency approach
        tts = UltraLowLatencyTTS(
            voice="af_bella",
            speed=1.0,
            sample_rate=24000
        )

        print("✅ Ultra-low latency TTS initialized")

        for name, text in test_cases:
            print(f"\n🧪 Test {name}: '{text}'")

            start_time = time.time()
            chunk_count = 0
            total_bytes = 0
            ttfb = None
            phrase_count = 0

            async for frame in tts.run_tts(text):
                if isinstance(frame, TTSAudioRawFrame):
                    chunk_count += 1
                    total_bytes += len(frame.audio)
                    if ttfb is None:
                        ttfb = (time.time() - start_time) * 1000
                        print(f"🚀 TTFB: {ttfb:.1f}ms, first chunk: {len(frame.audio)} bytes")

            total_time = (time.time() - start_time) * 1000
            print(f"📊 Total: {total_time:.1f}ms, {chunk_count} chunks, {total_bytes:,} bytes")

            # Check against target
            if ttfb and ttfb <= 80:
                print("✅ EXCELLENT: Within 40-80ms target!")
            elif ttfb and ttfb <= 200:
                print("🟡 GOOD: Under 200ms")
            elif ttfb and ttfb <= 500:
                print("🟠 OKAY: Under 500ms")
            else:
                print("❌ TOO SLOW: Far from 40-80ms target")

            await asyncio.sleep(1.0)  # Pause between tests

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ultra_low_latency())