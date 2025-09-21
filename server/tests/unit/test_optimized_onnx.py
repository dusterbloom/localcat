#!/usr/bin/env python3
"""
Test the optimized ONNX implementation.
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tts_native_kokoro import NativeKokoroTTSService


async def test_optimized_onnx():
    """Test the optimized ONNX implementation."""

    test_texts = [
        "Hello world",
        "This is a test of the optimized ONNX implementation with sentence aggregation.",
        "The quick brown fox jumps over the lazy dog and runs through the forest with great speed."
    ]

    print("🧪 Testing Optimized ONNX Kokoro Implementation\n")

    print("📊 Testing Optimized ONNX Kokoro...")
    try:
        async with NativeKokoroTTSService(voice="af_bella") as onnx_tts:
            total_start = time.time()

            for i, text in enumerate(test_texts):
                print(f"  Test {i+1}: {text[:60]}{'...' if len(text) > 60 else ''}")

                start_time = time.time()
                frames = []
                audio_frames = 0

                async for frame in onnx_tts.run_tts(text):
                    frames.append(frame)
                    if hasattr(frame, 'audio'):
                        audio_frames += 1

                generation_time = (time.time() - start_time) * 1000
                chars_per_sec = len(text) / (generation_time / 1000) if generation_time > 0 else 0

                print(f"    ⚡ {generation_time:.1f}ms ({chars_per_sec:.1f} chars/s) - {audio_frames} audio frames")

            total_time = (time.time() - total_start) * 1000
            print(f"\n📈 Total test time: {total_time:.1f}ms")
            print("✅ Optimized ONNX test completed!")

    except Exception as e:
        print(f"❌ Optimized ONNX test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_optimized_onnx())