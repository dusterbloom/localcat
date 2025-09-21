#!/usr/bin/env python3
"""Integration test to verify the updated bot.py works with native Kokoro."""

import asyncio
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))

# Test the TTS integration by importing and running a simple test
async def _run_integration():
    """Test that the native Kokoro TTS integrates properly."""

    print("🧪 Testing Native Kokoro TTS Integration")
    print("=" * 50)

    try:
        # Import the TTS service used in bot.py
        from tts_native_kokoro import NativeKokoroTTSService
        from pipecat.frames.frames import TTSAudioRawFrame

        print("✅ Import successful")

        # Initialize the same way as bot.py
        tts = NativeKokoroTTSService(
            voice="af_heart",
            speed=1.0,
            sample_rate=24000
        )

        print("✅ TTS initialization successful")

        # Test a simple generation
        test_text = "Hello, this is a test of the native Kokoro integration."
        print(f"🎤 Testing: '{test_text}'")

        start_time = time.time()
        audio_received = False
        frame_generator = tts.run_tts(test_text)

        try:
            async for frame in frame_generator:
                if isinstance(frame, TTSAudioRawFrame):
                    if not audio_received:
                        ttfb = (time.time() - start_time) * 1000
                        print(f"🚀 TTFB: {ttfb:.1f}ms")
                        print(f"📊 Audio: {len(frame.audio)} bytes at {frame.sample_rate}Hz")
                        audio_received = True
                        # Continue to consume all frames for proper cleanup
        except GeneratorExit:
            pass

        if audio_received:
            print("✅ Integration test PASSED - TTS working correctly")
            return True

        print("❌ Integration test FAILED - No audio received")
        return False

    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Pytest entrypoint that runs the async integration coroutine."""
    success = asyncio.run(_run_integration())
    assert success

if __name__ == "__main__":
    success = asyncio.run(_run_integration())
    sys.exit(0 if success else 1)
