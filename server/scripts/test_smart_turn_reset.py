#!/usr/bin/env python3
"""
Test that Parakeet streaming resets properly with Smart Turn events
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_smart_turn_reset():
    """Test that UserStartedSpeakingFrame resets the streaming context"""
    print("🧪 Testing Smart Turn Reset for Parakeet Streaming")
    print("=" * 50)

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        # Initialize STT
        stt = ParakeetStreamingSTT(
            enable_vad=False,
            volume_threshold=0.0001,
            chunk_duration=1.0,
            context_size=(256, 256),
            depth=3
        )

        print("✅ STT initialized")

        # Load test audio
        import wave
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())

        # Process first utterance
        print("\n📢 First utterance:")
        chunk = audio_data[:32000]  # First 2 seconds

        # Simulate user starts speaking
        await stt.process_frame(UserStartedSpeakingFrame())

        async for frame in stt.run_stt(chunk):
            if hasattr(frame, 'text') and frame.text.strip():
                print(f"   → '{frame.text.strip()}'")

        # Simulate user stops speaking
        await stt.process_frame(UserStoppedSpeakingFrame())

        # Process second utterance (different audio)
        print("\n📢 Second utterance (should NOT include first):")
        chunk = audio_data[64000:96000]  # Different 2 seconds

        # Simulate user starts speaking again - this should reset context
        await stt.process_frame(UserStartedSpeakingFrame())

        async for frame in stt.run_stt(chunk):
            if hasattr(frame, 'text') and frame.text.strip():
                text = frame.text.strip()
                print(f"   → '{text}'")

                # Check if it includes text from first utterance
                if "steel smell" in text.lower() or "stale smell" in text.lower():
                    print("   ❌ ERROR: Contains text from first utterance!")
                    return False

        print("\n✅ SUCCESS: Second utterance did not contain first utterance text")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_smart_turn_reset())
    print()
    if success:
        print("🎉 Smart Turn reset is working correctly!")
    else:
        print("💔 Smart Turn reset needs more work")

    sys.exit(0 if success else 1)