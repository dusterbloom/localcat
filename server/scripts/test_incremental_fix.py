#!/usr/bin/env python3
"""
Test the incremental text extraction fix for Parakeet streaming
"""

import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_incremental_extraction():
    """Test that streaming now extracts only incremental text"""
    print("🔬 Testing Incremental Text Extraction Fix")
    print("=" * 50)

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        # Initialize STT
        stt = ParakeetStreamingSTT(
            enable_vad=False,
            confidence_threshold=0.1,
            volume_threshold=0.0001,
            chunk_duration=1.0,
            context_size=(256, 256),
            depth=3
        )

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")
        print()

        # Test streaming with incremental extraction
        print("🌊 Testing Incremental Streaming:")
        print("-" * 30)

        chunk_size = 32000  # 2 seconds at 16kHz
        all_segments = []
        cumulative_text = ""

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < 1000:
                continue

            chunk_num = i // chunk_size + 1
            chunk_time = i / (16000 * 2)

            print(f"📦 Chunk {chunk_num}: t={chunk_time:.1f}s")

            async for frame in stt.run_stt(chunk):
                if hasattr(frame, 'text') and frame.text.strip():
                    incremental_text = frame.text.strip()
                    all_segments.append(incremental_text)
                    cumulative_text += " " + incremental_text

                    print(f"   ✅ NEW: '{incremental_text}'")
                    print(f"   📝 CUMULATIVE: '{cumulative_text.strip()[:80]}{'...' if len(cumulative_text.strip()) > 80 else ''}'")
                    print()

            # Stop after a few chunks to see the pattern
            if chunk_num >= 6:
                break

        # Flush
        print("🔄 Flushing...")
        async for frame in stt.flush():
            if hasattr(frame, 'text') and frame.text.strip():
                incremental_text = frame.text.strip()
                all_segments.append(incremental_text)
                cumulative_text += " " + incremental_text
                print(f"   ✅ FLUSH: '{incremental_text}'")

        print()
        print("📊 RESULTS:")
        print(f"   Segments: {len(all_segments)}")
        print(f"   Final text: '{cumulative_text.strip()}'")
        print()

        # Check for repetition issue
        has_repetition = False
        for i, segment in enumerate(all_segments):
            for j, other_segment in enumerate(all_segments[i+1:], i+1):
                if segment in other_segment or other_segment in segment:
                    if len(segment) > 10 and len(other_segment) > 10:  # Ignore short segments
                        has_repetition = True
                        print(f"⚠️  Potential repetition detected:")
                        print(f"     Segment {i}: '{segment}'")
                        print(f"     Segment {j}: '{other_segment}'")

        if not has_repetition:
            print("✅ SUCCESS: No repetition detected in segments!")
        else:
            print("❌ Issue: Some repetition still detected")

        return not has_repetition

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_incremental_extraction())
    print()
    if success:
        print("🎉 Incremental extraction fix appears to be working!")
    else:
        print("💔 Fix needs more work")

    sys.exit(0 if success else 1)