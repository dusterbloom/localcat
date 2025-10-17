#!/usr/bin/env python3
"""
Test the real Parakeet-MLX streaming implementation
"""

import asyncio
import sys
import time
import wave
from pathlib import Path
import difflib

sys.path.insert(0, str(Path(__file__).parent))

HARVARD_GROUND_TRUTH = """
The stale smell of old beer lingers. It takes heat to bring out the odor.
A cold dip restores health and zest. A salt pickle tastes fine with ham.
Tacos al pastor are my favorite. A zestful food is the hot cross bun.
""".strip()

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    operations = matcher.get_opcodes()

    total_errors = 0
    for op, i1, i2, j1, j2 in operations:
        if op == 'replace':
            total_errors += max(i2 - i1, j2 - j1)
        elif op == 'delete':
            total_errors += i2 - i1
        elif op == 'insert':
            total_errors += j2 - j1

    wer = (total_errors / len(ref_words)) * 100 if ref_words else 0
    return wer

async def test_real_streaming():
    """Test with the proper parakeet_mlx streaming API"""
    print("🚀 Testing REAL Parakeet-MLX Streaming")
    print("=" * 50)

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        # Initialize with streaming-optimized settings
        stt = ParakeetStreamingSTT(
            enable_vad=False,  # Disable VAD for pure streaming test
            volume_threshold=0.0001,
            chunk_duration=1.0,
            context_size=(256, 256),
            depth=3  # Higher depth for better quality
        )

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")

        # Test 1: Direct mode (baseline)
        print("\n🎯 DIRECT MODE:")
        start_time = time.time()
        direct_result = stt._process_audio_file_fallback(audio_data)
        direct_time = time.time() - start_time
        direct_wer = calculate_wer(HARVARD_GROUND_TRUTH, direct_result)
        print(f"   Time: {direct_time:.2f}s, RTF: {direct_time/duration:.3f}, WER: {direct_wer:.1f}%")
        print(f"   📝 Result: '{direct_result}'")

        # Test 2: Real streaming mode
        print("\n🌊 STREAMING MODE:")
        start_time = time.time()
        transcriptions = []

        # Process in 1-second chunks to test real streaming
        chunk_size = 16000  # 1 second at 16kHz
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < 1000:
                continue

            chunk_num = i // chunk_size + 1
            print(f"   Processing chunk {chunk_num}...")

            async for frame in stt.run_stt(chunk):
                if hasattr(frame, 'text') and frame.text.strip():
                    text = frame.text.strip()
                    transcriptions.append(text)
                    print(f"     ✅ '{text}'")

        # Flush remaining
        async for frame in stt.flush():
            if hasattr(frame, 'text') and frame.text.strip():
                text = frame.text.strip()
                transcriptions.append(text)
                print(f"   🔄 Flush: '{text}'")

        streaming_time = time.time() - start_time
        streaming_result = " ".join(transcriptions)
        streaming_wer = calculate_wer(HARVARD_GROUND_TRUTH, streaming_result)

        print(f"\n📊 STREAMING RESULTS:")
        print(f"   Time: {streaming_time:.2f}s, RTF: {streaming_time/duration:.3f}, WER: {streaming_wer:.1f}%")
        print(f"   Segments: {len(transcriptions)}")
        print(f"   📝 Full: '{streaming_result}'")

        # Comparison
        print(f"\n🏆 COMPARISON:")
        print(f"   Direct:    RTF: {direct_time/duration:.3f}, WER: {direct_wer:.1f}%")
        print(f"   Streaming: RTF: {streaming_time/duration:.3f}, WER: {streaming_wer:.1f}%")

        if streaming_wer < 50:  # Good streaming quality
            print(f"   🎉 SUCCESS: Real streaming is working!")
        else:
            print(f"   ⚠️  Issue: Streaming quality needs improvement")

        return streaming_wer < 50

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_streaming())
    sys.exit(0 if success else 1)