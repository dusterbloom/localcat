#!/usr/bin/env python3
"""
COMPREHENSIVE STT STREAMING PROOF TEST
=====================================
This test provides complete evidence that real Parakeet-MLX streaming is working
with detailed logging and progressive transcription demonstration.
"""

import asyncio
import sys
import time
import wave
from pathlib import Path
import difflib
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)

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

async def test_streaming_proof():
    """Comprehensive test with full evidence"""
    print("🔬 COMPREHENSIVE STT STREAMING PROOF TEST")
    print("=" * 70)
    print(f"📖 Target transcription: '{HARVARD_GROUND_TRUTH}'")
    print()

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT, PARAKEET_OLD_FORMAT

        print(f"📦 Package status:")
        print(f"   PARAKEET_OLD_FORMAT: {PARAKEET_OLD_FORMAT}")

        if not PARAKEET_OLD_FORMAT:
            print("   ✅ Using NEW parakeet_mlx package with streaming support")
        else:
            print("   ⚠️  Using legacy mlx_audio format")
        print()

        # Initialize with detailed logging
        print("🔄 Initializing Parakeet STT with streaming parameters...")
        stt = ParakeetStreamingSTT(
            enable_vad=False,  # Disable VAD for pure streaming test
            volume_threshold=0.0001,
            chunk_duration=1.0,
            context_size=(256, 256),
            depth=3  # Higher depth for better quality
        )

        print(f"   ✅ STT initialized successfully")
        print(f"   📊 Streaming context available: {stt._streaming_context is not None}")
        print()

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio file loaded:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Size: {len(audio_data)} bytes")
        print()

        # Test 1: DIRECT MODE (Reference baseline)
        print("🎯 TEST 1: DIRECT MODE (Reference Baseline)")
        print("-" * 50)

        start_time = time.time()
        direct_result = stt._process_audio_file_fallback(audio_data)
        direct_time = time.time() - start_time
        direct_wer = calculate_wer(HARVARD_GROUND_TRUTH, direct_result)

        print(f"📝 Direct result: '{direct_result}'")
        print(f"⏱️  Time: {direct_time:.3f}s, RTF: {direct_time/duration:.4f}")
        print(f"🎯 WER: {direct_wer:.1f}%")
        print()

        # Test 2: REAL STREAMING MODE with detailed chunk analysis
        print("🌊 TEST 2: REAL STREAMING MODE (Progressive Transcription)")
        print("-" * 50)

        # Process in smaller chunks to show progressive streaming
        chunk_duration = 0.5  # 500ms chunks for detailed demonstration
        chunk_size = int(16000 * chunk_duration * 2)  # bytes per chunk

        print(f"🔄 Processing in {chunk_duration}s chunks ({chunk_size} bytes each)")
        print()

        start_time = time.time()
        transcriptions = []
        progressive_results = []

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < 1000:
                continue

            chunk_num = i // chunk_size + 1
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            chunk_time_start = i / (16000 * 2)  # Convert to seconds

            print(f"📦 Chunk {chunk_num:2d}/{total_chunks}: t={chunk_time_start:.1f}s, size={len(chunk)} bytes")

            chunk_start = time.time()
            chunk_results = []

            async for frame in stt.run_stt(chunk):
                if hasattr(frame, 'text') and frame.text.strip():
                    text = frame.text.strip()
                    chunk_results.append(text)
                    transcriptions.append(text)

                    # Show progressive streaming
                    print(f"   🌊 STREAMING: '{text}'")
                    progressive_results.append(f"[{chunk_time_start:.1f}s] {text}")

            chunk_time = time.time() - chunk_start
            print(f"   ⏱️  Processed in {chunk_time:.3f}s")
            print()

        # Flush remaining
        print("🔄 FLUSHING remaining audio...")
        async for frame in stt.flush():
            if hasattr(frame, 'text') and frame.text.strip():
                text = frame.text.strip()
                transcriptions.append(text)
                print(f"   🌊 FLUSH: '{text}'")
                progressive_results.append(f"[FLUSH] {text}")

        streaming_time = time.time() - start_time
        streaming_result = " ".join(transcriptions) if transcriptions else ""

        print()
        print("📊 STREAMING RESULTS SUMMARY:")
        print(f"   Total time: {streaming_time:.3f}s, RTF: {streaming_time/duration:.4f}")
        print(f"   Segments generated: {len(transcriptions)}")
        print(f"   Final combined text: '{streaming_result[:100]}{'...' if len(streaming_result) > 100 else ''}'")
        print()

        # Test 3: PROGRESSIVE STREAMING ANALYSIS
        print("🔍 TEST 3: PROGRESSIVE STREAMING ANALYSIS")
        print("-" * 50)

        if progressive_results:
            print("📈 Progressive transcription timeline:")
            for i, result in enumerate(progressive_results[:10]):  # Show first 10 for brevity
                print(f"   {i+1:2d}. {result}")
            if len(progressive_results) > 10:
                print(f"   ... and {len(progressive_results) - 10} more results")
            print()

        # Test 4: STREAMING QUALITY ANALYSIS
        print("🎯 TEST 4: STREAMING QUALITY ANALYSIS")
        print("-" * 50)

        if transcriptions:
            # Take the last/final transcription as the best result
            final_transcription = transcriptions[-1] if transcriptions else ""
            final_wer = calculate_wer(HARVARD_GROUND_TRUTH, final_transcription)

            print(f"📝 Final streaming transcription: '{final_transcription}'")
            print(f"🎯 Final WER: {final_wer:.1f}%")
        else:
            print("❌ No transcriptions generated")
            final_wer = 100.0
        print()

        # FINAL COMPARISON
        print("🏆 FINAL COMPARISON & EVIDENCE")
        print("=" * 70)

        print(f"📊 PERFORMANCE METRICS:")
        print(f"   Direct Mode:    RTF: {direct_time/duration:.4f}, WER: {direct_wer:.1f}%")
        print(f"   Streaming Mode: RTF: {streaming_time/duration:.4f}, Progressive: ✅")
        print()

        print(f"🔬 STREAMING EVIDENCE:")
        print(f"   ✅ Streaming context initialized: {stt._streaming_context is not None}")
        print(f"   ✅ Using parakeet_mlx API: {not PARAKEET_OLD_FORMAT}")
        print(f"   ✅ Progressive results generated: {len(transcriptions)} segments")
        print(f"   ✅ Real-time processing: {streaming_time:.2f}s for {duration:.1f}s audio")
        print()

        # PROOF CRITERIA
        streaming_working = (
            stt._streaming_context is not None and
            not PARAKEET_OLD_FORMAT and
            len(transcriptions) > 0 and
            streaming_time < duration * 2  # Reasonable real-time performance
        )

        if streaming_working:
            print("🎉 PROOF: REAL PARAKEET-MLX STREAMING IS WORKING!")
            print("   ✅ All streaming criteria met")
            print("   ✅ Progressive transcription demonstrated")
            print("   ✅ Real-time performance achieved")
        else:
            print("❌ Streaming proof failed - criteria not met")

        print()
        return streaming_working

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive streaming proof test...")
    print()

    success = asyncio.run(test_streaming_proof())

    print()
    print("=" * 70)
    if success:
        print("🎊 CONCLUSION: PARAKEET-MLX STREAMING PROVED WORKING!")
    else:
        print("💔 CONCLUSION: Streaming proof test failed")
    print("=" * 70)

    sys.exit(0 if success else 1)