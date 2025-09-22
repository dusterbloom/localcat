#!/usr/bin/env python3
"""
Test Parakeet STT with different depth parameters to compare streaming quality
Following official Parakeet-MLX documentation
"""

import asyncio
import numpy as np
import sys
import time
import wave
from pathlib import Path
import difflib

# Add server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Ground truth for WER calculation
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

async def test_parakeet_depth(depth: int, context_size: tuple = (256, 256)):
    """Test Parakeet with specific depth parameter"""
    print(f"\n🔬 Testing Parakeet with depth={depth}, context_size={context_size}")
    print("-" * 60)

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        # Initialize with specific depth
        stt = ParakeetStreamingSTT(
            enable_vad=False,  # Disable to focus on streaming quality
            confidence_threshold=0.1,
            volume_threshold=0.0001,
            chunk_duration=1.0,
            context_size=context_size,
            depth=depth
        )

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")

        # Test 1: Direct processing (baseline)
        print("🎯 Direct mode (baseline):")
        start_time = time.time()
        direct_result = stt._process_audio_file_fallback(audio_data)
        direct_time = time.time() - start_time
        direct_wer = calculate_wer(HARVARD_GROUND_TRUTH, direct_result)
        print(f"   Time: {direct_time:.2f}s, RTF: {direct_time/duration:.3f}, WER: {direct_wer:.1f}%")
        print(f"   Result: '{direct_result}'")

        # Test 2: Streaming processing with depth
        print(f"🌊 Streaming mode (depth={depth}):")
        start_time = time.time()
        transcriptions = []

        # Process in 1-second chunks
        chunk_size = 32000  # 1 second at 16kHz
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < 1000:
                continue

            async for frame in stt.run_stt(chunk):
                if hasattr(frame, 'text') and frame.text.strip():
                    transcriptions.append(frame.text.strip())

        # Flush remaining
        async for frame in stt.flush():
            if hasattr(frame, 'text') and frame.text.strip():
                transcriptions.append(frame.text.strip())

        streaming_time = time.time() - start_time
        streaming_result = " ".join(transcriptions)
        streaming_wer = calculate_wer(HARVARD_GROUND_TRUTH, streaming_result)

        print(f"   Time: {streaming_time:.2f}s, RTF: {streaming_time/duration:.3f}, WER: {streaming_wer:.1f}%")
        print(f"   Segments: {len(transcriptions)}")
        print(f"   Result: '{streaming_result}'")

        return {
            "depth": depth,
            "context_size": context_size,
            "direct_wer": direct_wer,
            "streaming_wer": streaming_wer,
            "direct_time": direct_time,
            "streaming_time": streaming_time,
            "direct_rtf": direct_time / duration,
            "streaming_rtf": streaming_time / duration,
            "segments": len(transcriptions),
            "direct_result": direct_result,
            "streaming_result": streaming_result
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {
            "depth": depth,
            "context_size": context_size,
            "error": str(e)
        }

async def main():
    """Compare different depth parameters"""
    print("🎯 Parakeet Streaming Depth Comparison")
    print("=" * 70)
    print(f"📖 Ground Truth: '{HARVARD_GROUND_TRUTH}'")

    results = []

    # Test different depth values
    depth_values = [1, 2, 3, 4]  # According to docs, higher depth = better quality

    for depth in depth_values:
        result = await test_parakeet_depth(depth)
        results.append(result)

    # Summary comparison
    print("\n🏆 DEPTH COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Depth':<6} {'Direct RTF':<10} {'Stream RTF':<10} {'Direct WER':<11} {'Stream WER':<11} {'Quality Impact'}")
    print("-" * 70)

    valid_results = [r for r in results if 'error' not in r]

    for result in valid_results:
        quality_impact = result['streaming_wer'] - result['direct_wer']
        quality_desc = "🟢 Better" if quality_impact < 0 else "🟡 Same" if quality_impact == 0 else f"🔴 +{quality_impact:.1f}%"

        print(f"{result['depth']:<6} {result['direct_rtf']:<10.3f} {result['streaming_rtf']:<10.3f} "
              f"{result['direct_wer']:<11.1f} {result['streaming_wer']:<11.1f} {quality_desc}")

    if valid_results:
        print("\n📊 ANALYSIS:")

        # Best streaming quality (lowest WER)
        best_quality = min(valid_results, key=lambda x: x['streaming_wer'])
        print(f"🎯 Best Streaming Quality: depth={best_quality['depth']} (WER: {best_quality['streaming_wer']:.1f}%)")

        # Best streaming speed (lowest RTF)
        best_speed = min(valid_results, key=lambda x: x['streaming_rtf'])
        print(f"🚀 Fastest Streaming: depth={best_speed['depth']} (RTF: {best_speed['streaming_rtf']:.3f})")

        # Best balance (consider both quality and speed)
        for result in valid_results:
            result['score'] = (100 - result['streaming_wer']) / result['streaming_rtf']

        best_balance = max(valid_results, key=lambda x: x['score'])
        print(f"⚖️  Best Balance: depth={best_balance['depth']} (WER: {best_balance['streaming_wer']:.1f}%, RTF: {best_balance['streaming_rtf']:.3f})")

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • For best quality: depth={best_quality['depth']}")
        print(f"   • For best speed: depth={best_speed['depth']}")
        print(f"   • For balanced performance: depth={best_balance['depth']}")

if __name__ == "__main__":
    asyncio.run(main())