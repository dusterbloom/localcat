#!/usr/bin/env python3
"""
Performance comparison test between ONNX and MLX Kokoro TTS implementations.

This script directly tests the generation speed and quality of both TTS services
to validate the MLX performance improvements.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tts_native_kokoro import NativeKokoroTTSService
from tts_mlx_kokoro import MLXKokoroTTSService


async def test_tts_performance():
    """Compare ONNX vs MLX Kokoro TTS performance."""

    test_texts = [
        "Hello, this is a quick test.",
        "The quick brown fox jumps over the lazy dog and runs through the forest.",
        "This is a longer sentence designed to test the text-to-speech system's performance with more complex content that includes punctuation and various word patterns.",
        "Testing multiple sentences. Each sentence should flow naturally. The goal is smooth, continuous speech without gaps or delays.",
        "Numbers and symbols: testing 123 with special characters like @#$ and punctuation marks!"
    ]

    print("🧪 TTS Performance Comparison: ONNX vs MLX Kokoro\n")

    # Test ONNX implementation
    print("📊 Testing ONNX Kokoro TTS...")
    try:
        async with NativeKokoroTTSService(voice="af_bella") as onnx_tts:
            onnx_results = []

            for i, text in enumerate(test_texts):
                print(f"  Test {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")

                start_time = time.time()
                frames = []

                async for frame in onnx_tts.run_tts(text):
                    frames.append(frame)

                generation_time = (time.time() - start_time) * 1000
                chars_per_sec = len(text) / (generation_time / 1000) if generation_time > 0 else 0

                onnx_results.append({
                    'text_length': len(text),
                    'generation_time_ms': generation_time,
                    'chars_per_sec': chars_per_sec,
                    'frame_count': len(frames)
                })

                print(f"    ⏱️  {generation_time:.1f}ms ({chars_per_sec:.1f} chars/s)")

    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
        onnx_results = []

    print()

    # Test MLX implementation
    print("🚀 Testing MLX Kokoro TTS...")
    try:
        async with MLXKokoroTTSService(voice="af_bella") as mlx_tts:
            mlx_results = []

            for i, text in enumerate(test_texts):
                print(f"  Test {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")

                start_time = time.time()
                frames = []

                async for frame in mlx_tts.run_tts(text):
                    frames.append(frame)

                generation_time = (time.time() - start_time) * 1000
                chars_per_sec = len(text) / (generation_time / 1000) if generation_time > 0 else 0

                mlx_results.append({
                    'text_length': len(text),
                    'generation_time_ms': generation_time,
                    'chars_per_sec': chars_per_sec,
                    'frame_count': len(frames)
                })

                print(f"    ⚡ {generation_time:.1f}ms ({chars_per_sec:.1f} chars/s)")

    except Exception as e:
        print(f"❌ MLX test failed: {e}")
        mlx_results = []

    print()

    # Performance comparison
    if onnx_results and mlx_results:
        print("📈 Performance Comparison Summary:")
        print("=" * 60)

        avg_onnx_time = sum(r['generation_time_ms'] for r in onnx_results) / len(onnx_results)
        avg_mlx_time = sum(r['generation_time_ms'] for r in mlx_results) / len(mlx_results)

        avg_onnx_chars = sum(r['chars_per_sec'] for r in onnx_results) / len(onnx_results)
        avg_mlx_chars = sum(r['chars_per_sec'] for r in mlx_results) / len(mlx_results)

        speedup = avg_onnx_time / avg_mlx_time if avg_mlx_time > 0 else 0
        throughput_improvement = avg_mlx_chars / avg_onnx_chars if avg_onnx_chars > 0 else 0

        print(f"Average Generation Time:")
        print(f"  ONNX: {avg_onnx_time:.1f}ms")
        print(f"  MLX:  {avg_mlx_time:.1f}ms")
        print(f"  💪 Speedup: {speedup:.1f}x faster")
        print()

        print(f"Average Throughput:")
        print(f"  ONNX: {avg_onnx_chars:.1f} chars/sec")
        print(f"  MLX:  {avg_mlx_chars:.1f} chars/sec")
        print(f"  💪 Improvement: {throughput_improvement:.1f}x better")
        print()

        # Detailed results
        print("Detailed Results:")
        print("-" * 60)
        for i, (onnx, mlx) in enumerate(zip(onnx_results, mlx_results)):
            test_speedup = onnx['generation_time_ms'] / mlx['generation_time_ms'] if mlx['generation_time_ms'] > 0 else 0
            print(f"Test {i+1} ({onnx['text_length']} chars):")
            print(f"  ONNX: {onnx['generation_time_ms']:.1f}ms")
            print(f"  MLX:  {mlx['generation_time_ms']:.1f}ms")
            print(f"  Speedup: {test_speedup:.1f}x")

        # Performance verdict
        if speedup >= 3.0:
            print("🎉 EXCELLENT: MLX delivers 3x+ performance improvement!")
        elif speedup >= 2.0:
            print("✅ GOOD: MLX delivers 2x+ performance improvement!")
        elif speedup >= 1.5:
            print("👍 BETTER: MLX delivers meaningful performance improvement!")
        elif speedup >= 1.1:
            print("📈 MODEST: MLX delivers slight performance improvement")
        else:
            print("⚠️  CONCERNING: MLX performance is not better than ONNX")

    else:
        print("❌ Could not complete performance comparison due to test failures")


if __name__ == "__main__":
    asyncio.run(test_tts_performance())