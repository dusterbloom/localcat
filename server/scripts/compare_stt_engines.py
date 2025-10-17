#!/usr/bin/env python3
"""
Compare the 3 STT engines: Kyutai, Whisper MLX, and Parakeet
Test quality, latency, and WER characteristics with Harvard audio
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

# Ground truth Harvard sentences for WER calculation
HARVARD_GROUND_TRUTH = """
The stale smell of old beer lingers. It takes heat to bring out the odor.
A cold dip restores health and zest. A salt pickle tastes fine with ham.
Tacos al pastor are my favorite. A zestful food is the hot cross bun.
""".strip()

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Use difflib to calculate edit distance
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    operations = matcher.get_opcodes()

    substitutions = 0
    deletions = 0
    insertions = 0

    for op, i1, i2, j1, j2 in operations:
        if op == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif op == 'delete':
            deletions += i2 - i1
        elif op == 'insert':
            insertions += j2 - j1

    total_errors = substitutions + deletions + insertions
    total_words = len(ref_words)

    wer = (total_errors / total_words) * 100 if total_words > 0 else 0
    return wer

async def test_kyutai_stt():
    """Test Kyutai STT engine"""
    print("\n🔬 Testing Kyutai STT Engine")
    print("=" * 50)

    try:
        from core.stt.kyutai_streaming import KyutaiStreamingSTT

        # Initialize with optimized settings
        stt = KyutaiStreamingSTT(
            hf_repo="kyutai/moshi-mlx",
            enable_vad=False,  # Disable to avoid Smart Turn conflicts
            max_steps=16384
        )

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")

        # Test streaming performance
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

        processing_time = time.time() - start_time
        rtf = processing_time / duration

        full_text = " ".join(transcriptions)
        wer = calculate_wer(HARVARD_GROUND_TRUTH, full_text)

        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📈 RTF (Real-Time Factor): {rtf:.3f}")
        print(f"📝 Transcription: '{full_text}'")
        print(f"🎯 WER (Word Error Rate): {wer:.1f}%")
        print(f"📊 Segments: {len(transcriptions)}")

        return {
            "engine": "Kyutai",
            "transcription": full_text,
            "processing_time": processing_time,
            "rtf": rtf,
            "wer": wer,
            "segments": len(transcriptions),
            "available": True
        }

    except Exception as e:
        print(f"❌ Kyutai STT failed: {e}")
        return {
            "engine": "Kyutai",
            "transcription": "",
            "processing_time": 0,
            "rtf": 0,
            "wer": 100,
            "segments": 0,
            "available": False,
            "error": str(e)
        }

async def test_whisper_mlx():
    """Test Whisper MLX engine"""
    print("\n🔬 Testing Whisper MLX Engine")
    print("=" * 50)

    try:
        from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel

        stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")

        # Test batch processing (Whisper MLX doesn't stream)
        start_time = time.time()
        transcriptions = []

        # Process full audio at once (batch mode)
        async for frame in stt.run_stt(audio_data):
            if hasattr(frame, 'text') and frame.text.strip():
                transcriptions.append(frame.text.strip())

        processing_time = time.time() - start_time
        rtf = processing_time / duration

        full_text = " ".join(transcriptions)
        wer = calculate_wer(HARVARD_GROUND_TRUTH, full_text)

        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📈 RTF (Real-Time Factor): {rtf:.3f}")
        print(f"📝 Transcription: '{full_text}'")
        print(f"🎯 WER (Word Error Rate): {wer:.1f}%")
        print(f"📊 Segments: {len(transcriptions)}")

        return {
            "engine": "Whisper MLX",
            "transcription": full_text,
            "processing_time": processing_time,
            "rtf": rtf,
            "wer": wer,
            "segments": len(transcriptions),
            "available": True
        }

    except Exception as e:
        print(f"❌ Whisper MLX failed: {e}")
        return {
            "engine": "Whisper MLX",
            "transcription": "",
            "processing_time": 0,
            "rtf": 0,
            "wer": 100,
            "segments": 0,
            "available": False,
            "error": str(e)
        }

async def test_parakeet_stt():
    """Test Parakeet STT engine"""
    print("\n🔬 Testing Parakeet STT Engine (Optimized)")
    print("=" * 50)

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        stt = ParakeetStreamingSTT(
            enable_vad=False,  # Disable to avoid Smart Turn conflicts
            volume_threshold=0.0001,
            chunk_duration=1.0,
            sentence_pause_threshold=1.2,
        )

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")

        # Test streaming performance
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

        processing_time = time.time() - start_time
        rtf = processing_time / duration

        full_text = " ".join(transcriptions)
        wer = calculate_wer(HARVARD_GROUND_TRUTH, full_text)

        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📈 RTF (Real-Time Factor): {rtf:.3f}")
        print(f"📝 Transcription: '{full_text}'")
        print(f"🎯 WER (Word Error Rate): {wer:.1f}%")
        print(f"📊 Segments: {len(transcriptions)}")

        return {
            "engine": "Parakeet",
            "transcription": full_text,
            "processing_time": processing_time,
            "rtf": rtf,
            "wer": wer,
            "segments": len(transcriptions),
            "available": True
        }

    except Exception as e:
        print(f"❌ Parakeet STT failed: {e}")
        return {
            "engine": "Parakeet",
            "transcription": "",
            "processing_time": 0,
            "rtf": 0,
            "wer": 100,
            "segments": 0,
            "available": False,
            "error": str(e)
        }

async def main():
    """Run comprehensive STT engine comparison"""
    print("🎯 STT Engine Comparison: Quality, Latency, and WER Analysis")
    print("=" * 70)
    print(f"📖 Ground Truth: '{HARVARD_GROUND_TRUTH}'")

    # Test all engines
    results = []

    # Test Kyutai
    kyutai_result = await test_kyutai_stt()
    results.append(kyutai_result)

    # Test Whisper MLX
    whisper_result = await test_whisper_mlx()
    results.append(whisper_result)

    # Test Parakeet
    parakeet_result = await test_parakeet_stt()
    results.append(parakeet_result)

    # Summary comparison
    print("\n🏆 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Engine':<15} {'Available':<10} {'RTF':<8} {'WER %':<8} {'Segments':<10} {'Quality'}")
    print("-" * 70)

    available_results = [r for r in results if r['available']]

    for result in results:
        if result['available']:
            quality = "🥇 Excellent" if result['wer'] < 5 else "🥈 Good" if result['wer'] < 15 else "🥉 Fair" if result['wer'] < 30 else "❌ Poor"
            print(f"{result['engine']:<15} {'✅':<10} {result['rtf']:<8.3f} {result['wer']:<8.1f} {result['segments']:<10} {quality}")
        else:
            print(f"{result['engine']:<15} {'❌':<10} {'N/A':<8} {'N/A':<8} {'N/A':<10} Not Available")

    if available_results:
        print("\n🎯 RECOMMENDATIONS:")

        # Best latency (lowest RTF)
        best_latency = min(available_results, key=lambda x: x['rtf'])
        print(f"🚀 Lowest Latency: {best_latency['engine']} (RTF: {best_latency['rtf']:.3f})")

        # Best quality (lowest WER)
        best_quality = min(available_results, key=lambda x: x['wer'])
        print(f"🎯 Best Quality: {best_quality['engine']} (WER: {best_quality['wer']:.1f}%)")

        # Best overall (balance of latency and quality)
        for result in available_results:
            result['score'] = (1 / (result['rtf'] + 0.001)) * (100 / (result['wer'] + 1))

        best_overall = max(available_results, key=lambda x: x['score'])
        print(f"⚖️  Best Overall: {best_overall['engine']} (RTF: {best_overall['rtf']:.3f}, WER: {best_overall['wer']:.1f}%)")

        print(f"\n💡 For real-time voice applications:")
        print(f"   • Choose {best_latency['engine']} for minimal latency")
        print(f"   • Choose {best_quality['engine']} for best transcription accuracy")
        print(f"   • Choose {best_overall['engine']} for balanced performance")

if __name__ == "__main__":
    asyncio.run(main())