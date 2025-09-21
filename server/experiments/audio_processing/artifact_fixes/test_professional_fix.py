#!/usr/bin/env python3
"""
Comprehensive test of the professional Kokoro TTS fix.
Compares original vs. professional implementation to validate artifact elimination.
"""

import asyncio
import time
import sys
import os
import wave
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from tts_native_kokoro import NativeKokoroTTSService
from tts_kokoro_professional import ProfessionalKokoroTTSService
from pipecat.frames.frames import TTSAudioRawFrame


def save_audio_to_wav(audio_data: bytes, sample_rate: int, filepath: str):
    """Save raw audio bytes to a WAV file."""
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return len(audio_np) / sample_rate  # duration


def analyze_artifacts(audio_data: bytes, sample_rate: int) -> dict:
    """Analyze audio for artifacts and return comprehensive metrics."""
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    if len(audio_np) < 100:
        return {"error": "audio_too_short"}

    # Analyze ending region (last 400ms)
    ending_samples = min(int(0.4 * sample_rate), len(audio_np) // 2)
    ending_audio = audio_np[-ending_samples:]

    # Calculate artifact metrics
    audio_diff = np.abs(np.diff(ending_audio.astype(float)))
    change_threshold = np.mean(audio_diff) + 3 * np.std(audio_diff)
    artifact_count = np.sum(audio_diff > change_threshold)

    # RMS analysis
    ending_rms = np.sqrt(np.mean((ending_audio.astype(float) / 32768.0) ** 2))
    overall_rms = np.sqrt(np.mean((audio_np.astype(float) / 32768.0) ** 2))

    # Ending quality analysis
    final_sample = abs(audio_np[-1])
    max_ending_jump = np.max(audio_diff) if len(audio_diff) > 0 else 0

    return {
        "duration": len(audio_np) / sample_rate,
        "artifact_count": int(artifact_count),
        "ending_rms": float(ending_rms),
        "overall_rms": float(overall_rms),
        "rms_ratio": float(ending_rms / overall_rms) if overall_rms > 0 else 0,
        "final_sample": int(final_sample),
        "max_ending_jump": float(max_ending_jump),
        "peak_amplitude": int(np.max(np.abs(audio_np))),
    }


async def test_implementation(tts_service, name: str, text: str, output_path: Path) -> dict:
    """Test a TTS implementation and return metrics."""
    print(f"\n🎤 Testing {name}...")

    start_time = time.time()
    audio_data = b""
    frame_count = 0

    try:
        async for frame in tts_service.run_tts(text):
            if isinstance(frame, TTSAudioRawFrame):
                audio_data += frame.audio
                frame_count += 1

        generation_time = time.time() - start_time

        if audio_data:
            # Save audio
            duration = save_audio_to_wav(audio_data, 24000, str(output_path))
            print(f"   ✅ Generated: {duration:.2f}s audio in {generation_time:.2f}s")

            # Analyze quality
            metrics = analyze_artifacts(audio_data, 24000)
            metrics.update({
                "generation_time": generation_time,
                "frame_count": frame_count,
                "implementation": name
            })

            return metrics
        else:
            return {"error": "no_audio_generated", "implementation": name}

    except Exception as e:
        print(f"   ❌ {name} failed: {e}")
        return {"error": str(e), "implementation": name}


async def comprehensive_comparison_test():
    """Run comprehensive comparison between original and professional implementations."""

    print("🔬 COMPREHENSIVE KOKORO TTS ARTIFACT FIX VALIDATION")
    print("=" * 70)

    # Test cases with different complexity levels
    test_cases = [
        {
            "name": "Original problematic text",
            "text": "Of course! Your dog's name is Po and Potola. Is there anything else you'd like to tell me about him ?",
            "expected_improvement": "High"
        },
        {
            "name": "Simple question",
            "text": "How are you today?",
            "expected_improvement": "Medium"
        },
        {
            "name": "Long sentence",
            "text": "This is a much longer sentence that should really test how well the professional audio processing handles extended content with proper fade-out and artifact elimination.",
            "expected_improvement": "High"
        },
        {
            "name": "Multiple punctuation",
            "text": "Really?! That's amazing! Are you sure?",
            "expected_improvement": "High"
        }
    ]

    # Create output directory
    output_dir = Path("professional_fix_validation")
    output_dir.mkdir(exist_ok=True)

    overall_results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"Text: '{test_case['text']}'")
        print(f"Expected improvement: {test_case['expected_improvement']}")
        print(f"{'='*50}")

        try:
            # Initialize both services
            original_tts = NativeKokoroTTSService(
                voice="af_heart",
                speed=1.0,
                sample_rate=24000
            )

            professional_tts = ProfessionalKokoroTTSService(
                voice="af_heart",
                speed=1.0,
                sample_rate=24000,
                fade_duration_ms=50.0,
                target_peak_db=-3.0,
                enable_quality_logging=True
            )

            # Test original implementation
            original_path = output_dir / f"case_{i}_original.wav"
            original_metrics = await test_implementation(
                original_tts, "ORIGINAL", test_case['text'], original_path
            )

            # Test professional implementation
            professional_path = output_dir / f"case_{i}_professional.wav"
            professional_metrics = await test_implementation(
                professional_tts, "PROFESSIONAL", test_case['text'], professional_path
            )

            # Compare results
            if "error" not in original_metrics and "error" not in professional_metrics:
                artifact_reduction = original_metrics["artifact_count"] - professional_metrics["artifact_count"]
                improvement_pct = (artifact_reduction / original_metrics["artifact_count"] * 100) if original_metrics["artifact_count"] > 0 else 0

                print(f"\n📊 COMPARISON RESULTS:")
                print(f"   Original artifacts: {original_metrics['artifact_count']}")
                print(f"   Professional artifacts: {professional_metrics['artifact_count']}")
                print(f"   Artifact reduction: {artifact_reduction} ({improvement_pct:.1f}%)")
                print(f"   Original final sample: {original_metrics['final_sample']}")
                print(f"   Professional final sample: {professional_metrics['final_sample']}")
                print(f"   Original RMS ratio: {original_metrics['rms_ratio']:.3f}")
                print(f"   Professional RMS ratio: {professional_metrics['rms_ratio']:.3f}")

                # Quality assessment
                if artifact_reduction > 50:
                    print(f"   ✅ EXCELLENT IMPROVEMENT: Major artifact reduction!")
                elif artifact_reduction > 10:
                    print(f"   ✅ GOOD IMPROVEMENT: Significant artifact reduction")
                elif artifact_reduction > 0:
                    print(f"   ⚠️  MINOR IMPROVEMENT: Some artifact reduction")
                else:
                    print(f"   ❌ NO IMPROVEMENT: Artifacts not reduced")

                # Performance comparison
                time_diff = professional_metrics["generation_time"] - original_metrics["generation_time"]
                if time_diff < 0.1:
                    print(f"   ✅ PERFORMANCE: No significant overhead ({time_diff*1000:.1f}ms)")
                else:
                    print(f"   ⚠️  PERFORMANCE: Additional processing time ({time_diff*1000:.1f}ms)")

                overall_results.append({
                    "case": test_case['name'],
                    "original": original_metrics,
                    "professional": professional_metrics,
                    "improvement": artifact_reduction,
                    "improvement_pct": improvement_pct
                })

        except Exception as e:
            print(f"❌ Test case {i} failed: {e}")

    # Overall summary
    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION SUMMARY")
    print(f"{'='*70}")

    if overall_results:
        total_cases = len(overall_results)
        successful_improvements = sum(1 for r in overall_results if r["improvement"] > 0)
        major_improvements = sum(1 for r in overall_results if r["improvement"] > 50)
        avg_improvement = np.mean([r["improvement_pct"] for r in overall_results])

        print(f"📈 IMPROVEMENT STATISTICS:")
        print(f"   Test cases: {total_cases}")
        print(f"   Cases with improvement: {successful_improvements}/{total_cases}")
        print(f"   Major improvements (>50 artifacts): {major_improvements}/{total_cases}")
        print(f"   Average artifact reduction: {avg_improvement:.1f}%")

        print(f"\n🎯 VERDICT:")
        if successful_improvements == total_cases and avg_improvement > 50:
            print(f"   ✅ EXCELLENT: Professional fix successfully eliminates artifacts!")
        elif successful_improvements >= total_cases * 0.8:
            print(f"   ✅ GOOD: Professional fix shows significant improvements")
        elif successful_improvements > 0:
            print(f"   ⚠️  PARTIAL: Professional fix shows some improvements")
        else:
            print(f"   ❌ FAILED: Professional fix does not improve artifacts")

        print(f"\n📁 AUDIO FILES:")
        print(f"   Location: {output_dir.absolute()}")
        print(f"   Compare original vs professional versions")
        print(f"   Listen for cleaner sentence endings in professional versions")

    return overall_results


if __name__ == "__main__":
    results = asyncio.run(comprehensive_comparison_test())
    print(f"\n🏁 Validation complete. Check the generated audio files to confirm the improvements!")