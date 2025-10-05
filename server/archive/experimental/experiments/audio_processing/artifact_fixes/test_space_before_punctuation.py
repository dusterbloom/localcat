#!/usr/bin/env python3
"""
Test if the space before punctuation is causing the weird sound artifacts.
Compare original text vs. fixed text without space before question mark.
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
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame


def save_audio_to_wav(audio_data: bytes, sample_rate: int, filepath: str):
    """Save raw audio bytes to a WAV file."""
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    print(f"✅ Audio saved to: {filepath}")
    print(f"   Duration: {len(audio_np) / sample_rate:.2f}s")
    print(f"   Samples: {len(audio_np)}")


def analyze_ending_artifacts(audio_data: bytes, sample_rate: int, label: str):
    """Analyze audio for ending artifacts."""
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    if len(audio_np) < 100:
        print(f"   {label}: Audio too short for analysis")
        return 0

    # Analyze last 400ms (where artifacts typically occur)
    ending_samples = min(int(0.4 * sample_rate), len(audio_np) // 2)
    ending_audio = audio_np[-ending_samples:]

    # Calculate sudden amplitude changes
    audio_diff = np.abs(np.diff(ending_audio.astype(float)))

    # Threshold for detecting artifacts (3x standard deviation)
    change_threshold = np.mean(audio_diff) + 3 * np.std(audio_diff)
    artifact_count = np.sum(audio_diff > change_threshold)

    # Calculate RMS values
    ending_rms = np.sqrt(np.mean((ending_audio.astype(float) / 32768.0) ** 2))
    overall_rms = np.sqrt(np.mean((audio_np.astype(float) / 32768.0) ** 2))

    print(f"   {label}:")
    print(f"     Duration: {len(audio_np) / sample_rate:.3f}s")
    print(f"     Artifacts in ending: {artifact_count}")
    print(f"     Ending RMS: {ending_rms:.4f}")
    print(f"     Overall RMS: {overall_rms:.4f}")
    print(f"     RMS ratio: {ending_rms/overall_rms:.2f}")

    if artifact_count > 50:
        print(f"     ⚠️  HIGH ARTIFACT COUNT - likely problematic!")
    elif artifact_count > 10:
        print(f"     ⚠️  Moderate artifacts detected")
    else:
        print(f"     ✅ Low artifact count - likely clean")

    return artifact_count


async def test_space_hypothesis():
    """Test if space before punctuation causes artifacts."""

    print("🧪 TESTING SPACE BEFORE PUNCTUATION HYPOTHESIS")
    print("=" * 60)

    # Original problematic text (with space before ?)
    original_text = "Of course! Your dog's name is Po and Potola. Is there anything else you'd like to tell me about him ?"

    # Fixed text (without space before ?)
    fixed_text = "Of course! Your dog's name is Po and Potola. Is there anything else you'd like to tell me about him?"

    print(f"📝 Original text: '{original_text}'")
    print(f"📝 Fixed text:    '{fixed_text}'")
    print(f"🔍 Difference: Removed space before final '?'")

    # Create output directory
    output_dir = Path("space_test_analysis")
    output_dir.mkdir(exist_ok=True)

    try:
        # Initialize TTS service
        tts = NativeKokoroTTSService(
            voice="af_heart",
            speed=1.0,
            sample_rate=24000
        )

        print("✅ TTS initialization successful")

        # Test original text
        print(f"\n🎤 Testing ORIGINAL text (with space before ?)...")

        original_audio = b""
        start_time = time.time()

        async for frame in tts.run_tts(original_text):
            if isinstance(frame, TTSAudioRawFrame):
                original_audio += frame.audio

        original_time = time.time() - start_time

        if original_audio:
            original_path = output_dir / "original_with_space.wav"
            save_audio_to_wav(original_audio, 24000, str(original_path))
            original_artifacts = analyze_ending_artifacts(original_audio, 24000, "ORIGINAL")

        # Test fixed text
        print(f"\n🎤 Testing FIXED text (without space before ?)...")

        fixed_audio = b""
        start_time = time.time()

        async for frame in tts.run_tts(fixed_text):
            if isinstance(frame, TTSAudioRawFrame):
                fixed_audio += frame.audio

        fixed_time = time.time() - start_time

        if fixed_audio:
            fixed_path = output_dir / "fixed_no_space.wav"
            save_audio_to_wav(fixed_audio, 24000, str(fixed_path))
            fixed_artifacts = analyze_ending_artifacts(fixed_audio, 24000, "FIXED")

        # Test just the problematic ending part
        print(f"\n🎤 Testing ISOLATED endings...")

        problematic_ending = "tell me about him ?"
        clean_ending = "tell me about him?"

        # Test problematic ending
        problematic_audio = b""
        async for frame in tts.run_tts(problematic_ending):
            if isinstance(frame, TTSAudioRawFrame):
                problematic_audio += frame.audio

        # Test clean ending
        clean_audio = b""
        async for frame in tts.run_tts(clean_ending):
            if isinstance(frame, TTSAudioRawFrame):
                clean_audio += frame.audio

        if problematic_audio:
            prob_path = output_dir / "problematic_ending.wav"
            save_audio_to_wav(problematic_audio, 24000, str(prob_path))
            prob_artifacts = analyze_ending_artifacts(problematic_audio, 24000, "PROBLEMATIC ENDING")

        if clean_audio:
            clean_path = output_dir / "clean_ending.wav"
            save_audio_to_wav(clean_audio, 24000, str(clean_path))
            clean_artifacts = analyze_ending_artifacts(clean_audio, 24000, "CLEAN ENDING")

        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Original text artifacts: {original_artifacts}")
        print(f"   Fixed text artifacts: {fixed_artifacts}")
        print(f"   Problematic ending: {prob_artifacts}")
        print(f"   Clean ending: {clean_artifacts}")

        improvement = original_artifacts - fixed_artifacts
        ending_improvement = prob_artifacts - clean_artifacts

        print(f"\n🎯 HYPOTHESIS TEST RESULTS:")
        if improvement > 50:
            print(f"   ✅ HYPOTHESIS CONFIRMED! Removing space reduced artifacts by {improvement}")
        elif improvement > 0:
            print(f"   ⚠️  PARTIAL IMPROVEMENT: Reduced artifacts by {improvement}")
        elif improvement == 0:
            print(f"   ❓ NO CHANGE: Space doesn't affect artifact count")
        else:
            print(f"   ❌ HYPOTHESIS REJECTED: Fixed version has {abs(improvement)} more artifacts")

        if ending_improvement > 20:
            print(f"   ✅ ISOLATED TEST CONFIRMS: Space before ? causes {ending_improvement} extra artifacts")

        print(f"\n💡 RECOMMENDATIONS:")
        if improvement > 0 or ending_improvement > 0:
            print(f"   1. ✅ Remove spaces before punctuation marks (?, !, .)")
            print(f"   2. ✅ Update text preprocessing to normalize punctuation")
            print(f"   3. ✅ Test with other space-before-punctuation cases")
        else:
            print(f"   1. Space before punctuation is NOT the primary cause")
            print(f"   2. Continue investigating audio processing pipeline")
            print(f"   3. Focus on format conversion and frame assembly")

        return improvement > 0

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_space_hypothesis())
    print(f"\n🏁 Test completed. Space hypothesis: {'CONFIRMED' if success else 'REJECTED'}")