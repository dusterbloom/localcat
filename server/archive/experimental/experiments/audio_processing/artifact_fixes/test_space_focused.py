#!/usr/bin/env python3
"""
Focused test of space before punctuation using direct Kokoro calls
to eliminate chunking variables and test the core hypothesis.
"""

import asyncio
import sys
import os
import wave
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(__file__))


def save_audio_to_wav(audio_data, sample_rate: int, filepath: str):
    """Save audio data to a WAV file."""
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Convert to int16 if needed
        if hasattr(audio_data, 'dtype'):
            if audio_data.dtype != np.int16:
                audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data
        else:
            audio_int16 = audio_data

        wav_file.writeframes(audio_int16.tobytes())

    print(f"✅ Audio saved to: {filepath}")


def analyze_ending_detailed(audio_data, sample_rate: int, label: str):
    """Detailed analysis of audio ending."""
    if hasattr(audio_data, 'dtype'):
        audio_np = audio_data
    else:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

    print(f"\n🔍 {label} Analysis:")
    print(f"   Duration: {len(audio_np) / sample_rate:.3f}s")
    print(f"   Samples: {len(audio_np)}")

    # Look at the actual ending values
    if len(audio_np) > 50:
        last_50 = audio_np[-50:]
        print(f"   Last 50 sample values: {last_50}")

        # Check for abrupt ending
        last_value = abs(last_50[-1])
        second_last = abs(last_50[-2]) if len(last_50) > 1 else 0

        print(f"   Final sample: {last_50[-1]}")
        print(f"   Penultimate sample: {last_50[-2] if len(last_50) > 1 else 'N/A'}")

        if last_value > 100:
            print(f"   ⚠️  NON-ZERO ENDING: Audio ends abruptly at {last_value}")
        else:
            print(f"   ✅ Clean ending: Final sample is {last_value}")

        # Check for sudden jumps in the ending
        ending_diffs = np.abs(np.diff(last_50.astype(float)))
        max_jump = np.max(ending_diffs) if len(ending_diffs) > 0 else 0
        print(f"   Max ending jump: {max_jump}")

        if max_jump > 1000:
            print(f"   ⚠️  LARGE ENDING JUMP detected!")


def test_direct_kokoro():
    """Test Kokoro directly with space vs no-space."""
    print("🧪 DIRECT KOKORO TEST: SPACE BEFORE PUNCTUATION")
    print("=" * 60)

    try:
        from kokoro_onnx import Kokoro

        cache_dir = Path.home() / ".cache" / "kokoro"
        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"

        kokoro = Kokoro(
            model_path=str(model_path),
            voices_path=str(voices_path),
            espeak_config=None
        )

        # Create output directory
        output_dir = Path("direct_space_test")
        output_dir.mkdir(exist_ok=True)

        # Test cases focusing on the space issue
        test_cases = [
            {
                "name": "With space before ?",
                "text": "Is there anything else you'd like to tell me about him ?",
                "filename": "with_space.wav"
            },
            {
                "name": "Without space before ?",
                "text": "Is there anything else you'd like to tell me about him?",
                "filename": "without_space.wav"
            },
            {
                "name": "Simple with space",
                "text": "How are you ?",
                "filename": "simple_with_space.wav"
            },
            {
                "name": "Simple without space",
                "text": "How are you?",
                "filename": "simple_without_space.wav"
            },
            {
                "name": "Exclamation with space",
                "text": "Great job !",
                "filename": "exclaim_with_space.wav"
            },
            {
                "name": "Exclamation without space",
                "text": "Great job!",
                "filename": "exclaim_without_space.wav"
            }
        ]

        for test_case in test_cases:
            print(f"\n🎤 Testing: {test_case['name']}")
            print(f"   Text: '{test_case['text']}'")

            try:
                # Generate audio with Kokoro
                audio_data, sample_rate = kokoro.create(
                    test_case['text'],
                    voice="af_heart",
                    speed=1.0
                )

                # Save and analyze
                filepath = output_dir / test_case['filename']
                save_audio_to_wav(audio_data, sample_rate, str(filepath))
                analyze_ending_detailed(audio_data, sample_rate, test_case['name'])

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        print(f"\n📊 CONCLUSION:")
        print(f"   If space before punctuation is the issue, you should see:")
        print(f"   - Different ending patterns between 'with space' vs 'without space' versions")
        print(f"   - Abrupt endings or large jumps in 'with space' versions")
        print(f"   - Clean endings in 'without space' versions")

        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Listen to the generated files in {output_dir}/")
        print(f"   2. Compare the 'Last 50 sample values' between versions")
        print(f"   3. Look for patterns in the final sample values")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_direct_kokoro()