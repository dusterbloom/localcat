#!/usr/bin/env python3
"""
Investigate Kokoro model parameters and test different configurations
to identify the source of sentence-ending artifacts.
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from tts_native_kokoro import NativeKokoroTTSService
from pipecat.frames.frames import TTSAudioRawFrame


def test_kokoro_direct():
    """Test Kokoro directly to see model behavior."""
    print("🔍 INVESTIGATING KOKORO MODEL PARAMETERS")
    print("=" * 60)

    try:
        from kokoro_onnx import Kokoro

        # Initialize Kokoro directly
        cache_dir = Path.home() / ".cache" / "kokoro"
        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"

        print(f"📂 Model path: {model_path}")
        print(f"📂 Voices path: {voices_path}")

        kokoro = Kokoro(
            model_path=str(model_path),
            voices_path=str(voices_path),
            espeak_config=None
        )

        # Test problematic texts
        test_cases = [
            {
                "name": "Simple sentence",
                "text": "Hello world."
            },
            {
                "name": "Question mark",
                "text": "How are you?"
            },
            {
                "name": "Exclamation",
                "text": "Great job!"
            },
            {
                "name": "Apostrophe word",
                "text": "Your dog's name."
            },
            {
                "name": "Question with apostrophe",
                "text": "What's your name?"
            },
            {
                "name": "Ending with space",
                "text": "Test with space "
            },
            {
                "name": "Multiple punctuation",
                "text": "Really?!"
            },
            {
                "name": "Original problematic chunk 1",
                "text": "Of course! Your dog's name is Po and Potola."
            },
            {
                "name": "Original problematic chunk 2",
                "text": "Is there anything else you'd like to tell me about him ?"
            }
        ]

        for test_case in test_cases:
            print(f"\n🎤 Testing: {test_case['name']}")
            print(f"   Text: '{test_case['text']}'")

            try:
                # Test with different voices to see if it's voice-specific
                for voice in ["af_heart", "af_bella", "af_sarah"]:
                    try:
                        audio_data, sample_rate = kokoro.create(
                            test_case['text'],
                            voice=voice,
                            speed=1.0
                        )

                        print(f"   Voice {voice}: {len(audio_data)} samples at {sample_rate}Hz")

                        # Check for potential issues in the audio data
                        if len(audio_data) > 100:
                            # Check ending samples for artifacts
                            ending_samples = audio_data[-100:]  # Last 100 samples
                            max_ending = max(abs(s) for s in ending_samples)
                            avg_ending = sum(abs(s) for s in ending_samples) / len(ending_samples)

                            # Check overall audio
                            max_overall = max(abs(s) for s in audio_data)
                            avg_overall = sum(abs(s) for s in audio_data) / len(audio_data)

                            print(f"     Ending max: {max_ending}, avg: {avg_ending:.1f}")
                            print(f"     Overall max: {max_overall}, avg: {avg_overall:.1f}")

                            # Flag potential issues
                            if max_ending > max_overall * 0.8:
                                print(f"     ⚠️  HIGH ENDING AMPLITUDE - possible artifact!")

                            # Check for abrupt cutoffs
                            last_10 = audio_data[-10:]
                            if len(last_10) > 1:
                                last_change = abs(last_10[-1] - last_10[-2])
                                if last_change > avg_overall * 2:
                                    print(f"     ⚠️  ABRUPT ENDING CHANGE - possible cutoff!")

                    except Exception as voice_error:
                        print(f"   Voice {voice}: FAILED - {voice_error}")

            except Exception as e:
                print(f"   ❌ FAILED: {e}")

    except ImportError:
        print("❌ kokoro-onnx not available")
    except Exception as e:
        print(f"❌ Test failed: {e}")


async def test_tts_service_params():
    """Test the TTS service with different parameters."""
    print(f"\n🔧 TESTING TTS SERVICE PARAMETERS")
    print("=" * 40)

    # Test different speed settings
    speeds = [0.8, 1.0, 1.2]
    voices = ["af_heart", "af_bella"]

    problematic_text = "Your dog's name is Po and Potola."

    for voice in voices:
        for speed in speeds:
            print(f"\n🎛️  Testing voice={voice}, speed={speed}")

            try:
                tts = NativeKokoroTTSService(
                    voice=voice,
                    speed=speed,
                    sample_rate=24000
                )

                audio_bytes = b""
                async for frame in tts.run_tts(problematic_text):
                    if isinstance(frame, TTSAudioRawFrame):
                        audio_bytes += frame.audio

                if audio_bytes:
                    # Convert to analysis format
                    import numpy as np
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                    print(f"   Generated {len(audio_np)} samples")

                    # Analyze ending
                    if len(audio_np) > 100:
                        ending = audio_np[-100:]
                        ending_max = np.max(np.abs(ending))
                        overall_max = np.max(np.abs(audio_np))

                        print(f"   Ending max: {ending_max}, Overall max: {overall_max}")

                        if ending_max > overall_max * 0.7:
                            print(f"   ⚠️  Potential ending artifact detected!")

            except Exception as e:
                print(f"   ❌ Failed: {e}")


def analyze_specific_characters():
    """Test specific characters that might cause issues."""
    print(f"\n🔤 TESTING SPECIFIC CHARACTERS")
    print("=" * 40)

    # Characters that might cause issues
    problem_chars = [
        "'",  # Standard apostrophe
        "'",  # Smart apostrophe
        "?",  # Question mark
        "!",  # Exclamation
        " ",  # Space at end
        ".",  # Period
        ",",  # Comma
    ]

    base_text = "Test"

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

        for char in problem_chars:
            test_text = base_text + char
            print(f"\n🔤 Testing character: '{char}' (ord: {ord(char)})")
            print(f"   Text: '{test_text}'")

            try:
                audio_data, sample_rate = kokoro.create(
                    test_text,
                    voice="af_heart",
                    speed=1.0
                )

                # Analyze the ending
                if len(audio_data) > 50:
                    ending = audio_data[-50:]
                    ending_max = max(abs(s) for s in ending)
                    overall_max = max(abs(s) for s in audio_data)

                    print(f"   Samples: {len(audio_data)}")
                    print(f"   Ending max: {ending_max}, Overall max: {overall_max}")

                    if ending_max > overall_max * 0.7:
                        print(f"   ⚠️  Character '{char}' may cause ending artifacts!")

            except Exception as e:
                print(f"   ❌ Failed for '{char}': {e}")

    except Exception as e:
        print(f"❌ Character test failed: {e}")


async def main():
    """Main investigation function."""
    test_kokoro_direct()
    await test_tts_service_params()
    analyze_specific_characters()


if __name__ == "__main__":
    asyncio.run(main())