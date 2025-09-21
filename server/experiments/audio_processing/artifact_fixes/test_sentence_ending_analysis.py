#!/usr/bin/env python3
"""
Test to record and analyze the weird sound issue at sentence endings in Kokoro TTS.
This test will generate audio for the problematic text and save it for analysis.
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
    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    print(f"✅ Audio saved to: {filepath}")
    print(f"   Duration: {len(audio_np) / sample_rate:.2f}s")
    print(f"   Samples: {len(audio_np)}")


def analyze_audio_chunks(chunks: list, sample_rate: int):
    """Analyze audio chunks for potential artifacts at endings."""
    print("\n🔍 ANALYZING AUDIO CHUNKS:")

    for i, (chunk_text, chunk_audio) in enumerate(chunks):
        if not chunk_audio:
            continue

        audio_np = np.frombuffer(chunk_audio, dtype=np.int16)
        duration = len(audio_np) / sample_rate

        # Analyze ending of the chunk (last 200ms or 10% of duration, whichever is smaller)
        ending_samples = min(int(0.2 * sample_rate), len(audio_np) // 10)
        if ending_samples > 0:
            ending_audio = audio_np[-ending_samples:]

            # Calculate RMS energy in the ending
            ending_rms = np.sqrt(np.mean(ending_audio.astype(float) ** 2))

            # Calculate overall RMS for comparison
            overall_rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))

            # Check for sudden changes or artifacts
            if len(ending_audio) > 100:
                # Look for abrupt amplitude changes in the ending
                ending_diff = np.abs(np.diff(ending_audio.astype(float)))
                max_ending_change = np.max(ending_diff)
                mean_ending_change = np.mean(ending_diff)

                print(f"\n   Chunk {i+1}: '{chunk_text[:40]}...'")
                print(f"     Duration: {duration:.3f}s")
                print(f"     Overall RMS: {overall_rms:.1f}")
                print(f"     Ending RMS: {ending_rms:.1f}")
                print(f"     Ending max change: {max_ending_change:.1f}")
                print(f"     Ending mean change: {mean_ending_change:.1f}")

                # Flag potential artifacts
                if ending_rms > overall_rms * 1.5:
                    print(f"     ⚠️  HIGH ENDING ENERGY - possible artifact!")
                elif ending_rms < overall_rms * 0.1:
                    print(f"     ⚠️  VERY LOW ENDING ENERGY - possible cutoff!")

                if max_ending_change > overall_rms * 3:
                    print(f"     ⚠️  SUDDEN AMPLITUDE CHANGE - possible click/pop!")


async def test_sentence_ending_issue():
    """Test the specific problematic text that causes weird sounds at sentence endings."""

    print("🧪 TESTING SENTENCE ENDING ARTIFACTS")
    print("=" * 60)

    # The exact problematic text from the user
    problematic_text = "Of course! Your dog's name is Po and Potola. Is there anything else you'd like to tell me about him ?"

    print(f"🎯 Testing problematic text:")
    print(f"   '{problematic_text}'")

    try:
        # Initialize TTS service
        tts = NativeKokoroTTSService(
            voice="af_heart",  # Using the same voice as mentioned in the files
            speed=1.0,
            sample_rate=24000
        )

        print("✅ TTS initialization successful")

        # Create output directory
        output_dir = Path("audio_analysis")
        output_dir.mkdir(exist_ok=True)

        # Test the full text
        print(f"\n🎤 Generating audio for full text...")

        start_time = time.time()
        audio_chunks = []
        current_chunk_audio = b""
        current_chunk_text = ""
        chunk_count = 0

        frame_generator = tts.run_tts(problematic_text)

        async for frame in frame_generator:
            if isinstance(frame, TTSStartedFrame):
                print("   📡 TTS Started")

            elif isinstance(frame, TTSAudioRawFrame):
                current_chunk_audio += frame.audio
                chunk_count += 1

                if chunk_count == 1:
                    ttfb = (time.time() - start_time) * 1000
                    print(f"   🚀 TTFB: {ttfb:.1f}ms")

            elif isinstance(frame, TTSStoppedFrame):
                print("   🔚 TTS Stopped")

                # Save the complete audio
                if current_chunk_audio:
                    audio_chunks.append((problematic_text, current_chunk_audio))

        # Save full audio file
        if current_chunk_audio:
            full_audio_path = output_dir / "problematic_full.wav"
            save_audio_to_wav(current_chunk_audio, 24000, str(full_audio_path))

            # Also save chunks separately to see where the issue occurs
            from tools.text_formatter import split_text_for_kokoro_streaming
            text_chunks = split_text_for_kokoro_streaming(problematic_text, min_length=50, max_length=120)

            print(f"\n📝 Text was split into {len(text_chunks)} chunks:")
            for i, chunk in enumerate(text_chunks):
                print(f"   Chunk {i+1}: '{chunk}'")

            # Generate audio for each chunk separately to isolate the issue
            chunk_audio_data = []

            for i, chunk_text in enumerate(text_chunks):
                print(f"\n🎤 Generating audio for chunk {i+1}: '{chunk_text}'")

                chunk_audio = b""
                frame_generator = tts.run_tts(chunk_text)

                async for frame in frame_generator:
                    if isinstance(frame, TTSAudioRawFrame):
                        chunk_audio += frame.audio

                if chunk_audio:
                    chunk_path = output_dir / f"chunk_{i+1}.wav"
                    save_audio_to_wav(chunk_audio, 24000, str(chunk_path))
                    chunk_audio_data.append((chunk_text, chunk_audio))

            # Analyze all chunks for artifacts
            analyze_audio_chunks(chunk_audio_data, 24000)

        total_time = time.time() - start_time
        print(f"\n📊 GENERATION COMPLETE")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Audio frames received: {chunk_count}")

        # Print analysis summary
        print(f"\n🔍 ANALYSIS SUMMARY:")
        print(f"   Output directory: {output_dir.absolute()}")
        print(f"   Files generated:")
        print(f"     - problematic_full.wav (complete audio)")
        for i in range(len(text_chunks)):
            print(f"     - chunk_{i+1}.wav (individual chunk)")

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Listen to the generated WAV files to identify the weird sound")
        print(f"   2. Check if the artifact occurs in individual chunks or only when combined")
        print(f"   3. Examine the RMS energy analysis above for sudden changes")
        print(f"   4. Look for patterns in chunks that end with specific punctuation")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_sentence_ending_issue())
    sys.exit(0 if success else 1)