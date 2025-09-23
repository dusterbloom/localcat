#!/usr/bin/env python
"""
Test STT with real audio file to verify transcription fixes
"""

import asyncio
import wave
import numpy as np
import sys
import os

# Ensure server root is importable
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for p in (_SERVER_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_wav_file(file_path):
    """Load WAV file and return audio data as bytes"""
    with wave.open(file_path, 'rb') as wav_file:
        # Verify format
        assert wav_file.getnchannels() == 1, "Audio must be mono"
        assert wav_file.getsampwidth() == 2, "Audio must be 16-bit"
        assert wav_file.getframerate() == 16000, f"Audio must be 16kHz, got {wav_file.getframerate()}"

        # Read all frames
        frames = wav_file.readframes(wav_file.getnframes())
        return frames


async def test_real_audio_transcription():
    """Test transcription with real audio file"""
    audio_file = '/Users/peppi/Dev/experiments/voice-agent-optimization/harvard_16k.wav'

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return

    print(f"Loading audio file: {audio_file}")

    # Load audio
    audio_bytes = load_wav_file(audio_file)
    audio_length_seconds = len(audio_bytes) / (16000 * 2)  # 16kHz, 16-bit
    print(".2f")

    # Initialize STT
    from core.stt.parakeet_streaming import ParakeetStreamingSTT

    print("Initializing ParakeetStreamingSTT...")
    stt = ParakeetStreamingSTT(
        enable_vad=False,  # Disable VAD for this test
        depth=3
    )

    # Simulate VAD start
    from pipecat.frames.frames import UserStartedSpeakingFrame
    await stt.process_frame(UserStartedSpeakingFrame())

    # Process audio in chunks (simulate real-time streaming)
    chunk_size = 16000  # 1 second at 16kHz
    bytes_per_sample = 2  # 16-bit
    chunk_bytes = chunk_size * bytes_per_sample

    all_frames = []
    total_audio_processed = 0

    print("Processing audio in chunks...")

    for i in range(0, len(audio_bytes), chunk_bytes):
        chunk = audio_bytes[i:i + chunk_bytes]
        if len(chunk) == 0:
            break

        total_audio_processed += len(chunk) / (16000 * 2)
        print(".2f")

        # Process chunk
        frames = []
        async for frame in stt.run_stt(chunk):
            frames.append(frame)
            if hasattr(frame, 'text'):
                print(f"  Frame: {type(frame).__name__} - '{frame.text}'")

        all_frames.extend(frames)

    # Simulate VAD stop
    from pipecat.frames.frames import UserStoppedSpeakingFrame
    await stt.process_frame(UserStoppedSpeakingFrame())

    # Analyze results
    interim_frames = [f for f in all_frames if hasattr(f, 'text') and 'Interim' in str(type(f))]
    final_frames = [f for f in all_frames if hasattr(f, 'text') and 'Transcription' in str(type(f))]

    print("\n📊 Results:")
    print(f"Total frames: {len(all_frames)}")
    print(f"Interim frames: {len(interim_frames)}")
    print(f"Final frames: {len(final_frames)}")

    # Collect all text
    all_texts = []
    for frame in all_frames:
        if hasattr(frame, 'text'):
            all_texts.append(frame.text)

    print(f"\nAll transcribed texts: {all_texts}")

    # Check for duplicates/repetition
    if len(all_texts) > 1:
        duplicates = []
        seen = set()
        for text in all_texts:
            if text in seen:
                duplicates.append(text)
            seen.add(text)

        if duplicates:
            print(f"❌ Found duplicate texts: {duplicates}")
        else:
            print("✅ No duplicate texts found")

    # Check for repetitive patterns
    combined_text = ' '.join(all_texts)
    if 'wanna talk about wanna talk about' in combined_text.lower():
        print("❌ Found repetitive 'wanna talk about' pattern")
    else:
        print("✅ No repetitive patterns detected")

    print("\n🎯 Test completed!")


if __name__ == "__main__":
    asyncio.run(test_real_audio_transcription())