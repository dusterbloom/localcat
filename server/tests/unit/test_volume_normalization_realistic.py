#!/usr/bin/env python
"""
Test volume normalization with realistic speech-like audio
"""

import numpy as np
import sys
import os
import asyncio

# Ensure server root is importable
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for p in (_SERVER_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)


def generate_speech_like_audio(duration_seconds=2.0, sample_rate=16000, volume_factor=1.0):
    """Generate synthetic speech-like audio with varying volume"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))

    # Create speech-like signal with multiple frequency components
    signal = (
        0.3 * np.sin(2 * np.pi * 300 * t) +  # Fundamental frequency ~300Hz
        0.2 * np.sin(2 * np.pi * 600 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 900 * t) +  # Second harmonic
        0.05 * np.random.normal(0, 0.1, len(t))  # Some noise
    )

    # Apply volume factor and convert to int16 bytes
    signal = signal * volume_factor
    signal = np.clip(signal, -1, 1)  # Ensure within [-1, 1]

    # Convert to int16 bytes (what the STT service expects)
    audio_int16 = (signal * 32767).astype(np.int16)
    return audio_int16.tobytes()


async def test_volume_normalization_with_speech():
    """Test volume normalization with realistic speech-like audio"""
    from core.stt.parakeet_streaming import ParakeetStreamingSTT

    # Test different volume levels
    volume_levels = [0.01, 0.05, 0.1, 0.2, 0.5]  # Very quiet to loud

    print("Testing volume normalization with synthetic speech-like audio...")

    for volume in volume_levels:
        print(f"\n--- Testing volume factor: {volume} ---")

        # Generate test audio
        test_audio = generate_speech_like_audio(volume_factor=volume)

        # Convert to numpy for RMS calculation
        audio_array = np.frombuffer(test_audio, dtype=np.int16).astype(np.float32) / 32768.0
        original_rms = np.sqrt(np.mean(audio_array**2))
        print(".4f")

        # Create STT instance and test normalization
        stt = ParakeetStreamingSTT.__new__(ParakeetStreamingSTT)  # Create without __init__
        normalized = stt._normalize_audio(audio_array)
        normalized_rms = np.sqrt(np.mean(normalized**2))
        print(".4f")

        # Verify normalization worked
        if original_rms < 0.01:  # Very quiet
            assert normalized_rms > original_rms, f"Quiet audio should be amplified: {original_rms} -> {normalized_rms}"
        elif original_rms > 0.2:  # Very loud
            assert normalized_rms < original_rms, f"Loud audio should be attenuated: {original_rms} -> {normalized_rms}"

        # Check that normalized audio is within reasonable bounds
        assert 0.05 <= normalized_rms <= 0.15, f"RMS should be normalized to ~0.1: got {normalized_rms}"
        assert np.max(np.abs(normalized)) <= 1.0, "Audio should not clip"

        print("✅ Volume normalization working correctly")

    print("\n🎉 All volume normalization tests with speech-like audio passed!")


async def test_stt_pipeline_with_normalization():
    """Test the full STT pipeline with volume normalization"""
    print("\n--- Testing full STT pipeline with volume normalization ---")

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        # Create STT service with volume normalization
        stt = ParakeetStreamingSTT(
            enable_vad=False,  # Disable VAD for this test
            depth=3
        )

        # Generate quiet speech-like audio
        quiet_audio = generate_speech_like_audio(volume_factor=0.02)  # Very quiet

        print("Processing quiet speech-like audio through STT pipeline...")

        # Process through STT pipeline
        frames = []
        async for frame in stt.run_stt(quiet_audio):
            frames.append(frame)
            print(f"Received frame: {type(frame).__name__}")

        print(f"✅ STT pipeline processed {len(frames)} frames successfully")

        # Test with external VAD simulation
        print("\nSimulating VAD start/stop with quiet audio...")

        # Simulate VAD start
        from pipecat.frames.frames import UserStartedSpeakingFrame
        await stt.process_frame(UserStartedSpeakingFrame())

        # Process audio
        frames_vad = []
        async for frame in stt.run_stt(quiet_audio):
            frames_vad.append(frame)

        # Simulate VAD stop
        from pipecat.frames.frames import UserStoppedSpeakingFrame
        await stt.process_frame(UserStoppedSpeakingFrame())

        print(f"✅ VAD-gated STT processed {len(frames_vad)} frames successfully")

    except Exception as e:
        print(f"❌ STT pipeline test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_volume_normalization_with_speech())
    asyncio.run(test_stt_pipeline_with_normalization())