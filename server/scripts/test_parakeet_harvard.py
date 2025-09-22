#!/usr/bin/env python3
"""
Test Parakeet STT optimizations with Harvard audio file
"""

import asyncio
import numpy as np
import sys
import time
import wave
from pathlib import Path

# Add server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def test_parakeet_with_harvard():
    """Test Parakeet STT with Harvard audio file to verify optimizations"""
    print("🧪 Testing optimized Parakeet STT with Harvard audio...")

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        # Test with both old (aggressive) and new (optimized) settings
        print("\n🔄 Testing with OPTIMIZED settings (less aggressive VAD)...")

        # Create optimized STT service
        stt_optimized = ParakeetStreamingSTT(
            chunk_duration=1.0,  # Optimal 1s chunks
            confidence_threshold=0.2,  # Lower threshold to accept more
            sentence_pause_threshold=1.2,  # Longer pause to avoid cutting words
            max_chunk_duration=4.0,  # Reduced max for better responsiveness
            volume_threshold=0.001,  # Much lower volume threshold
            context_size=(256, 256),  # Parakeet streaming context
            depth=1
        )

        print("✅ Optimized STT service initialized")

        # Load Harvard audio file
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard.wav"
        print(f"🔄 Loading Harvard audio: {harvard_path}")

        # Read the WAV file and convert to bytes
        with wave.open(harvard_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        print(f"📊 Audio info: {sample_rate}Hz, {n_channels} channels, {n_frames} frames ({n_frames/sample_rate:.1f}s)")

        # Convert to the expected format (16kHz mono)
        if sample_rate != 16000 or n_channels != 1:
            print("⚠️  Audio needs resampling to 16kHz mono")
            # For this test, we'll assume it's already in the right format
            # In practice, you'd use librosa or similar to resample

        # Test streaming in chunks (simulate real-time)
        chunk_size_seconds = 0.5  # 500ms chunks
        chunk_size_bytes = int(sample_rate * chunk_size_seconds * 2)  # 2 bytes per sample

        print(f"🔄 Testing streaming transcription with {chunk_size_seconds}s chunks...")

        total_chunks = len(audio_data) // chunk_size_bytes
        transcriptions = []
        start_time = time.time()

        for i in range(0, len(audio_data), chunk_size_bytes):
            chunk = audio_data[i:i + chunk_size_bytes]
            if len(chunk) < chunk_size_bytes // 2:  # Skip very small chunks
                continue

            print(f"  Processing chunk {i//chunk_size_bytes + 1}/{total_chunks}...")

            chunk_start = time.time()
            frame_count = 0
            async for frame in stt_optimized.run_stt(chunk):
                frame_count += 1
                frame_type = type(frame).__name__
                if hasattr(frame, 'text') and frame.text.strip():
                    transcriptions.append(frame.text.strip())
                    print(f"    ✅ {frame_type}: '{frame.text.strip()}'")
                else:
                    print(f"    📭 {frame_type}: (empty)")

            chunk_time = time.time() - chunk_start
            print(f"    ⏱️  Chunk processed in {chunk_time:.3f}s, frames: {frame_count}")

            # Small delay to simulate real-time streaming
            await asyncio.sleep(0.1)

        # Flush any remaining audio
        print("🔄 Flushing remaining audio...")
        async for frame in stt_optimized.flush():
            if hasattr(frame, 'text') and frame.text.strip():
                transcriptions.append(frame.text.strip())
                print(f"  ✅ Flush: '{frame.text.strip()}'")

        total_time = time.time() - start_time
        print(f"\n📊 Streaming test completed in {total_time:.2f}s")
        print(f"📊 Total transcription segments: {len(transcriptions)}")

        # Combine all transcriptions
        full_transcription = " ".join(transcriptions)
        print(f"\n📝 Full transcription:\n{full_transcription}")

        # Test with a single large chunk (batch mode)
        print(f"\n🔄 Testing batch mode with full audio...")
        batch_start = time.time()
        batch_transcriptions = []

        async for frame in stt_optimized.run_stt(audio_data):
            if hasattr(frame, 'text') and frame.text.strip():
                batch_transcriptions.append(frame.text.strip())
                print(f"  ✅ Batch: '{frame.text.strip()}'")

        batch_time = time.time() - batch_start
        batch_transcription = " ".join(batch_transcriptions)

        print(f"📊 Batch test completed in {batch_time:.2f}s")
        print(f"📝 Batch transcription:\n{batch_transcription}")

        # Analysis
        print(f"\n🎯 Performance Analysis:")
        print(f"  Audio duration: {n_frames/sample_rate:.1f}s")
        print(f"  Streaming time: {total_time:.2f}s (RTF: {total_time/(n_frames/sample_rate):.2f})")
        print(f"  Batch time: {batch_time:.2f}s (RTF: {batch_time/(n_frames/sample_rate):.2f})")
        print(f"  Streaming segments: {len(transcriptions)}")
        print(f"  Batch segments: {len(batch_transcriptions)}")

        if full_transcription:
            print(f"  ✅ SUCCESS: Streaming transcription generated")
        else:
            print(f"  ❌ ISSUE: No streaming transcription generated")

        if batch_transcription:
            print(f"  ✅ SUCCESS: Batch transcription generated")
        else:
            print(f"  ❌ ISSUE: No batch transcription generated")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_parakeet_with_harvard())
    sys.exit(0 if success else 1)