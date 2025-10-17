#!/usr/bin/env python3
"""
Direct test of Parakeet STT with properly formatted Harvard audio
"""

import asyncio
import numpy as np
import sys
import time
import wave
from pathlib import Path

# Add server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def test_harvard_direct():
    """Test with proper format and no VAD conflicts"""
    print("🧪 Testing Parakeet STT with properly formatted Harvard audio...")

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        # Create STT service with VAD disabled to avoid conflicts with Smart Turn
        print("🔄 Initializing Parakeet STT (VAD disabled to avoid Smart Turn conflicts)...")
        stt_service = ParakeetStreamingSTT(
            enable_vad=False,  # Disable internal VAD - let Smart Turn handle it
            volume_threshold=0.0001,  # Very low volume threshold
            chunk_duration=1.0,  # 1s chunks as recommended
            sentence_pause_threshold=1.2,  # Longer pause
        )

        # Load properly formatted audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"

        with wave.open(harvard_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        print(f"📊 Audio: {sample_rate}Hz, {n_channels} channel, {n_frames} frames ({n_frames/sample_rate:.1f}s)")
        print(f"📊 Audio size: {len(audio_data)} bytes")

        # Test 1: Direct processing of full audio
        print("\n🔄 Test 1: Direct processing of full audio...")
        start_time = time.time()
        result = stt_service._process_audio_file_fallback(audio_data)
        process_time = time.time() - start_time
        print(f"📝 Result: '{result}'")
        print(f"⏱️  Processed in {process_time:.2f}s (RTF: {process_time/(n_frames/sample_rate):.2f})")

        # Test 2: Streaming chunks
        print("\n🔄 Test 2: Streaming in 2-second chunks...")
        chunk_duration = 2.0  # 2 seconds
        chunk_size = int(sample_rate * chunk_duration * 2)  # 2 bytes per sample
        transcriptions = []

        start_time = time.time()
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < 1000:  # Skip tiny chunks
                continue

            chunk_num = i // chunk_size + 1
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            print(f"  Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} bytes)...")

            chunk_start = time.time()
            frame_count = 0
            async for frame in stt_service.run_stt(chunk):
                frame_count += 1
                if hasattr(frame, 'text') and frame.text.strip():
                    text = frame.text.strip()
                    transcriptions.append(text)
                    print(f"    ✅ '{text}'")
                else:
                    print(f"    📭 {type(frame).__name__}: (empty)")

            chunk_time = time.time() - chunk_start
            print(f"    ⏱️  {chunk_time:.3f}s, {frame_count} frames")

        # Flush any remaining
        print("  🔄 Flushing...")
        async for frame in stt_service.flush():
            if hasattr(frame, 'text') and frame.text.strip():
                text = frame.text.strip()
                transcriptions.append(text)
                print(f"    ✅ Flush: '{text}'")

        total_time = time.time() - start_time
        full_transcription = " ".join(transcriptions)

        print(f"\n📊 Streaming Results:")
        print(f"  Total time: {total_time:.2f}s (RTF: {total_time/(n_frames/sample_rate):.2f})")
        print(f"  Segments: {len(transcriptions)}")
        print(f"  📝 Full: '{full_transcription}'")

        # Test 3: Small chunks to simulate real-time
        print("\n🔄 Test 3: Real-time simulation (0.5s chunks)...")
        small_chunk_duration = 0.5
        small_chunk_size = int(sample_rate * small_chunk_duration * 2)
        rt_transcriptions = []

        start_time = time.time()
        for i in range(0, len(audio_data), small_chunk_size):
            chunk = audio_data[i:i + small_chunk_size]
            if len(chunk) < 500:  # Skip tiny chunks
                continue

            # Simulate real-time by waiting
            await asyncio.sleep(0.1)

            async for frame in stt_service.run_stt(chunk):
                if hasattr(frame, 'text') and frame.text.strip():
                    text = frame.text.strip()
                    rt_transcriptions.append(text)
                    print(f"    RT: '{text}'")

        rt_full = " ".join(rt_transcriptions)
        print(f"  📝 Real-time: '{rt_full}'")

        # Summary
        print(f"\n🎯 Summary:")
        if result:
            print(f"  ✅ Direct processing: SUCCESS")
        else:
            print(f"  ❌ Direct processing: No output")

        if full_transcription:
            print(f"  ✅ Streaming: SUCCESS ({len(transcriptions)} segments)")
        else:
            print(f"  ❌ Streaming: No output")

        if rt_full:
            print(f"  ✅ Real-time: SUCCESS ({len(rt_transcriptions)} segments)")
        else:
            print(f"  ❌ Real-time: No output")

        return bool(result or full_transcription or rt_full)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_harvard_direct())
    sys.exit(0 if success else 1)