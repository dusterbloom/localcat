#!/usr/bin/env python3
"""Test Kokoro TTS streaming directly."""

import asyncio
from tts_mlx_isolated import TTSMLXIsolated
from pipecat.frames.frames import Frame


async def test_streaming():
    """Test Kokoro TTS streaming."""
    print("Initializing Kokoro TTS...")
    tts = TTSMLXIsolated(model="mlx-community/Kokoro-82M-bf16", voice="af_heart", sample_rate=24000)

    try:
        # Initialize if needed
        await tts._initialize_if_needed()
        print("TTS initialized, generating speech...")

        test_text = "Hello, this is a test of Kokoro streaming. We should see multiple audio chunks being generated progressively."

        chunk_count = 0
        audio_frames = []

        async for frame in tts.run_tts(test_text):
            frame_type = type(frame).__name__
            if frame_type == "TTSStartedFrame":
                print("TTS Started")
            elif frame_type == "TTSAudioRawFrame":
                chunk_count += 1
                audio_frames.append(frame)
                print(f"  Chunk {chunk_count}: {len(frame.audio)} bytes")
            elif frame_type == "TTSStoppedFrame":
                print("TTS Stopped")
            elif frame_type == "ErrorFrame":
                print(f"Error: {frame.error}")

        print(f"\nTotal chunks generated: {chunk_count}")
        total_bytes = sum(len(f.audio) for f in audio_frames)
        print(f"Total audio bytes: {total_bytes}")

        if chunk_count > 1:
            print("✅ Streaming is working! Multiple chunks generated.")
        elif chunk_count == 1:
            print("⚠️  Only one chunk generated. Text might be too short for streaming.")
        else:
            print("❌ No audio chunks generated.")
    finally:
        # Cleanup
        tts._cleanup()


if __name__ == "__main__":
    asyncio.run(test_streaming())