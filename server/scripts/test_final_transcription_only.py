#!/usr/bin/env python3
"""
Test that Parakeet only sends one final transcription per utterance,
not multiple accumulated ones during streaming.
"""

import asyncio
import sys
import wave
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

async def test_final_transcription_only():
    """Test that only one TranscriptionFrame is sent per utterance"""
    print("🧪 Testing Final Transcription Only")
    print("=" * 50)

    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT
        from pipecat.frames.frames import (
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            TranscriptionFrame,
            InterimTranscriptionFrame
        )

        # Initialize STT
        stt = ParakeetStreamingSTT(
            enable_vad=False,
            volume_threshold=0.0001,
            chunk_duration=1.0,
            context_size=(256, 256),
            depth=3
        )

        print("✅ STT initialized")

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())

        print("\n📊 Testing utterance processing:\n")

        # Test first utterance
        print("📢 First utterance (2 seconds):")
        chunk_size = 16000  # 1 second chunks at 16kHz
        transcription_frames: List[TranscriptionFrame] = []
        interim_frames: List[InterimTranscriptionFrame] = []

        # Simulate user starts speaking
        await stt.process_frame(UserStartedSpeakingFrame())

        # Send audio in chunks (simulating streaming)
        for i in range(0, 32000, chunk_size):  # 2 seconds total
            chunk = audio_data[i:i + chunk_size]

            async for frame in stt.run_stt(chunk):
                if isinstance(frame, TranscriptionFrame):
                    transcription_frames.append(frame)
                    print(f"   ❌ TranscriptionFrame during streaming: '{frame.text}'")
                elif isinstance(frame, InterimTranscriptionFrame):
                    interim_frames.append(frame)
                    print(f"   ⏸️  InterimTranscriptionFrame: '{frame.text}'")

        # Simulate user stops speaking - should trigger flush
        stop_frame = UserStoppedSpeakingFrame()

        # Capture frames from process_frame
        # We need to manually collect frames since we're not in a pipeline
        class FrameCollector:
            def __init__(self):
                self.frames = []

            async def push_frame(self, frame, direction=None):
                self.frames.append(frame)

        collector = FrameCollector()
        stt.push_frame = collector.push_frame
        await stt.process_frame(stop_frame)

        # Add collected frames to our lists
        for frame in collector.frames:
            if isinstance(frame, TranscriptionFrame):
                transcription_frames.append(frame)
                print(f"   📝 Final TranscriptionFrame: '{frame.text}'")

        print(f"\n📊 Results for first utterance:")
        print(f"   TranscriptionFrames sent: {len(transcription_frames)}")
        print(f"   InterimTranscriptionFrames sent: {len(interim_frames)}")

        if len(transcription_frames) == 1:
            print(f"   ✅ SUCCESS: Only one final TranscriptionFrame: '{transcription_frames[0].text}'")
        elif len(transcription_frames) == 0:
            print(f"   ❌ ERROR: No TranscriptionFrame was sent!")
        else:
            print(f"   ❌ ERROR: Multiple TranscriptionFrames sent:")
            for i, frame in enumerate(transcription_frames, 1):
                print(f"      {i}. '{frame.text}'")

        # Test second utterance to ensure reset works
        print("\n📢 Second utterance (different 2 seconds):")
        transcription_frames.clear()
        interim_frames.clear()

        # Simulate user starts speaking again
        await stt.process_frame(UserStartedSpeakingFrame())

        # Send different audio
        for i in range(64000, 96000, chunk_size):  # Different 2 seconds
            chunk = audio_data[i:i + chunk_size]

            async for frame in stt.run_stt(chunk):
                if isinstance(frame, TranscriptionFrame):
                    transcription_frames.append(frame)
                    print(f"   ❌ TranscriptionFrame during streaming: '{frame.text}'")
                elif isinstance(frame, InterimTranscriptionFrame):
                    interim_frames.append(frame)
                    print(f"   ⏸️  InterimTranscriptionFrame: '{frame.text}'")

        # Simulate user stops speaking
        stop_frame = UserStoppedSpeakingFrame()

        # Capture frames from process_frame
        collector = FrameCollector()
        stt.push_frame = collector.push_frame
        await stt.process_frame(stop_frame)

        # Add collected frames to our lists
        for frame in collector.frames:
            if isinstance(frame, TranscriptionFrame):
                transcription_frames.append(frame)
                print(f"   📝 Final TranscriptionFrame: '{frame.text}'")

        print(f"\n📊 Results for second utterance:")
        print(f"   TranscriptionFrames sent: {len(transcription_frames)}")
        print(f"   InterimTranscriptionFrames sent: {len(interim_frames)}")

        if len(transcription_frames) == 1:
            print(f"   ✅ SUCCESS: Only one final TranscriptionFrame: '{transcription_frames[0].text}'")

            # Check it doesn't contain first utterance
            if "stale smell" in transcription_frames[0].text.lower() or "steel smell" in transcription_frames[0].text.lower():
                print(f"   ❌ ERROR: Second utterance contains text from first!")
                return False
        elif len(transcription_frames) == 0:
            print(f"   ❌ ERROR: No TranscriptionFrame was sent!")
            return False
        else:
            print(f"   ❌ ERROR: Multiple TranscriptionFrames sent:")
            for i, frame in enumerate(transcription_frames, 1):
                print(f"      {i}. '{frame.text}'")
            return False

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_final_transcription_only())
    print()
    if success:
        print("🎉 SUCCESS: Parakeet now sends only ONE final transcription per utterance!")
        print("   The duplicate transcription issue has been fixed.")
    else:
        print("💔 The transcription issue needs more work")

    sys.exit(0 if success else 1)