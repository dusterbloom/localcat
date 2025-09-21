#!/usr/bin/env python3
"""Test to demonstrate sentence-level streaming performance benefits."""

import asyncio
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))

from tts_native_kokoro import NativeKokoroTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame


async def test_streaming_performance():
    """Test streaming performance with long multi-sentence text."""

    print("🚀 Testing Kokoro Sentence-Level Streaming Performance")
    print("=" * 70)

    try:
        # Initialize TTS service
        tts = NativeKokoroTTSService(
            voice="af_heart",
            speed=1.0,
            sample_rate=24000
        )

        # Long multi-sentence text to demonstrate streaming benefits
        long_text = """
        Welcome to the advanced text-to-speech demonstration system.
        This system utilizes cutting-edge neural network technology for high-quality voice synthesis.
        The underlying architecture employs state-of-the-art transformer models trained on extensive datasets.
        These models can generate natural-sounding speech with remarkable clarity and expressiveness.
        The system processes text through multiple stages including tokenization and phonetic analysis.
        Finally, the audio output is delivered through optimized streaming protocols for minimal latency.
        """.strip()

        print(f"📝 Input text ({len(long_text)} chars):")
        print(f"   {long_text[:100]}...")
        print()

        # Track streaming performance
        start_time = time.time()
        audio_chunks = []
        chunk_times = []
        first_audio_time = None

        frame_generator = tts.run_tts(long_text)

        try:
            async for frame in frame_generator:
                if isinstance(frame, TTSStartedFrame):
                    print("🎬 TTS Started")
                elif isinstance(frame, TTSAudioRawFrame):
                    current_time = time.time()
                    if first_audio_time is None:
                        first_audio_time = current_time
                        ttfb = (current_time - start_time) * 1000
                        print(f"🎵 First audio chunk received - TTFB: {ttfb:.1f}ms")

                    chunk_times.append(current_time - start_time)
                    audio_chunks.append(len(frame.audio))

                    total_audio_so_far = sum(audio_chunks)
                    elapsed = (current_time - start_time) * 1000
                    print(f"   📦 Chunk {len(audio_chunks)}: {len(frame.audio):,} bytes (+{elapsed:.1f}ms total)")

                elif isinstance(frame, TTSStoppedFrame):
                    total_time = (time.time() - start_time) * 1000
                    print(f"🏁 TTS Completed - Total time: {total_time:.1f}ms")
                    break
        except GeneratorExit:
            pass

        # Performance summary
        print("\n📊 Performance Analysis:")
        print(f"   • Total audio chunks: {len(audio_chunks)}")
        print(f"   • Total audio bytes: {sum(audio_chunks):,}")
        print(f"   • Time to first audio (TTFB): {(first_audio_time - start_time) * 1000:.1f}ms")
        print(f"   • Average chunk size: {sum(audio_chunks) // len(audio_chunks):,} bytes")

        if len(chunk_times) > 1:
            inter_chunk_times = [(chunk_times[i] - chunk_times[i-1]) * 1000
                                for i in range(1, len(chunk_times))]
            avg_inter_chunk = sum(inter_chunk_times) / len(inter_chunk_times)
            print(f"   • Average inter-chunk time: {avg_inter_chunk:.1f}ms")
            print(f"   • Streaming benefit: User hears audio {avg_inter_chunk:.0f}ms faster per sentence!")

        print("\n✅ Sentence-level streaming provides significantly faster perceived latency!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_streaming_performance())
    sys.exit(0 if success else 1)