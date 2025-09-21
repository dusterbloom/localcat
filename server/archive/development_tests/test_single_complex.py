#!/usr/bin/env python3
"""Test a single complex case to see token chunking performance."""

import asyncio
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))

from tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from text_chunker import chunk_text_ultra_low_latency, estimate_tokens

async def test_complex_case():
    """Test the complex technical explanation that would cause PC heating."""

    # This is the text that caused PC heating before
    complex_text = """The implementation of token-based text chunking in Kokoro TTS represents a significant advancement in ultra-low latency speech synthesis. By pre-processing text into optimal token ranges of 175-250 tokens before sending to the MLX worker process, we can achieve consistent time-to-first-byte performance while preventing the system overload that occurs when processing large text blocks as single units. This approach leverages the natural language processing capabilities of the tokenization algorithm to identify semantic boundaries within the text, ensuring that chunk breaks occur at linguistically appropriate points rather than arbitrary character limits. The resulting audio maintains natural prosody and intonation while delivering the responsiveness required for real-time conversational applications. Furthermore, this chunking strategy aligns with Apple Silicon hardware optimization patterns, maximizing the efficiency of MLX framework operations on M-series processors."""

    print(f"🧪 Testing complex case that previously caused PC heating")
    print(f"Text length: {len(complex_text)} chars")
    print(f"Estimated tokens: {estimate_tokens(complex_text)}")

    # Show chunking preview
    chunks = chunk_text_ultra_low_latency(complex_text)
    print(f"Will be chunked into {len(chunks)} token-optimized chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: ~{estimate_tokens(chunk)} tokens")
    print()

    # Initialize TTS with ultra-low latency settings
    tts = TTSMLXUltraLowLatency(
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_heart",
        use_boundaries=False,  # Token-based chunking instead
        buffer_ms=50
    )

    if not await tts._initialize_if_needed():
        print("❌ Failed to initialize TTS")
        return

    print("✅ TTS initialized")
    print("🚀 Starting token-based TTS generation...")

    start_time = time.time()
    chunk_count = 0
    total_bytes = 0
    ttfb = None

    async for frame in tts.run_tts(complex_text):
        if isinstance(frame, TTSStartedFrame):
            print("🎤 TTS Started")
        elif isinstance(frame, TTSAudioRawFrame):
            chunk_count += 1
            total_bytes += len(frame.audio)
            if ttfb is None:
                ttfb = (time.time() - start_time) * 1000
                print(f"🚀 First audio chunk: {ttfb:.1f}ms TTFB, {len(frame.audio)} bytes")
            else:
                print(f"🔊 Audio chunk {chunk_count}: {len(frame.audio)} bytes")
        elif isinstance(frame, TTSStoppedFrame):
            print("⏹️ TTS Stopped")

    total_time = (time.time() - start_time) * 1000

    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"  TTFB: {ttfb:.1f}ms")
    print(f"  Total time: {total_time:.1f}ms")
    print(f"  Audio chunks: {chunk_count}")
    print(f"  Total audio: {total_bytes:,} bytes")
    print(f"  Avg per token chunk: {total_time / len(chunks):.1f}ms")

    # Check if this would have been fast enough to prevent PC heating
    if ttfb and ttfb < 3000:  # Under 3 seconds TTFB
        print("✅ SUCCESS: Fast enough to prevent PC heating!")
    else:
        print("⚠️ Still too slow, may cause heating on complex questions")

if __name__ == "__main__":
    asyncio.run(test_complex_case())