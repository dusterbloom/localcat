#!/usr/bin/env python3
"""Test real-world TTS performance patterns."""

import asyncio
import time
from tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency

async def test_real_scenario():
    """Test performance with actual conversational responses."""

    print("🧪 Testing Real-World TTS Performance")
    print("=" * 50)

    tts = TTSMLXUltraLowLatency(use_boundaries=True, buffer_ms=80)
    await tts._initialize_if_needed()

    # Real responses from your logs
    test_responses = [
        "Okay, let's talk about quantum physics!",
        "It's definitely a tricky subject, but here's a simplified version: At its core, quantum physics deals with the super tiny - things like atoms and the particles within them.",
        "Here's the gist: Things aren't always definite: Unlike everyday objects, tiny particles don't always have a definite position or speed.",
        "They exist in a probability of being in different places until we measure them.",
        "Think of it like a blurry cloud of possibilities.",
    ]

    total_start = time.time()

    for i, response in enumerate(test_responses):
        print(f"\n📝 Response {i+1}: {response}")
        print(f"   Length: {len(response)} chars")

        start = time.time()
        chunk_count = 0
        first_chunk_time = None

        async for frame in tts.run_tts(response):
            if hasattr(frame, 'audio') and len(frame.audio) > 0:
                chunk_count += 1
                if first_chunk_time is None:
                    first_chunk_time = (time.time() - start) * 1000

        total_time = (time.time() - start) * 1000

        if first_chunk_time:
            print(f"   ⏱️  TTFB: {first_chunk_time:.0f}ms, Total: {total_time:.0f}ms")
            print(f"   🔊 Chunks: {chunk_count}")
        else:
            print(f"   ❌ No audio generated")

        if total_time > 2000:  # Over 2 seconds is problematic
            print(f"   ⚠️  SLOW RESPONSE!")

    total_duration = time.time() - total_start
    print(f"\n🏁 Complete test suite: {total_duration:.1f}s")

    if total_duration > 10:
        print("❌ Overall performance is too slow for real-time conversation")
    else:
        print("✅ Performance acceptable for real-time conversation")

if __name__ == "__main__":
    asyncio.run(test_real_scenario())