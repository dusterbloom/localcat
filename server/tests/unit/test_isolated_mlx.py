#!/usr/bin/env python3
"""
Test the isolated MLX implementation.
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tts_mlx_isolated import MLXKokoroIsolated


async def test_isolated_mlx():
    """Test the isolated MLX implementation."""

    test_texts = [
        "Hello world",
        "This is a test of the isolated MLX worker.",
    ]

    print("🧪 Testing Isolated MLX Kokoro Implementation\n")

    print("🚀 Testing MLX Isolated Worker...")
    try:
        async with MLXKokoroIsolated(voice="af_bella") as mlx_tts:
            for i, text in enumerate(test_texts):
                print(f"  Test {i+1}: {text}")

                start_time = time.time()
                frames = []

                async for frame in mlx_tts.run_tts(text):
                    frames.append(frame)

                generation_time = (time.time() - start_time) * 1000
                chars_per_sec = len(text) / (generation_time / 1000) if generation_time > 0 else 0

                print(f"    ⚡ {generation_time:.1f}ms ({chars_per_sec:.1f} chars/s) - {len(frames)} frames")

        print("\n✅ Isolated MLX test completed successfully!")

    except Exception as e:
        print(f"❌ Isolated MLX test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_isolated_mlx())