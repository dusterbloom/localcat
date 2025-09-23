#!/usr/bin/env python3
"""
Test the simple MLX implementation vs ONNX.
"""

import asyncio
import time
import sys
import os
from pathlib import Path

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.tts.kokoro_professional import ProfessionalKokoroTTSService as NativeKokoroTTSService
from core.tts.kokoro_mlx import MLXKokoroTTSService as MLXKokoroSimple


async def test_simple_mlx():
    """Test the simple MLX implementation."""

    test_texts = [
        "Hello world",
        "This is a quick test of the simple MLX implementation.",
        "The quick brown fox jumps over the lazy dog and runs through the forest."
    ]

    print("🧪 Testing Simple MLX Kokoro Implementation\n")

    # Test Simple MLX
    print("🚀 Testing Simple MLX Kokoro...")
    try:
        async with MLXKokoroSimple(voice="af_bella") as mlx_tts:
            for i, text in enumerate(test_texts):
                print(f"  Test {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")

                start_time = time.time()
                frames = []

                async for frame in mlx_tts.run_tts(text):
                    frames.append(frame)

                generation_time = (time.time() - start_time) * 1000
                chars_per_sec = len(text) / (generation_time / 1000) if generation_time > 0 else 0

                print(f"    ⚡ {generation_time:.1f}ms ({chars_per_sec:.1f} chars/s) - {len(frames)} frames")

    except Exception as e:
        print(f"❌ Simple MLX test failed: {e}")

    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(test_simple_mlx())