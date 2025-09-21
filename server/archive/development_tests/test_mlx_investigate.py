#!/usr/bin/env python3
"""Investigate how MLX Kokoro generation actually works."""

import time
from mlx_audio.tts.utils import load_model

def test_mlx_generation():
    """Test if MLX Kokoro supports true streaming or generates complete audio."""

    print("🔍 Investigating MLX Kokoro generation behavior...")

    # Load model
    model = load_model("mlx-community/Kokoro-82M-bf16")

    test_texts = [
        "Hello there!",
        "This is a longer sentence that should take more time to generate completely.",
        "This is an even longer sentence with multiple clauses and phrases that definitely should produce multiple chunks if streaming is supported at the model level."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n🧪 Test {i}: '{text}'")
        print(f"Text length: {len(text)} chars")

        start_time = time.time()
        chunk_count = 0
        total_audio_length = 0

        # Check what the generator actually yields
        for result in model.generate(text=text, voice="af_heart", speed=1.0):
            chunk_count += 1
            audio_length = len(result.audio) if hasattr(result.audio, '__len__') else 0
            total_audio_length += audio_length

            elapsed = (time.time() - start_time) * 1000
            print(f"  Chunk {chunk_count}: {audio_length} samples at {elapsed:.1f}ms")

            # Check if we get more chunks quickly or if this is the only one
            if chunk_count == 1:
                immediate_time = elapsed

        total_time = (time.time() - start_time) * 1000
        print(f"  📊 Total: {chunk_count} chunks, {total_audio_length} samples in {total_time:.1f}ms")

        if chunk_count == 1:
            print(f"  ❌ Single chunk - MLX Kokoro generates complete audio (no streaming)")
        else:
            print(f"  ✅ Multiple chunks - MLX Kokoro supports streaming!")

        print()

if __name__ == "__main__":
    test_mlx_generation()