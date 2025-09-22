#!/usr/bin/env python3
"""
Test Parakeet in direct mode for fair comparison
"""

import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_parakeet_direct():
    """Test Parakeet direct processing only"""
    try:
        from core.stt.parakeet_streaming import ParakeetStreamingSTT

        stt = ParakeetStreamingSTT(enable_vad=False, confidence_threshold=0.1)

        # Load test audio
        harvard_path = "/Users/peppi/Dev/localcat-streaming/server/experiments/harvard_16k_mono.wav"
        with wave.open(harvard_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            duration = wav_file.getnframes() / wav_file.getframerate()

        print(f"📊 Audio duration: {duration:.1f}s")

        # Test direct processing
        start_time = time.time()
        result = stt._process_audio_file_fallback(audio_data)
        processing_time = time.time() - start_time

        rtf = processing_time / duration
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📈 RTF: {rtf:.3f}")
        print(f"📝 Result: '{result}'")

        return result

    except Exception as e:
        print(f"❌ Failed: {e}")
        return ""

if __name__ == "__main__":
    asyncio.run(test_parakeet_direct())