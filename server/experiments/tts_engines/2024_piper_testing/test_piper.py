#!/usr/bin/env python3
"""Test Piper streaming TTS implementation."""

import asyncio
import time
from loguru import logger

async def test_piper_tts():
    """Test Piper streaming TTS."""
    from tts_piper_streaming import PiperStreamingTTS

    logger.info("Testing Piper Streaming TTS...")

    try:
        # Initialize Piper TTS
        tts = PiperStreamingTTS(
            voice="en_US-lessac-medium",
            sample_rate=22050,
            length_scale=1.0  # Normal speed
        )

        # Test texts
        test_texts = [
            "Hello! This is Piper.",
            "Piper provides ultra-fast neural text to speech with streaming support.",
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet, making it perfect for testing text to speech systems.",
        ]

        for text in test_texts:
            logger.info(f"\n🎤 Testing: '{text[:50]}...'")

            start_time = time.time()
            frames_received = 0
            first_frame_time = None
            total_audio_bytes = 0

            async for frame in tts.run_tts(text):
                frame_type = type(frame).__name__

                if "TTSAudioRawFrame" in frame_type:
                    frames_received += 1
                    total_audio_bytes += len(frame.audio)

                    if first_frame_time is None:
                        first_frame_time = time.time() - start_time
                        logger.info(f"✅ TTFB: {first_frame_time*1000:.1f}ms")

                elif "TTSStartedFrame" in frame_type:
                    logger.debug("Started TTS")
                elif "TTSStoppedFrame" in frame_type:
                    logger.debug("Stopped TTS")

            total_time = time.time() - start_time

            logger.info(f"📊 Results:")
            logger.info(f"   Total time: {total_time*1000:.1f}ms")
            logger.info(f"   Audio frames: {frames_received}")
            logger.info(f"   Total audio: {total_audio_bytes/1024:.1f}KB")
            logger.info(f"   Chars/sec: {len(text)/total_time:.1f}")
            logger.info(f"   Real-time factor: {total_time:.2f}s processing for ~{len(text)/15:.1f}s speech")

            if frames_received == 0:
                logger.error("❌ NO AUDIO GENERATED!")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Piper TTS failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run test."""
    logger.info("🚀 Testing Piper Streaming TTS...")

    success = await test_piper_tts()

    if success:
        logger.info("\n✅ Piper TTS test PASSED! Ultra-fast streaming is working!")
    else:
        logger.error("\n❌ Piper TTS test FAILED!")

if __name__ == "__main__":
    asyncio.run(main())