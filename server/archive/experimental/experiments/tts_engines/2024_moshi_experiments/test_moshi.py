#!/usr/bin/env python3
"""Test Moshi TTS implementation."""

import asyncio
import time
from loguru import logger

async def test_moshi_tts():
    """Test Moshi TTS with Delayed Streams."""
    from tts_moshi_delayed import MoshiDelayedTTS

    logger.info("Testing Moshi TTS with Delayed Streams...")

    try:
        # Initialize Moshi TTS
        tts = MoshiDelayedTTS(
            voice="expressive_1",
            sample_rate=24000,
            temperature=0.7
        )

        # Test text
        test_text = "Hello! This is a test of the Moshi text to speech system with delayed streams modeling for ultra-low latency."

        logger.info(f"Testing with: '{test_text[:50]}...'")

        start_time = time.time()
        frames_received = 0
        first_frame_time = None
        total_audio_bytes = 0

        async for frame in tts.run_tts(test_text):
            frame_type = type(frame).__name__

            if "TTSAudioRawFrame" in frame_type:
                frames_received += 1
                total_audio_bytes += len(frame.audio)

                if first_frame_time is None:
                    first_frame_time = time.time() - start_time
                    logger.info(f"✅ First audio frame in {first_frame_time*1000:.1f}ms")

            elif "TTSStartedFrame" in frame_type:
                logger.debug("Started TTS generation")
            elif "TTSStoppedFrame" in frame_type:
                logger.debug("Stopped TTS generation")

        total_time = time.time() - start_time

        logger.info(f"📊 Results:")
        logger.info(f"   Total time: {total_time*1000:.1f}ms")
        logger.info(f"   Audio frames: {frames_received}")
        logger.info(f"   Total audio: {total_audio_bytes/1024:.1f}KB")
        logger.info(f"   Chars/sec: {len(test_text)/total_time:.1f}")

        if frames_received == 0:
            logger.error("❌ NO AUDIO GENERATED!")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Moshi TTS failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run test."""
    logger.info("🚀 Testing Moshi TTS with Delayed Streams Modeling...")

    success = await test_moshi_tts()

    if success:
        logger.info("✅ Moshi TTS test PASSED!")
    else:
        logger.error("❌ Moshi TTS test FAILED!")
        logger.info("Falling back to Kokoro TTS")

if __name__ == "__main__":
    asyncio.run(main())