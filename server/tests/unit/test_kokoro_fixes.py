#!/usr/bin/env python3
"""Test Kokoro TTS to identify specific issues."""

import asyncio
import time
from loguru import logger

# Test the current Kokoro implementation
async def test_kokoro_simple():
    """Test the MLXKokoroSimple implementation."""
    from tts_mlx_simple import MLXKokoroSimple

    logger.info("Testing MLXKokoroSimple...")

    try:
        tts = MLXKokoroSimple(voice="af_heart", sample_rate=24000)

        # Test texts of different lengths
        test_texts = [
            "Hello!",
            "This is a medium length sentence to test streaming.",
            "This is a much longer piece of text that should really test how well the Kokoro TTS system handles extended content with multiple clauses, and we want to see if it maintains good quality throughout the entire generation process."
        ]

        for text in test_texts:
            logger.info(f"\n🎤 Testing: '{text[:50]}...' ({len(text)} chars)")

            start_time = time.time()
            frames_received = 0
            total_audio_bytes = 0
            first_frame_time = None

            async for frame in tts.run_tts(text):
                frame_type = type(frame).__name__

                if "TTSAudioRawFrame" in frame_type:
                    frames_received += 1
                    total_audio_bytes += len(frame.audio)

                    if first_frame_time is None:
                        first_frame_time = time.time() - start_time
                        logger.info(f"  ✅ First audio frame in {first_frame_time*1000:.1f}ms")

                elif "TTSStartedFrame" in frame_type:
                    logger.debug("  Started TTS generation")
                elif "TTSStoppedFrame" in frame_type:
                    logger.debug("  Stopped TTS generation")

            total_time = time.time() - start_time

            logger.info(f"  📊 Results:")
            logger.info(f"     - Total time: {total_time*1000:.1f}ms")
            logger.info(f"     - Audio frames: {frames_received}")
            logger.info(f"     - Total audio: {total_audio_bytes/1024:.1f}KB")
            logger.info(f"     - Chars/sec: {len(text)/total_time:.1f}")

            if frames_received == 0:
                logger.error("  ❌ NO AUDIO GENERATED!")
                return False

    except Exception as e:
        logger.error(f"❌ MLXKokoroSimple failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return True

async def test_native_kokoro():
    """Test native Kokoro package directly."""
    logger.info("\n🔬 Testing native Kokoro package...")

    try:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code='a')

        test_text = "Hello, this is a test of the native Kokoro pipeline."
        logger.info(f"Testing: '{test_text}'")

        start_time = time.time()
        chunks_received = 0

        generator = pipeline(test_text, voice="af_heart")

        for gs, ps, audio in generator:
            chunks_received += 1
            logger.debug(f"  Chunk {chunks_received}: audio shape={audio.shape if hasattr(audio, 'shape') else 'unknown'}")

            if chunks_received >= 5:  # Just test first few chunks
                break

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"  ✅ Native Kokoro: {chunks_received} chunks in {elapsed:.1f}ms")

        if chunks_received == 0:
            logger.error("  ❌ No audio chunks generated!")
            return False

    except Exception as e:
        logger.error(f"❌ Native Kokoro failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return True

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Kokoro TTS diagnostics...")

    # Test native first
    native_ok = await test_native_kokoro()

    # Then test our wrapper
    simple_ok = await test_kokoro_simple()

    # Summary
    logger.info("\n📊 DIAGNOSTIC SUMMARY:")
    logger.info(f"  Native Kokoro: {'✅ OK' if native_ok else '❌ FAILED'}")
    logger.info(f"  MLXKokoroSimple: {'✅ OK' if simple_ok else '❌ FAILED'}")

    if not native_ok:
        logger.error("\n⚠️  Native Kokoro is broken - need alternative TTS!")
        logger.info("Suggested alternatives:")
        logger.info("  1. Marvis TTS (already in codebase)")
        logger.info("  2. StyleTTS2 (high quality)")
        logger.info("  3. Coqui TTS (XTTS v2)")
        logger.info("  4. Piper TTS (very fast)")
    elif not simple_ok:
        logger.error("\n⚠️  Wrapper implementation has issues")

if __name__ == "__main__":
    asyncio.run(main())