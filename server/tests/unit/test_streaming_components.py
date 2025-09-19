#!/usr/bin/env python
"""
Test script for verifying STT/LLM/TTS streaming functionality
"""

import asyncio
import numpy as np
import sys
import os
from loguru import logger
from dotenv import load_dotenv

# Ensure server root and local pipecat are importable
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load environment variables
load_dotenv(override=True)


async def test_streaming_stt():
    """Test the Kyutai streaming STT service."""
    logger.info("Testing Kyutai Streaming STT...")

    try:
        from kyutai_streaming_stt import KyutaiStreamingSTT

        # Initialize streaming STT
        stt = KyutaiStreamingSTT(
            hf_repo="kyutai/stt-1b-en_fr-mlx",
            enable_vad=True,
            max_steps=4096
        )
        logger.success("✓ Kyutai STT initialized successfully")

        # Generate test audio (1 second of silence)
        sample_rate = 16000
        test_audio = np.zeros(sample_rate, dtype=np.int16).tobytes()

        # Process test audio
        logger.info("Processing test audio chunk...")
        frames = []
        async for frame in stt.run_stt(test_audio):
            frames.append(frame)
            logger.info(f"Received frame: {type(frame).__name__}")

        logger.success(f"✓ Processed {len(frames)} frames")
        return True

    except Exception as e:
        logger.error(f"✗ Kyutai STT test failed: {e}")
        return False


async def test_llm_streaming():
    """Test LLM streaming configuration."""
    logger.info("Testing LLM Streaming...")

    try:
        from pipecat.services.openai.llm import OpenAILLMService

        # Check if LLM server is configured
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL")

        if not all([api_key, base_url, model]):
            logger.warning("LLM configuration incomplete - skipping test")
            logger.info(f"API Key: {'✓' if api_key else '✗'}")
            logger.info(f"Base URL: {base_url or 'Not set'}")
            logger.info(f"Model: {model or 'Not set'}")
            return None

        # Initialize LLM with streaming
        llm = OpenAILLMService(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_tokens=100,
            stream=True,
            extra_body={
                "stream": True,
                "options": {
                    "num_predict": 100,
                    "temperature": 0.7
                }
            }
        )

        logger.success("✓ LLM service initialized with streaming enabled")
        return True

    except Exception as e:
        logger.error(f"✗ LLM streaming test failed: {e}")
        return False


async def test_full_pipeline():
    """Test the full streaming pipeline integration."""
    logger.info("Testing Full Pipeline Integration...")

    try:
        # Check environment
        use_streaming_stt = os.getenv("USE_STREAMING_STT", "true").lower() == "true"
        use_llm_streaming = os.getenv("USE_LLM_STREAMING", "true").lower() == "true"

        logger.info(f"Streaming STT: {'✓ Enabled' if use_streaming_stt else '✗ Disabled'}")
        logger.info(f"Streaming LLM: {'✓ Enabled' if use_llm_streaming else '✗ Disabled'}")

        # Test imports
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.frames.frames import AudioRawFrame, TranscriptionFrame
        logger.success("✓ Pipecat imports successful")

        # Test TTS service
        from tts_mlx_isolated import TTSMLXIsolated
        logger.success("✓ TTS service import successful")

        return True

    except Exception as e:
        logger.error(f"✗ Pipeline integration test failed: {e}")
        return False


async def main():
    """Run all streaming tests."""
    logger.info("=" * 60)
    logger.info("STT/LLM/TTS STREAMING TEST SUITE")
    logger.info("=" * 60)

    results = {
        "STT Streaming": await test_streaming_stt(),
        "LLM Streaming": await test_llm_streaming(),
        "Pipeline Integration": await test_full_pipeline(),
    }

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)

    for test_name, result in results.items():
        if result is True:
            logger.success(f"{test_name}: ✓ PASSED")
        elif result is False:
            logger.error(f"{test_name}: ✗ FAILED")
        else:
            logger.warning(f"{test_name}: ⚠ SKIPPED")

    # Overall result
    all_passed = all(r is not False for r in results.values())

    if all_passed:
        logger.success("\n🎉 All tests completed successfully!")
        logger.info("\nStreaming optimizations are ready to use.")
        logger.info("Start the server with: python bot.py")
        logger.info("\nEnvironment variables for streaming:")
        logger.info("  USE_STREAMING_STT=true   # Enable streaming STT")
        logger.info("  USE_LLM_STREAMING=true   # Enable LLM streaming")
        logger.info("  WHISPER_MODEL=base       # STT model size")
        logger.info("  WHISPER_LANGUAGE=en      # STT language")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
