#!/usr/bin/env python3
"""
Minimal reproduction case for hotmem backend bug.
"""

import os
import sys
import tempfile
from pathlib import Path
from loguru import logger

# Ensure we're in the server directory
os.chdir(Path(__file__).parent)

# Add current directory to path for imports
sys.path.insert(0, '.')

def test_llm_builder_bug():
    """Reproduce the LLM builder bug with hotmem backend."""
    logger.info("🐛 Testing LLM Builder Bug with HotMem")

    try:
        # Set environment to trigger the bug
        os.environ["MEMORY_BACKEND"] = "hotmem"
        logger.info(f"✅ Set MEMORY_BACKEND={os.environ['MEMORY_BACKEND']}")

        # Import config
        from config import VoiceAgentConfig
        from core.factories.builders.llm_builder import LLMServiceBuilder

        # Create config and builder
        config = VoiceAgentConfig()
        builder = LLMServiceBuilder(config)

        # This should trigger the UnboundLocalError
        logger.info("🏗️ Building LLM service...")
        llm_service = builder.build()

        logger.info(f"✅ LLM service created successfully: {type(llm_service)}")
        return True

    except UnboundLocalError as e:
        logger.error(f"❌ UnboundLocalError (this is the bug): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"❌ Other error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_hotmem_vs_hotpath():
    """Compare hotmem vs hotpath LLM builder behavior."""
    logger.info("🔍 Comparing hotmem vs hotpath LLM builder")

    results = {}

    for backend in ["hotpath", "hotmem"]:
        logger.info(f"\n--- Testing {backend} backend ---")
        os.environ["MEMORY_BACKEND"] = backend

        try:
            success = test_llm_builder_bug()
            results[backend] = success

        except Exception as e:
            logger.error(f"❌ {backend} failed: {e}")
            results[backend] = False

    logger.info(f"\n📊 RESULTS:")
    for backend, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {backend}: {status}")

    return results

def main():
    """Run the reproduction case."""
    logger.info("🎯 HotMem Backend Bug Reproduction Case")
    logger.info("=" * 50)

    try:
        results = test_hotmem_vs_hotpath()

        if results.get("hotpath") and not results.get("hotmem"):
            logger.info("\n🎯 BUG CONFIRMED:")
            logger.info("  ✅ hotpath backend works")
            logger.info("  ❌ hotmem backend fails")
            logger.info("\n🔍 ROOT CAUSE:")
            logger.info("  The issue is in llm_builder.py line 64 where OpenAILLMService")
            logger.info("  is referenced but not imported when use_direct_mlx=True")
            logger.info("  and the model doesn't support tools.")
            return True
        elif not results.get("hotpath") and not results.get("hotmem"):
            logger.error("\n❌ BOTH BACKENDS FAIL - this is a different issue")
            return False
        else:
            logger.info("\n✅ BOTH BACKENDS WORK - bug might be fixed")
            return True

    except Exception as e:
        logger.error(f"❌ Reproduction case failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)