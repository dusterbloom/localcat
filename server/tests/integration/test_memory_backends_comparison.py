#!/usr/bin/env python3
"""
Final test to compare hotpath vs hotmem backends after the fix.
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

def test_memory_backend(backend_name: str):
    """Test a specific memory backend."""
    logger.info(f"🧪 Testing {backend_name} backend")

    os.environ["MEMORY_BACKEND"] = backend_name

    try:
        from config import VoiceAgentConfig
        from core.factory import VoiceAgentFactory

        # Create factory and services
        config = VoiceAgentConfig()
        factory = VoiceAgentFactory(config)

        # Test LLM service creation (this was failing before)
        llm_service = factory.create_llm_service()
        logger.info(f"✅ LLM service created: {type(llm_service).__name__}")

        # Test memory service creation
        session_tracker = factory.create_session_tracker()

        if backend_name == "hotmem":
            memory_service = factory.create_hotmem_service(None, session_tracker)
            logger.info(f"✅ HotMemService created: {type(memory_service).__name__}")

            # Test tool integration
            try:
                from core.memory.hotmem_tool_integration import create_hotmem_tool_integration
                tool_integration = create_hotmem_tool_integration(memory_service)
                logger.info(f"✅ Tool integration created: {len(tool_integration.get_tools_schema().standard_tools)} tools")
            except Exception as e:
                logger.warning(f"⚠️ Tool integration failed: {e}")
        else:
            memory_service = factory.create_memory_processor(None, session_tracker)
            logger.info(f"✅ HotPathMemoryProcessor created: {type(memory_service).__name__}")

        return True

    except Exception as e:
        logger.error(f"❌ {backend_name} failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main():
    """Compare both memory backends."""
    logger.info("🔍 Memory Backend Comparison Test")
    logger.info("=" * 40)

    results = {}

    # Test both backends
    for backend in ["hotpath", "hotmem"]:
        results[backend] = test_memory_backend(backend)
        logger.info("")

    # Summary
    logger.info("📊 RESULTS:")
    for backend, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {backend}: {status}")

    # Final assessment
    hotpath_works = results.get("hotpath", False)
    hotmem_works = results.get("hotmem", False)

    if hotpath_works and hotmem_works:
        logger.info("\n🎉 SUCCESS: Both memory backends are working!")
        logger.info("✅ The UnboundLocalError bug has been fixed")
        logger.info("✅ HotMem backend is now functional")
        return True
    elif hotpath_works and not hotmem_works:
        logger.error("\n❌ PARTIAL: hotpath works but hotmem still has issues")
        return False
    elif not hotpath_works and hotmem_works:
        logger.error("\n❌ PARTIAL: hotmem works but hotpath is broken")
        return False
    else:
        logger.error("\n💥 FAILURE: Both backends are broken")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)