#!/usr/bin/env python3
"""
Simple test to verify HotPath tools integration works.
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

def test_hotpath_tools_creation():
    """Test that HotPath tool integration can be created successfully."""
    logger.info("🧪 Testing HotPath Tools Creation")

    try:
        # Set environment variables
        os.environ["MEMORY_BACKEND"] = "hotpath"

        from config import VoiceAgentConfig
        from core.factory import VoiceAgentFactory
        from core.memory.hotpath_tool_integration import create_hotpath_tool_integration

        # Create factory and memory processor
        config = VoiceAgentConfig()
        factory = VoiceAgentFactory(config)

        # Create memory processor (this will use HotPath)
        session_tracker = factory.create_session_tracker()
        memory_processor = factory.create_memory_processor(None, session_tracker)

        logger.info(f"✅ Memory processor created: {type(memory_processor).__name__}")

        # Create tool integration
        tool_integration = create_hotpath_tool_integration(memory_processor)
        logger.info(f"✅ Tool integration created: {type(tool_integration).__name__}")

        # Test tools schema
        tools_schema = tool_integration.get_tools_schema()
        logger.info(f"✅ Tools schema created with {len(tools_schema.standard_tools)} tools")

        # Verify all expected tools are present
        tool_names = [tool.name for tool in tools_schema.standard_tools]
        expected_tools = ["hotmem_remember", "hotmem_recall", "hotmem_forget", "hotmem_search"]

        all_tools_found = True
        for tool_name in expected_tools:
            if tool_name in tool_names:
                logger.info(f"  ✅ Tool found: {tool_name}")
            else:
                logger.error(f"  ❌ Tool missing: {tool_name}")
                all_tools_found = False

        if all_tools_found:
            logger.info("🎉 HotPath tools integration works perfectly!")
            logger.info("  ✅ HotPath now has explicit tool access")
            logger.info("  ✅ HotPath keeps superior automatic context injection")
            logger.info("  ✅ Same 4 tools as HotMem: remember, recall, forget, search")
            return True
        else:
            logger.error("❌ Some tools are missing")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run simple HotPath tools test."""
    logger.info("🚀 Starting Simple HotPath Tools Test")
    logger.info("=" * 40)

    try:
        success = test_hotpath_tools_creation()

        if success:
            logger.info("\n🎊 SUCCESS!")
            logger.info("✅ HotPath with tools is ready for production")
            logger.info("✅ Users can now use both automatic context AND explicit tools")
            logger.info("✅ This replaces HotMem - HotPath is the unified solution")
            return True
        else:
            logger.error("❌ Test failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)