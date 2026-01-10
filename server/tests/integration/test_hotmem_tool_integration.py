#!/usr/bin/env python3
"""
Test script for HotMemService tool integration.

This script verifies that:
1. HotMemService tools are properly registered with the LLM service
2. Tool schemas are correctly formatted
3. Function call handlers work as expected
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from loguru import logger

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory
from core.memory.hotmem_service import HotMemService
from core.memory.hotmem_tool_integration import HotMemToolIntegration
from core.memory.session_tracker import SessionTracker


async def test_hotmem_tool_integration():
    """Test that HotMemService tools are properly integrated with LLM service."""

    logger.info("🧪 Testing HotMemService tool integration")

    # Set up test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sqlite_path = temp_path / "test_memory.db"
        lmdb_dir = temp_path / "test_lmdb"

        # Set environment variables for testing
        os.environ["MEMORY_BACKEND"] = "hotmem"
        os.environ["MEMORY_SQLITE_PATH"] = str(sqlite_path)
        os.environ["MEMORY_LMDB_PATH"] = str(lmdb_dir)
        os.environ["USER_ID"] = "test-user"
        os.environ["AGENT_ID"] = "test-agent"

        try:
            # Create config
            config = VoiceAgentConfig()

            # Create factory
            factory = VoiceAgentFactory(config)

            # Create session tracker
            session_tracker = factory.create_session_tracker()

            # Create HotMemService
            hotmem_service = factory.create_hotmem_service(session_tracker)
            logger.info("✅ HotMemService created")

            # Test 1: Create tool integration
            hotmem_integration = HotMemToolIntegration(hotmem_service)
            logger.info("✅ HotMemToolIntegration created")

            # Test 2: Verify tool schemas
            tools_schema = hotmem_integration.get_tools_schema()
            tool_names = [tool.name for tool in tools_schema.standard_tools]
            expected_tools = {'hotmem_remember', 'hotmem_recall', 'hotmem_forget', 'hotmem_search'}

            assert set(tool_names) == expected_tools, f"Expected {expected_tools}, got {set(tool_names)}"
            logger.info(f"✅ All {len(expected_tools)} tools have correct schemas: {tool_names}")

            # Test 3: Create LLM service and register tools
            llm_service = factory.create_llm_service()
            logger.info("✅ LLM service created")

            # Register tools with LLM
            hotmem_integration.register_tools_with_llm(llm_service)
            logger.info("✅ Tools registered with LLM service")

            # Test 4: Verify function handlers are registered
            registered_functions = set(llm_service._functions.keys())
            # Check that all expected tools are registered (None may or may not be present)
            assert expected_tools.issubset(registered_functions), \
                f"Missing functions: {expected_tools - registered_functions}"
            logger.info(f"✅ All tool handlers registered: {expected_tools}")

            # Test 5: Test tool call handling (mock)
            from pipecat.services.llm_service import FunctionCallParams

            # Mock result callback
            results = []
            async def mock_result_callback(result):
                results.append(result)
                logger.info(f"📞 Tool call result: {result[:100]}...")

            # Test remember function
            remember_params = FunctionCallParams(
                function_name="hotmem_remember",
                tool_call_id="test-remember-001",
                arguments={"information": "The user prefers coffee over tea"},
                llm=llm_service,
                context=None,  # Not needed for this test
                result_callback=mock_result_callback
            )

            await hotmem_integration._handle_hotmem_remember(remember_params)
            assert len(results) == 1
            assert "Remembered" in results[0]
            logger.info("✅ hotmem_remember handler works")

            # Test recall function
            results.clear()
            recall_params = FunctionCallParams(
                function_name="hotmem_recall",
                tool_call_id="test-recall-001",
                arguments={"query": "What does the user prefer to drink?"},
                llm=llm_service,
                context=None,
                result_callback=mock_result_callback
            )

            await hotmem_integration._handle_hotmem_recall(recall_params)
            assert len(results) == 1
            logger.info("✅ hotmem_recall handler works")

            # Test search function
            results.clear()
            search_params = FunctionCallParams(
                function_name="hotmem_search",
                tool_call_id="test-search-001",
                arguments={"query": "preferences", "search_type": "semantic"},
                llm=llm_service,
                context=None,
                result_callback=mock_result_callback
            )

            await hotmem_integration._handle_hotmem_search(search_params)
            assert len(results) == 1
            logger.info("✅ hotmem_search handler works")

            # Test forget function
            results.clear()
            forget_params = FunctionCallParams(
                function_name="hotmem_forget",
                tool_call_id="test-forget-001",
                arguments={"query": "old preference"},
                llm=llm_service,
                context=None,
                result_callback=mock_result_callback
            )

            await hotmem_integration._handle_hotmem_forget(forget_params)
            assert len(results) == 1
            # New implementation returns "forgotten" or "No memory found"
            assert "forgotten" in results[0] or "No memory found" in results[0] or "Processed forget" in results[0]
            logger.info("✅ hotmem_forget handler works")

            # Cleanup
            await hotmem_service.cleanup()

            logger.info("🎉 All HotMem tool integration tests passed!")
            return True

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_factory_integration():
    """Test that tools are properly integrated through the factory."""

    logger.info("🏭 Testing factory integration of HotMem tools")

    # Set up test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sqlite_path = temp_path / "test_factory_memory.db"
        lmdb_dir = temp_path / "test_factory_lmdb"

        # Set environment variables
        os.environ["MEMORY_BACKEND"] = "hotmem"
        os.environ["MEMORY_SQLITE_PATH"] = str(sqlite_path)
        os.environ["MEMORY_LMDB_PATH"] = str(lmdb_dir)
        os.environ["USER_ID"] = "factory-test-user"

        try:
            # Create config
            config = VoiceAgentConfig()

            # Create factory
            factory = VoiceAgentFactory(config)

            # Create individual services to test tool integration (avoid full voice agent creation)
            session_tracker = factory.create_session_tracker()
            memory = factory.create_hotmem_service(session_tracker)
            llm_service = factory.create_llm_service()

            logger.info("✅ Individual services created through factory")

            # Now test tool integration manually
            from core.memory.hotmem_tool_integration import create_hotmem_tool_integration

            hotmem_integration = create_hotmem_tool_integration(memory)
            hotmem_integration.register_tools_with_llm(llm_service)

            logger.info("✅ HotMem tools registered through factory pattern")

            # Verify LLM service has tools registered
            registered_functions = set(llm_service._functions.keys())
            expected_tools = {'hotmem_remember', 'hotmem_recall', 'hotmem_forget', 'hotmem_search'}

            assert expected_tools.issubset(registered_functions), \
                f"Missing tools in LLM service: {expected_tools - registered_functions}"
            logger.info("✅ HotMem tools registered with LLM service through factory pattern")

            # Cleanup
            await memory.cleanup()

            logger.info("🎉 Factory integration test passed!")
            return True

        except Exception as e:
            logger.error(f"❌ Factory integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting HotMem tool integration tests")

    # Change to server directory if needed
    server_dir = Path(__file__).parent
    if server_dir.name != 'server':
        server_dir = server_dir / 'server'
        if server_dir.exists():
            os.chdir(server_dir)

    # Run tests
    test1_passed = await test_hotmem_tool_integration()
    test2_passed = await test_factory_integration()

    if test1_passed and test2_passed:
        logger.info("🎊 ALL TESTS PASSED! HotMem tool integration is working correctly.")
        sys.exit(0)
    else:
        logger.error("💥 SOME TESTS FAILED! Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())