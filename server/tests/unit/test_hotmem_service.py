#!/usr/bin/env python3
"""
Unit tests for HotMemService - Pipecat-compatible memory service
"""

import os
import sys
import pytest
from unittest.mock import Mock
from loguru import logger

# Add server root to path for imports
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.memory import HotMemService
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.aggregators.llm_context import LLMContext


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_initialization():
    """Test HotMemService initializes correctly with required parameters."""

    # Test initialization with minimal required parameters
    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent"
    )

    assert service.user_id == "test_user"
    assert service.agent_id == "test_agent"
    assert service._session_id.startswith("test_user_")
    assert hasattr(service, 'TOOL_DEFINITIONS')
    assert len(service.TOOL_DEFINITIONS) == 4

    # Cleanup
    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_tool_definitions():
    """Test that tool definitions are correctly structured."""

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent"
    )

    tools = service.TOOL_DEFINITIONS
    tool_names = {tool['name'] for tool in tools}

    # Check expected tools are present
    expected_tools = {'hotmem_remember', 'hotmem_recall', 'hotmem_forget', 'hotmem_search'}
    assert tool_names == expected_tools

    # Check each tool has required structure
    for tool in tools:
        assert 'name' in tool
        assert 'description' in tool
        assert 'parameters' in tool
        assert 'type' in tool['parameters']
        assert 'properties' in tool['parameters']
        assert 'required' in tool['parameters']

    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_store_messages():
    """Test message storage functionality."""

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",  # Use in-memory DB for testing
        lmdb_dir=None  # Skip LMDB for fast tests
    )

    # Test storing messages
    messages = [
        {"role": "user", "content": "My name is Alice and I work as a developer"},
        {"role": "assistant", "content": "Nice to meet you Alice!"}
    ]

    # Should not raise exception
    service._store_messages(messages)

    # Check turn counter increased
    assert service._turn_id > 0

    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_retrieve_memories():
    """Test memory retrieval functionality."""

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",
        lmdb_dir=None
    )

    # Store some test data first
    messages = [
        {"role": "user", "content": "I love pizza and Italian food"}
    ]
    service._store_messages(messages)

    # Test retrieval
    result = service._retrieve_memories("favorite food")

    assert "results" in result
    assert isinstance(result["results"], list)

    # If memories were found, check structure
    for memory in result["results"]:
        assert "memory" in memory
        assert "score" in memory
        assert "metadata" in memory

    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_context_enhancement():
    """Test context enhancement with memories and tool notice."""

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",
        lmdb_dir=None
    )

    # Create test context
    context = LLMContext()
    context.add_message({"role": "user", "content": "What do you know about me?"})

    initial_message_count = len(context.get_messages())

    # Test context enhancement
    service._enhance_context_with_memories(context, "What do you know about me?")

    enhanced_messages = context.get_messages()

    # Should have added system message with tool notice or memories
    assert len(enhanced_messages) > initial_message_count

    # Check that system message was added
    system_messages = [msg for msg in enhanced_messages if msg.get("role") == "system"]
    assert len(system_messages) > 0

    # Should contain tool availability notice
    tool_notice_found = any(
        "Memory tools available" in msg.get("content", "")
        for msg in system_messages
    )
    assert tool_notice_found

    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_mem0_compatibility():
    """Test Mem0MemoryService compatibility interface."""

    # Test initialization with Mem0-style parameters
    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        run_id="test_run",
        # These should be ignored but not cause errors
        api_key="ignored",
        host="ignored",
        local_config={"ignored": True}
    )

    assert service.user_id == "test_user"
    assert service.agent_id == "test_agent"
    assert service.run_id == "test_run"

    # Test that last_query is tracked (Mem0 compatibility)
    assert service.last_query is None

    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_error_handling():
    """Test error handling for invalid inputs."""

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",
        lmdb_dir=None
    )

    # Test with empty messages list
    service._store_messages([])  # Should not crash

    # Test retrieval with empty query
    result = service._retrieve_memories("")
    assert "results" in result
    assert isinstance(result["results"], list)

    # Test retrieval with None query
    result = service._retrieve_memories(None)
    assert "results" in result

    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
def test_hotmem_service_memory_stats():
    """Test memory statistics retrieval."""

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",
        lmdb_dir=None
    )

    stats = service.get_memory_stats()

    # Check required fields are present
    assert 'session_id' in stats
    assert 'turn_id' in stats
    assert 'user_id' in stats
    assert 'agent_id' in stats

    # Check values are correct
    assert stats['user_id'] == "test_user"
    assert stats['agent_id'] == "test_agent"
    assert isinstance(stats['turn_id'], int)


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_initialization_errors():
    """Test that HotMemService raises appropriate errors for invalid initialization."""

    # Test with no user_id, agent_id, or run_id
    with pytest.raises(ValueError, match="At least one of user_id, agent_id, or run_id must be provided"):
        HotMemService()

    # Test with valid parameters should not raise
    service = HotMemService(user_id="test")
    assert service.user_id == "test"
    await service.cleanup()


@pytest.mark.fast
@pytest.mark.ci
async def test_hotmem_service_performance():
    """Test that HotMemService meets performance requirements."""
    import time

    service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",
        lmdb_dir=None
    )

    # Test storage performance
    messages = [{"role": "user", "content": "Performance test message"}]

    start = time.perf_counter()
    service._store_messages(messages)
    storage_time = (time.perf_counter() - start) * 1000

    # Should be well under 200ms target
    assert storage_time < 200, f"Storage took {storage_time:.1f}ms, expected <200ms"

    # Test retrieval performance
    start = time.perf_counter()
    service._retrieve_memories("test")
    retrieval_time = (time.perf_counter() - start) * 1000

    assert retrieval_time < 200, f"Retrieval took {retrieval_time:.1f}ms, expected <200ms"

    logger.info(f"Performance: storage={storage_time:.1f}ms, retrieval={retrieval_time:.1f}ms")

    await service.cleanup()


if __name__ == "__main__":
    # Run tests when executed directly (legacy mode support)
    import asyncio

    async def run_tests():
        """Run all tests in sequence."""

        print("\n" + "="*60)
        print("HOTMEM SERVICE UNIT TESTS")
        print("="*60)

        tests = [
            test_hotmem_service_initialization,
            test_hotmem_service_tool_definitions,
            test_hotmem_service_store_messages,
            test_hotmem_service_retrieve_memories,
            test_hotmem_service_context_enhancement,
            test_hotmem_service_mem0_compatibility,
            test_hotmem_service_error_handling,
            test_hotmem_service_performance,
            test_hotmem_service_initialization_errors,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                print(f"\n🧪 Running {test.__name__}...")
                await test()
                print(f"✅ {test.__name__} PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test.__name__} FAILED: {e}")
                failed += 1

        # Run sync tests
        sync_tests = [test_hotmem_service_memory_stats]
        for test in sync_tests:
            try:
                print(f"\n🧪 Running {test.__name__}...")
                test()
                print(f"✅ {test.__name__} PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test.__name__} FAILED: {e}")
                failed += 1

        print("\n" + "="*60)
        print(f"TEST SUMMARY: {passed} passed, {failed} failed")
        print("="*60)

        return failed == 0

    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)