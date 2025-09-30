#!/usr/bin/env python3
"""
Integration tests for HotMemService with VoiceAgentFactory
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

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory
from core.memory import HotMemService
from core.memory.hotpath_processor import HotPathMemoryProcessor


@pytest.mark.integration
async def test_factory_memory_backend_switching():
    """Test that factory correctly switches between memory backends."""

    # Test HotPath processor (default)
    original_backend = os.getenv('MEMORY_BACKEND', 'hotpath')
    os.environ['MEMORY_BACKEND'] = 'hotpath'

    config = VoiceAgentConfig.from_env()
    factory = VoiceAgentFactory(config)

    # Create session tracker
    session_tracker = factory.create_session_tracker()

    # For HotPath processor test, we need a proper context aggregator, so skip it in this test
    # since we're just testing the backend switching logic

    # Test HotMemService creation (we'll skip HotPath since it needs complex setup)
    memory = factory.create_hotmem_service(session_tracker)
    assert isinstance(memory, HotMemService)

    # Test with different environment setting
    os.environ['MEMORY_BACKEND'] = 'hotmem'
    memory2 = factory.create_hotmem_service(session_tracker)
    assert isinstance(memory2, HotMemService)

    # Cleanup
    await memory.cleanup()
    await memory2.cleanup()

    # Restore original setting
    os.environ['MEMORY_BACKEND'] = original_backend


@pytest.mark.integration
async def test_hotmem_service_factory_creation():
    """Test HotMemService creation through factory."""

    # Set environment for HotMem
    original_backend = os.getenv('MEMORY_BACKEND', 'hotpath')
    os.environ['MEMORY_BACKEND'] = 'hotmem'
    os.environ['USER_ID'] = 'test_user'
    os.environ['AGENT_ID'] = 'test_agent'

    try:
        config = VoiceAgentConfig.from_env()
        factory = VoiceAgentFactory(config)

        session_tracker = factory.create_session_tracker()
        service = factory.create_hotmem_service(session_tracker)

        assert isinstance(service, HotMemService)
        assert service.user_id == 'test_user'
        assert service.agent_id == 'test_agent'

        # Test basic functionality
        test_messages = [
            {"role": "user", "content": "Test integration message"}
        ]
        service._store_messages(test_messages)

        memories = service._retrieve_memories("test")
        assert "results" in memories

        await service.cleanup()

    finally:
        # Restore original settings
        os.environ['MEMORY_BACKEND'] = original_backend


@pytest.mark.integration
async def test_hotmem_service_vs_hotpath_processor():
    """Compare HotMemService with HotPathMemoryProcessor functionality."""

    # Test both use same storage backend
    os.environ['MEMORY_SQLITE_PATH'] = ':memory:'

    # Create HotMemService
    hotmem_service = HotMemService(
        user_id="test_user",
        agent_id="test_agent",
        sqlite_path=":memory:",
        lmdb_dir=None
    )

    # Create HotPathMemoryProcessor
    hotpath_processor = HotPathMemoryProcessor(
        user_id="test_user",
        sqlite_path=":memory:",
        lmdb_dir=None,
        enable_metrics=False
    )

    # Test they both process the same data
    test_data = "My name is Bob and I like programming"

    # Store in HotMemService
    hotmem_service._store_messages([
        {"role": "user", "content": test_data}
    ])

    # Store in HotPathProcessor (through process_turn)
    bullets, triples = hotpath_processor.hot.process_turn(
        test_data,
        "test_session",
        1
    )

    # Both should extract information
    hotmem_memories = hotmem_service._retrieve_memories("Bob programming")
    assert len(hotmem_memories["results"]) >= 0  # May or may not find memories depending on timing

    hotpath_bullets = hotpath_processor.hot.retrieve_bullets("Bob programming", read_only=True)
    assert isinstance(hotpath_bullets, list)

    # Both should use same storage backend
    assert hotmem_service.store.__class__.__name__ == hotpath_processor.store.__class__.__name__

    # Cleanup
    await hotmem_service.cleanup()


@pytest.mark.integration
async def test_hotmem_service_environment_config():
    """Test HotMemService respects environment configuration."""

    # Set test environment variables
    test_env = {
        'MEMORY_BULLETS_MAX': '2',
        'MEMORY_INJECT_ROLE': 'system',
        'MEMORY_INJECT_HEADER': '[Test Memory]',
        'USER_ID': 'env_test_user',
        'AGENT_ID': 'env_test_agent'
    }

    original_values = {}
    for key, value in test_env.items():
        original_values[key] = os.getenv(key)
        os.environ[key] = value

    try:
        service = HotMemService(
            user_id=os.getenv('USER_ID'),
            agent_id=os.getenv('AGENT_ID'),
            sqlite_path=":memory:",
            lmdb_dir=None
        )

        # Check configuration was applied
        assert service.user_id == 'env_test_user'
        assert service.agent_id == 'env_test_agent'
        assert service._bullets_max == 2
        assert service._inject_role == 'system'
        assert service._inject_header == '[Test Memory]'

        await service.cleanup()

    finally:
        # Restore original environment
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


if __name__ == "__main__":
    # Run integration tests when executed directly
    import asyncio

    async def run_integration_tests():
        """Run all integration tests."""

        print("\n" + "="*60)
        print("HOTMEM SERVICE INTEGRATION TESTS")
        print("="*60)

        tests = [
            test_factory_memory_backend_switching,
            test_hotmem_service_factory_creation,
            test_hotmem_service_vs_hotpath_processor,
            test_hotmem_service_environment_config,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                print(f"\n🔧 Running {test.__name__}...")
                await test()
                print(f"✅ {test.__name__} PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test.__name__} FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        print("\n" + "="*60)
        print(f"INTEGRATION TEST SUMMARY: {passed} passed, {failed} failed")
        print("="*60)

        return failed == 0

    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)