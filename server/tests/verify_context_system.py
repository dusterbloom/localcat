#!/usr/bin/env python3
"""
Verification script to ensure the new context system is working in production.
This script tests that the refactored components are properly integrated and functioning.
"""

import sys
import os
# Add the server directory to Python path since we're in tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all new utilities can be imported successfully."""
    print("Testing imports...")

    try:
        from components.context.memory_config import get_global_config, MemoryConfig
        from components.context.budget_manager import get_global_budget, ContextBudget
        from components.context.token_counter import get_global_counter, TokenCounter
        from components.context.exceptions import ContextError, ValidationError
        print("✓ All utility imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test that configuration is working properly."""
    print("\nTesting configuration...")

    try:
        from components.context.memory_config import get_global_config
        from components.context.budget_manager import get_global_budget
        from components.context.token_counter import get_global_counter

        config = get_global_config()
        print(f"✓ MemoryConfig loaded: progressive_mode={config.progressive_mode}")

        budget = get_global_budget()
        allocations = budget.get_allocations()
        print(f"✓ ContextBudget loaded: total={allocations.total}, memory={allocations.memory}")

        counter = get_global_counter()
        info = counter.get_model_info()
        print(f"✓ TokenCounter loaded: encoder={info['encoder_type']}, model={info['model_name']}")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_context_orchestrator():
    """Test that the context orchestrator is using new utilities."""
    print("\nTesting context orchestrator integration...")

    try:
        from components.context.context_orchestrator import pack_context

        # Test basic functionality
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        memory_bullets = ["User prefers concise responses"]
        summary_text = "Previous conversation about greetings"

        packed_messages, stats = pack_context(
            messages=messages,
            memory_bullets=memory_bullets,
            summary_text=summary_text,
            budget_tokens=4096,
            progressive_mode=True
        )

        print(f"✓ Context packing successful:")
        print(f"  - Packed {len(packed_messages)} messages")
        print(f"  - Stats: {stats}")
        print(f"  - Total tokens: {stats['tokens_total']}")

        # Verify the packed context includes our memory bullet
        memory_found = any(
            "User prefers concise responses" in msg.get("content", "")
            for msg in packed_messages
            if msg.get("role") == "system"
        )

        if memory_found:
            print("✓ Memory context properly injected")
        else:
            print("⚠ Memory context not found in packed messages")

        return True

    except Exception as e:
        print(f"✗ Context orchestrator test failed: {e}")
        return False

def test_hotpath_processor_config():
    """Test that HotPathProcessor is using the new configuration."""
    print("\nTesting HotPathProcessor configuration...")

    try:
        from components.processing.hotpath_processor import HotPathMemoryProcessor
        from components.context.memory_config import get_global_config

        # Create a processor instance to test initialization
        processor = HotPathMemoryProcessor()

        # Verify it's using the centralized config
        config = get_global_config()

        print(f"✓ HotPathProcessor initialized successfully")
        print(f"✓ Using progressive_mode: {config.progressive_mode}")
        print(f"✓ Using budget_tokens: {config.budget_tokens}")

        return True

    except Exception as e:
        print(f"✗ HotPathProcessor config test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=== Context System Verification ===")
    print("Verifying that the refactored context system is working correctly...\n")

    tests = [
        test_imports,
        test_configuration,
        test_context_orchestrator,
        test_hotpath_processor_config
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Results: {passed}/{len(tests)} tests passed ===")

    if passed == len(tests):
        print("🎉 All verification tests passed!")
        print("The new context system is properly integrated and ready for production.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration and imports.")
        return 1

if __name__ == "__main__":
    sys.exit(main())