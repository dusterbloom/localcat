#!/usr/bin/env python3
"""
TEST ANONYMOUS MODE FIXES
Quick test to verify that SOTA classifier and memory are properly disabled
"""

import sys
import time
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory

def test_anonymous_mode_fixes():
    """Test that anonymous mode properly disables heavy components"""
    print("🧪 Testing Anonymous Mode Fixes")
    print("=" * 50)

    # Set anonymous mode environment variables
    os.environ["MEMORY_ENABLED"] = "false"  # Force disable memory
    os.environ["HOTMEM_USE_SOTA_CLASSIFIER"] = "false"  # Already in .env

    try:
        # Step 1: Load configuration
        config = VoiceAgentConfig.from_env()
        print(f"✅ Configuration loaded")
        print(f"   Memory enabled: {getattr(config, 'memory_enabled', 'unknown')}")
        print(f"   Hotpath enabled: {getattr(config, 'hotpath_enabled', 'unknown')}")

        # Step 2: Create VoiceAgentFactory
        start_time = time.time()
        factory = VoiceAgentFactory(config)
        factory_time = (time.time() - start_time) * 1000
        print(f"✅ VoiceAgentFactory created in {factory_time:.1f}ms")

        # Step 3: Test intent service creation (should be None)
        start_time = time.time()
        intent_service = factory.create_intent_service()
        intent_time = (time.time() - start_time) * 1000
        print(f"✅ Intent service created in {intent_time:.1f}ms: {type(intent_service).__name__ if intent_service else 'None'}")

        # Step 4: Test LLM service creation
        start_time = time.time()
        llm_service = factory.create_llm_service()
        llm_time = (time.time() - start_time) * 1000
        print(f"✅ LLM service created in {llm_time:.1f}ms")

        # Step 5: Test memory service creation (should be minimal)
        start_time = time.time()
        memory_service = factory.create_hotmem_service()
        memory_time = (time.time() - start_time) * 1000
        print(f"✅ Memory service created in {memory_time:.1f}ms: {type(memory_service).__name__}")

        total_time = factory_time + intent_time + llm_time + memory_time
        print(f"\n📊 Total service creation time: {total_time:.1f}ms")

        if total_time < 100:
            print("🎉 EXCELLENT: All services created quickly (lightweight mode)")
        elif total_time < 500:
            print("✅ GOOD: Services created reasonably fast")
        else:
            print("⚠️  SLOW: Service creation still taking too long")

        # Test system prompt generation
        start_time = time.time()
        system_prompt = factory.build_system_prompt(skip_memory=True, camera_active=False)
        prompt_time = (time.time() - start_time) * 1000
        print(f"✅ System prompt generated in {prompt_time:.1f}ms")
        print(f"   Length: {len(system_prompt)} chars")
        print(f"   Contains 'Memory': {'Memory' in system_prompt}")
        print(f"   Contains 'memory': {'memory' in system_prompt}")

        return total_time

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

if __name__ == "__main__":
    test_anonymous_mode_fixes()