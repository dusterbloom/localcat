#!/usr/bin/env python3
"""
COMPLETE FIX VERIFICATION TEST
Test all performance fixes together: global ServiceFactory + SOTA disabled + prewarming
"""

import sys
import time
import asyncio
import httpx
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory

async def test_complete_fix():
    """Test complete performance fix implementation"""
    print("🚀 Testing Complete Performance Fix")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Load configuration (using current .env)
        config = VoiceAgentConfig.from_env()
        config_time = (time.time() - start_time) * 1000
        print(f"✅ Configuration loaded in {config_time:.1f}ms")

        # Step 2: Create VoiceAgentFactory (should use global ServiceFactory)
        factory_start = time.time()
        factory = VoiceAgentFactory(config)
        factory_time = (time.time() - factory_start) * 1000
        print(f"✅ VoiceAgentFactory created in {factory_time:.1f}ms")

        # Step 3: Create all services
        services_start = time.time()

        # LLM service (should be cached + prewarmed)
        llm_start = time.time()
        llm_service = factory.create_llm_service()
        llm_time = (time.time() - llm_start) * 1000
        print(f"✅ LLM service created in {llm_time:.1f}ms")

        # Intent service (should be None - SOTA disabled)
        intent_start = time.time()
        intent_service = factory.create_intent_service()
        intent_time = (time.time() - intent_start) * 1000
        print(f"✅ Intent service created in {intent_time:.1f}ms: {type(intent_service).__name__ if intent_service else 'None'}")

        # Memory service (should be lightweight)
        memory_start = time.time()
        memory_service = factory.create_hotmem_service()
        memory_time = (time.time() - memory_start) * 1000
        print(f"✅ Memory service created in {memory_time:.1f}ms")

        services_time = (time.time() - services_start) * 1000
        print(f"✅ All services created in {services_time:.1f}ms")

        # Step 4: Test system prompt generation (should be memory-free for anonymous)
        prompt_start = time.time()
        system_prompt = factory.build_system_prompt(skip_memory=True, camera_active=False)
        prompt_time = (time.time() - prompt_start) * 1000
        print(f"✅ System prompt generated in {prompt_time:.1f}ms")

        total_time = (time.time() - start_time) * 1000
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   Total startup time: {total_time:.1f}ms")
        print(f"   Service creation time: {services_time:.1f}ms")
        print(f"   LLM service: {llm_time:.1f}ms")
        print(f"   Intent service: {intent_time:.1f}ms (should be 0.0ms)")
        print(f"   Memory service: {memory_time:.1f}ms")

        # Performance assessment
        if total_time < 100:
            print("🎉 EXCELLENT: Ultra-fast startup (<100ms)")
        elif total_time < 500:
            print("✅ GOOD: Fast startup (<500ms)")
        elif total_time < 2000:
            print("⚠️  OK: Acceptable startup (<2s)")
        else:
            print("🔥 SLOW: Startup still too slow (>2s)")

        # Step 5: Test LLM responsiveness (if LM Studio is running)
        await test_llm_responsiveness(config)

        return total_time

    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        print(f"❌ Test failed after {error_time:.1f}ms: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

async def test_llm_responsiveness(config):
    """Test actual LLM responsiveness to verify prewarming worked"""
    print(f"\n🧠 Testing LLM Responsiveness")
    print("-" * 40)

    llm_config = config.get_component_config("llm")
    base_url = llm_config.get("base_url", "http://127.0.0.1:1234/v1")
    model = llm_config.get("model", "unknown")

    try:
        # Test actual LLM inference time
        test_url = f"{base_url}/chat/completions"
        test_data = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 5,
            "stream": False
        }

        print(f"Testing LM Studio at: {base_url}")
        print(f"Model: {model}")

        start_time = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                test_url,
                json=test_data,
                headers={"Authorization": f"Bearer {llm_config.get('api_key', 'not-needed')}"}
            )

        response_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ LLM response in {response_time:.1f}ms: '{content}'")

            if response_time < 1000:
                print("🎉 EXCELLENT: LLM is prewarmed and responsive (<1s)")
            elif response_time < 5000:
                print("✅ GOOD: LLM responding well (<5s)")
            elif response_time < 15000:
                print("⚠️  OK: LLM responding but slow (<15s)")
            else:
                print(f"🔥 CRITICAL: LLM still very slow ({response_time:.1f}ms)")
        else:
            print(f"❌ LLM test failed: HTTP {response.status_code}")

    except httpx.ConnectError:
        print("⚠️ LM Studio not running - start LM Studio for full testing")
    except Exception as e:
        print(f"❌ LLM test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_complete_fix())