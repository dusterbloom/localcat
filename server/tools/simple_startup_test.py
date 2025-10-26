#!/usr/bin/env python3
"""
SIMPLE STARTUP LATENCY TEST
Simple test to verify the 40-second LLM latency is eliminated
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory

def test_startup_latency():
    """Test bot startup latency to verify 40-second fix"""
    print("🚀 Testing Bot Startup Latency")
    print("=" * 50)

    start_time = time.time()

    try:
        # Step 1: Load configuration
        config_start = time.time()
        config = VoiceAgentConfig.from_env()
        config_time = (time.time() - config_start) * 1000
        print(f"✅ Configuration loaded in {config_time:.1f}ms")

        # Step 2: Create VoiceAgentFactory
        factory_start = time.time()
        factory = VoiceAgentFactory(config)
        factory_time = (time.time() - factory_start) * 1000
        print(f"✅ VoiceAgentFactory created in {factory_time:.1f}ms")

        # Step 3: Create LLM service (this was the bottleneck)
        llm_start = time.time()
        llm_service = factory.create_llm_service()
        llm_time = (time.time() - llm_start) * 1000
        print(f"✅ LLM service created in {llm_time:.1f}ms")

        # Step 4: Create second LLM service (should be instant)
        llm2_start = time.time()
        llm_service2 = factory.create_llm_service()
        llm2_time = (time.time() - llm2_start) * 1000
        print(f"✅ Second LLM service created in {llm2_time:.1f}ms")

        # Verify they're the same instance
        same_instance = llm_service is llm_service2
        print(f"✅ Same LLM instance reused: {same_instance}")

        total_time = (time.time() - start_time) * 1000
        print(f"\n📊 Total startup time: {total_time:.1f}ms")

        if total_time < 5000:  # Under 5 seconds
            print("🎉 SUCCESS: Startup latency is excellent!")
        elif total_time < 15000:  # Under 15 seconds
            print("✅ GOOD: Startup latency is reasonable")
        elif total_time < 40000:  # Under 40 seconds
            print("⚠️  OK: Startup latency improved but could be better")
        else:
            print("🔥 CRITICAL: Startup latency still too high!")

        return total_time, same_instance

    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        print(f"❌ Startup failed after {error_time:.1f}ms: {e}")
        return error_time, False

if __name__ == "__main__":
    test_startup_latency()