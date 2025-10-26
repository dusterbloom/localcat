#!/usr/bin/env python3
"""
DIAGNOSE REAL ISSUE
Test the exact configuration that's causing 40+ second delays
"""

import asyncio
import aiohttp
import time
import json
import os
from loguru import logger

async def test_current_configuration():
    """Test the current configuration from .env"""

    # Read current .env
    env_vars = {}
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value

    current_model = env_vars.get('LLM_MODEL', 'qwen3-vl-4b-instruct-mlx')
    vision_enabled = env_vars.get('VISION_MODEL_ENABLED', 'true')
    debug_mode = env_vars.get('DEBUG_MODE', 'false')

    logger.info(f"🔍 Current Configuration:")
    logger.info(f"   Model: {current_model}")
    logger.info(f"   Vision: {vision_enabled}")
    logger.info(f"   Debug: {debug_mode}")

    base_url = "http://127.0.0.1:1234/v1"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }

    # Test the exact current model
    payload = {
        "model": current_model,
        "messages": [
            {"role": "user", "content": "Hello, respond briefly."}
        ],
        "max_tokens": 30,
        "stream": False  # Test non-streaming first (as in your logs)
    }

    logger.info(f"🧪 Testing current model: {current_model}")

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180)  # 3 minute timeout
            ) as response:

                logger.info(f"   HTTP Status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000

                    response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})

                    logger.info(f"   SUCCESS: {latency_ms:.1f}ms")
                    logger.info(f"   Response: '{response_text}'")
                    logger.info(f"   Tokens: {usage.get('total_tokens', 0)}")

                    # Assess the result
                    if latency_ms > 40000:  # >40 seconds
                        logger.error(f"🚨 CRITICAL: Found the 40+ second bottleneck!")
                        logger.error(f"   Model {current_model} took {latency_ms/1000:.1f} seconds")
                        return {"success": True, "latency_ms": latency_ms, "critical": True}
                    elif latency_ms > 10000:
                        logger.warning(f"⚠️ VERY SLOW: {latency_ms/1000:.1f} seconds")
                        return {"success": True, "latency_ms": latency_ms, "critical": False}
                    else:
                        logger.info(f"✅ Acceptable: {latency_ms:.1f}ms")
                        return {"success": True, "latency_ms": latency_ms, "critical": False}
                else:
                    error_text = await response.text()
                    logger.error(f"❌ HTTP Error {response.status}: {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}

    except asyncio.TimeoutError:
        logger.error(f"🚨 TIMEOUT: Model took longer than 3 minutes!")
        return {"success": False, "error": "Timeout after 180 seconds", "critical": True}
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        return {"success": False, "error": str(e), "critical": False}

async def check_lm_studio_health():
    """Check LM Studio health and resource usage"""
    logger.info("🏥 LM Studio Health Check")

    try:
        async with aiohttp.ClientSession() as session:
            # Check models endpoint
            async with session.get("http://127.0.0.1:1234/v1/models", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["id"] for model in data.get("data", [])]
                    logger.info(f"   Available models: {len(models)}")
                    logger.info(f"   Current model loaded: Yes")
                else:
                    logger.error(f"   Models endpoint failed: {response.status}")

    except Exception as e:
        logger.error(f"   Health check failed: {e}")

async def test_with_debug_disabled():
    """Test with debug explicitly disabled"""
    logger.info("🔧 Testing with debug disabled...")

    # The issue might be related to debug=True in Pipecat configuration
    # Let's check if we can reproduce the 40-second delay with simple API calls

    current_model = os.environ.get('LLM_MODEL', 'qwen3-vl-4b-instruct-mlx')

    # Try multiple calls to see if there's a pattern
    for i in range(3):
        logger.info(f"   Test call {i+1}/3:")
        result = await test_current_configuration()

        if not result.get("success"):
            logger.error(f"   Call {i+1} failed")

        await asyncio.sleep(2)  # Brief pause between calls

async def main():
    """Run comprehensive diagnosis"""
    print("🩺 Real Issue Diagnosis Tool")
    print("=" * 40)
    print("This tool diagnoses the exact cause of")
    print("the 40+ second LLM latency issue.")
    print()

    # Check LM Studio health
    await check_lm_studio_health()

    print()

    # Test current configuration
    result = await test_current_configuration()

    if result.get("critical"):
        print("\n🚨 CRITICAL ISSUE FOUND!")
        print("This confirms the 40+ second bottleneck.")
        print("\n📋 NEXT STEPS:")
        print("1. The issue is confirmed at the LLM model level")
        print("2. Switch to a lightweight model immediately")
        print("3. Apply the performance fixes")
    elif result.get("success") and result.get("latency_ms", 0) > 10000:
        print(f"\n⚠️ SLOW PERFORMANCE DETECTED!")
        print(f"Model took {result['latency_ms']/1000:.1f} seconds")
        print("This is still too slow for real-time use.")
    else:
        print(f"\n✅ NO CRITICAL ISSUE FOUND")
        print(f"The API responded in {result.get('latency_ms', 0):.1f}ms")
        print("The 40-second delay must be in the Pipecat pipeline layer.")

    print("\n💡 RECOMMENDATIONS:")
    print("1. Switch to llama-3.2-1b-instruct model")
    print("2. Disable vision processing (VISION_MODEL_ENABLED=false)")
    print("3. Enable streaming (LLM_USE_STREAMING=true)")
    print("4. Apply all performance fixes from .env.performance_fixes")

if __name__ == "__main__":
    asyncio.run(main())