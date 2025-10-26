#!/usr/bin/env python3
"""
TEST PERFORMANCE FIXES
Apply and test the performance fixes for the 40-second LLM latency issue
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path

# Add server path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.latency_tracer import LatencyTracer
from loguru import logger

class PerformanceFixTester:
    """Test the effectiveness of performance fixes"""

    def __init__(self):
        self.tracer = LatencyTracer(enable_detailed_logging=True)
        self.original_env = {}

    def save_original_env(self):
        """Save original environment variables"""
        keys_to_save = [
            "LLM_MODEL", "VISION_MODEL_ENABLED", "DEBUG_MODE", "LOG_LEVEL",
            "LLM_USE_STREAMING", "MEMORY_ENABLED", "HOTMEM_USE_SOTA_CLASSIFIER"
        ]

        for key in keys_to_save:
            self.original_env[key] = os.environ.get(key, None)

    def restore_original_env(self):
        """Restore original environment variables"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def apply_performance_fixes(self):
        """Apply the performance fixes"""
        logger.info("🔧 Applying performance fixes...")

        # Load and apply fixes
        fixes_file = Path(__file__).parent.parent / ".env.performance_fixes"
        if fixes_file.exists():
            with open(fixes_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                        logger.info(f"   Applied: {key}={value}")

        logger.info("✅ Performance fixes applied")

    async def test_llm_direct(self, model_name: str) -> dict:
        """Test LLM performance directly"""
        import aiohttp

        base_url = "http://127.0.0.1:1234/v1"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer not-needed"
        }

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello, respond briefly."}
            ],
            "max_tokens": 30,
            "stream": os.environ.get("LLM_USE_STREAMING", "false").lower() == "true"
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        if payload["stream"]:
                            # Handle streaming response
                            response_text = ""
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: ') and line != 'data: [DONE]':
                                    try:
                                        data = json.loads(line[6:])
                                        if "choices" in data and data["choices"]:
                                            delta = data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                response_text += delta["content"]
                                    except:
                                        continue
                        else:
                            # Handle non-streaming response
                            data = await response.json()
                            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        latency_ms = (time.time() - start_time) * 1000

                        return {
                            "success": True,
                            "latency_ms": latency_ms,
                            "response": response_text[:100],
                            "model": model_name,
                            "streaming": payload["stream"]
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                            "model": model_name
                        }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model_name
            }

    async def run_comparison_test(self) -> dict:
        """Run before/after comparison test"""
        logger.info("🔍 Running performance comparison test...")

        results = {
            "timestamp": time.time(),
            "original_env": {},
            "fixed_env": {},
            "improvement": {}
        }

        # Test 1: Original configuration
        logger.info("\n📊 Test 1: Original Configuration")
        self.restore_original_env()
        results["original_env"] = {
            "model": os.environ.get("LLM_MODEL", "not_set"),
            "vision": os.environ.get("VISION_MODEL_ENABLED", "not_set"),
            "streaming": os.environ.get("LLM_USE_STREAMING", "not_set"),
            "debug": os.environ.get("DEBUG_MODE", "not_set")
        }

        original_model = os.environ.get("LLM_MODEL", "qwen3-vl-4b-instruct-mlx")

        with self.tracer.trace_sync("comparison", "original_test"):
            original_result = await self.test_llm_direct(original_model)
            results["original_env"]["result"] = original_result

        if original_result.get("success"):
            logger.info(f"   Original: {original_result['latency_ms']:.1f}ms")
        else:
            logger.error(f"   Original FAILED: {original_result.get('error')}")

        # Test 2: Fixed configuration
        logger.info("\n📊 Test 2: Fixed Configuration")
        self.apply_performance_fixes()

        results["fixed_env"] = {
            "model": os.environ.get("LLM_MODEL", "not_set"),
            "vision": os.environ.get("VISION_MODEL_ENABLED", "not_set"),
            "streaming": os.environ.get("LLM_USE_STREAMING", "not_set"),
            "debug": os.environ.get("DEBUG_MODE", "not_set")
        }

        fixed_model = os.environ.get("LLM_MODEL", "llama-3.2-1b-instruct")

        with self.tracer.trace_sync("comparison", "fixed_test"):
            fixed_result = await self.test_llm_direct(fixed_model)
            results["fixed_env"]["result"] = fixed_result

        if fixed_result.get("success"):
            logger.info(f"   Fixed: {fixed_result['latency_ms']:.1f}ms")
        else:
            logger.error(f"   Fixed FAILED: {fixed_result.get('error')}")

        # Calculate improvement
        if original_result.get("success") and fixed_result.get("success"):
            original_latency = original_result["latency_ms"]
            fixed_latency = fixed_result["latency_ms"]
            improvement_ms = original_latency - fixed_latency
            improvement_factor = original_latency / fixed_latency

            results["improvement"] = {
                "original_latency_ms": original_latency,
                "fixed_latency_ms": fixed_latency,
                "improvement_ms": improvement_ms,
                "improvement_factor": improvement_factor,
                "improvement_percent": (improvement_ms / original_latency) * 100
            }

            logger.info(f"\n🎯 IMPROVEMENT:")
            logger.info(f"   Latency reduced by {improvement_ms:.1f}ms ({results['improvement']['improvement_percent']:.1f}%)")
            logger.info(f"   Speed improvement: {improvement_factor:.1f}x faster")

        return results

    def print_summary(self, results: dict):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("🔧 PERFORMANCE FIXES VALIDATION")
        print("="*80)

        print(f"\n⚙️ CONFIGURATION CHANGES:")

        original = results.get("original_env", {})
        fixed = results.get("fixed_env", {})

        print(f"   Model: {original.get('model', 'N/A')} → {fixed.get('model', 'N/A')}")
        print(f"   Vision: {original.get('vision', 'N/A')} → {fixed.get('vision', 'N/A')}")
        print(f"   Streaming: {original.get('streaming', 'N/A')} → {fixed.get('streaming', 'N/A')}")
        print(f"   Debug: {original.get('debug', 'N/A')} → {fixed.get('debug', 'N/A')}")

        print(f"\n📊 PERFORMANCE RESULTS:")

        orig_result = original.get("result", {})
        fixed_result = fixed.get("result", {})

        if orig_result.get("success"):
            print(f"   Original: {orig_result['latency_ms']:.1f}ms ({orig_result.get('model', 'unknown')})")
        else:
            print(f"   Original: FAILED - {orig_result.get('error', 'unknown')}")

        if fixed_result.get("success"):
            print(f"   Fixed: {fixed_result['latency_ms']:.1f}ms ({fixed_result.get('model', 'unknown')})")
        else:
            print(f"   Fixed: FAILED - {fixed_result.get('error', 'unknown')}")

        improvement = results.get("improvement", {})
        if improvement:
            print(f"\n🚀 IMPROVEMENT:")
            print(f"   Latency reduction: {improvement['improvement_ms']:.1f}ms ({improvement['improvement_percent']:.1f}%)")
            print(f"   Speed improvement: {improvement['improvement_factor']:.1f}x faster")

            # Status assessment
            if improvement['fixed_latency_ms'] < 250:
                print(f"   Status: ✅ EXCELLENT - Under 250ms target!")
            elif improvement['fixed_latency_ms'] < 1000:
                print(f"   Status: ✅ GOOD - Under 1 second")
            elif improvement['fixed_latency_ms'] < 5000:
                print(f"   Status: ⚠️ ACCEPTABLE - Under 5 seconds")
            else:
                print(f"   Status: 🔥 STILL SLOW - {improvement['fixed_latency_ms']/1000:.1f}s")

        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Apply the performance fixes to your main .env file")
        print(f"   2. Test the full voice pipeline end-to-end")
        print(f"   3. Monitor with the latency observer during real usage")
        print(f"   4. Consider additional optimizations if needed")

        print("="*80)

    async def run_test(self):
        """Run the complete performance test"""
        print("🔧 Performance Fixes Validation Tool")
        print("=" * 40)
        print("This tool applies and validates performance fixes")
        print("for the 40+ second LLM latency issue.")
        print()

        # Save original environment
        self.save_original_env()

        try:
            # Run comparison test
            results = await self.run_comparison_test()
            self.print_summary(results)

            # Export results
            export_path = "/tmp/performance_fixes_validation.json"
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Results exported to: {export_path}")

        finally:
            # Restore original environment
            self.restore_original_env()
            logger.info("🔄 Original environment restored")

async def main():
    """Run performance fixes validation"""
    tester = PerformanceFixTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())