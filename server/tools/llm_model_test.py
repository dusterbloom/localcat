#!/usr/bin/env python3
"""
LLM MODEL PERFORMANCE TEST
Directly tests different LLM models to identify the 40-second latency bottleneck
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path

# Add server path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipecat.services.openai.llm import OpenAILLMService
from loguru import logger

class LLMModelTester:
    """Test LLM model performance directly"""

    def __init__(self):
        self.base_url = "http://127.0.0.1:1234/v1"
        self.api_key = "not-needed"
        self.test_message = "Hello, how are you today? Please respond briefly."

    async def test_model(self, model_name: str, timeout_seconds: int = 60) -> dict:
        """Test a specific model's performance"""
        logger.info(f"🧪 Testing model: {model_name}")

        try:
            # Create LLM service
            llm_service = OpenAILLMService(
                api_key=self.api_key,
                model=model_name,
                base_url=self.base_url,
                max_tokens=50,  # Limit response length
                stream=False,   # Test non-streaming first
                debug=False     # No debug overhead
            )

            start_time = time.time()

            # Test context
            from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
            context = OpenAILLMContext()
            context.add_message({"role": "user", "content": self.test_message})

            # Time the LLM call
            response_text = None
            async with asyncio.timeout(timeout_seconds):
                async for chunk in llm_service.run_llm(context):
                    if hasattr(chunk, 'content'):
                        response_text = chunk.content
                        break

            if response_text is None:
                # Try to get response differently
                response_chunks = []
                async for chunk in llm_service.run_llm(context):
                    if hasattr(chunk, 'content'):
                        response_chunks.append(chunk.content)
                response_text = "".join(response_chunks)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            result = {
                "model": model_name,
                "success": True,
                "latency_ms": latency_ms,
                "response": response_text[:100] if response_text else "No response",
                "timeout_used_s": timeout_seconds
            }

            # Status classification
            if latency_ms < 250:
                result["status"] = "✅ EXCELLENT"
            elif latency_ms < 1000:
                result["status"] = "⚠️ ACCEPTABLE"
            elif latency_ms < 10000:
                result["status"] = "🔥 SLOW"
            else:
                result["status"] = "🚨 CRITICAL"

            logger.info(f"✅ {model_name}: {latency_ms:.1f}ms {result['status']}")

            return result

        except asyncio.TimeoutError:
            logger.error(f"❌ {model_name}: TIMEOUT after {timeout_seconds}s")
            return {
                "model": model_name,
                "success": False,
                "error": f"Timeout after {timeout_seconds}s",
                "latency_ms": timeout_seconds * 1000,
                "status": "🚨 TIMEOUT"
            }

        except Exception as e:
            logger.error(f"❌ {model_name}: {str(e)}")
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "latency_ms": None,
                "status": "❌ ERROR"
            }

    async def test_models_comparison(self) -> list:
        """Test multiple models and compare performance"""
        logger.info("🔍 Starting LLM model performance comparison...")

        # Models to test (from your available models)
        models_to_test = [
            "llama-3.2-1b-instruct",      # Lightweight (fastest expected)
            "gemma-3n-e4b",               # Small MLX model
            "qwen3-vl-4b-instruct-mlx",  # Current problematic model
            "llama-3.2-3b-instruct",      # Medium model
            "qwen/qwen3-1.7b",           # Another lightweight option
        ]

        results = []

        for model in models_to_test:
            result = await self.test_model(model, timeout_seconds=120)  # 2 minute timeout
            results.append(result)

            # If this is the problematic model, test with streaming too
            if model == "qwen3-vl-4b-instruct-mlx" and result.get("success"):
                logger.info(f"🔄 Testing {model} with streaming...")
                streaming_result = await self.test_model_streaming(model, timeout_seconds=120)
                streaming_result["model"] = f"{model} (streaming)"
                results.append(streaming_result)

        return results

    async def test_model_streaming(self, model_name: str, timeout_seconds: int = 60) -> dict:
        """Test a model with streaming enabled"""
        logger.info(f"🧪 Testing model (streaming): {model_name}")

        try:
            # Create LLM service with streaming
            llm_service = OpenAILLMService(
                api_key=self.api_key,
                model=model_name,
                base_url=self.base_url,
                max_tokens=50,
                stream=True,    # Enable streaming
                debug=False
            )

            start_time = time.time()

            # Test context
            from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
            context = OpenAILLMContext()
            context.add_message({"role": "user", "content": self.test_message})

            # Time the streaming LLM call
            response_chunks = []
            first_chunk_time = None

            async with asyncio.timeout(timeout_seconds):
                async for chunk in llm_service.run_llm(context):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()

                    if hasattr(chunk, 'content'):
                        response_chunks.append(chunk.content)

            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            ttfb_ms = (first_chunk_time - start_time) * 1000 if first_chunk_time else total_latency_ms

            response_text = "".join(response_chunks)

            result = {
                "model": model_name,
                "success": True,
                "total_latency_ms": total_latency_ms,
                "ttfb_ms": ttfb_ms,
                "response": response_text[:100] if response_text else "No response",
                "streaming": True,
                "timeout_used_s": timeout_seconds
            }

            # Status classification
            if total_latency_ms < 250:
                result["status"] = "✅ EXCELLENT"
            elif total_latency_ms < 1000:
                result["status"] = "⚠️ ACCEPTABLE"
            elif total_latency_ms < 10000:
                result["status"] = "🔥 SLOW"
            else:
                result["status"] = "🚨 CRITICAL"

            logger.info(f"✅ {model_name} (streaming): {total_latency_ms:.1f}ms total, {ttfb_ms:.1f}ms TTFB {result['status']}")

            return result

        except asyncio.TimeoutError:
            logger.error(f"❌ {model_name} (streaming): TIMEOUT after {timeout_seconds}s")
            return {
                "model": model_name,
                "success": False,
                "error": f"Timeout after {timeout_seconds}s",
                "total_latency_ms": timeout_seconds * 1000,
                "ttfb_ms": None,
                "status": "🚨 TIMEOUT"
            }

        except Exception as e:
            logger.error(f"❌ {model_name} (streaming): {str(e)}")
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "total_latency_ms": None,
                "ttfb_ms": None,
                "status": "❌ ERROR"
            }

    def print_results(self, results: list):
        """Print comprehensive results comparison"""
        print("\n" + "="*80)
        print("🤖 LLM MODEL PERFORMANCE COMPARISON")
        print("="*80)

        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]

        print(f"\n📊 SUMMARY: {len(successful_results)} successful, {len(failed_results)} failed")

        # Sort successful results by latency
        successful_results.sort(key=lambda x: x.get("total_latency_ms") or x.get("latency_ms", float('inf')))

        if successful_results:
            print(f"\n✅ SUCCESSFUL TESTS:")
            print(f"{'Model':<35} {'Latency (ms)':<12} {'TTFB (ms)':<12} {'Status':<12}")
            print("-" * 75)

            for result in successful_results:
                model = result["model"]
                if "total_latency_ms" in result:
                    latency = f"{result['total_latency_ms']:.1f}"
                    ttfb = f"{result.get('ttfb_ms', 0):.1f}" if result.get('ttfb_ms') else "N/A"
                else:
                    latency = f"{result['latency_ms']:.1f}"
                    ttfb = "N/A"

                status = result["status"]
                print(f"{model:<35} {latency:<12} {ttfb:<12} {status:<12}")

                # Show response preview
                response = result.get("response", "")
                if response:
                    print(f"   Response: {response}...")

        if failed_results:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_results:
                model = result["model"]
                error = result.get("error", "Unknown error")
                status = result.get("status", "❌ ERROR")
                print(f"   {model}: {error} ({status})")

        # Analysis and recommendations
        print(f"\n💡 ANALYSIS:")

        if successful_results:
            fastest = successful_results[0]
            slowest = successful_results[-1]

            print(f"   Fastest model: {fastest['model']} ({(fastest.get('total_latency_ms') or fastest['latency_ms']):.1f}ms)")
            print(f"   Slowest model: {slowest['model']} ({(slowest.get('total_latency_ms') or slowest['latency_ms']):.1f}ms)")

            # Check for problematic models
            for result in successful_results:
                latency = result.get("total_latency_ms") or result.get("latency_ms", 0)
                if latency > 10000:  # >10 seconds
                    print(f"   🚨 CRITICAL: {result['model']} has {latency/1000:.1f}s latency - UNSUITABLE for real-time!")

                if "vl" in result["model"].lower() or "vision" in result["model"].lower():
                    if latency > 2000:  # >2 seconds
                        print(f"   ⚠️ Vision model {result['model']} is slow ({latency:.1f}ms) - consider text-only model")

        print(f"\n🎯 RECOMMENDATIONS:")
        if successful_results:
            fastest = successful_results[0]
            if (fastest.get("total_latency_ms") or fastest["latency_ms"]) < 1000:
                print(f"   1. Use fastest model: {fastest['model']}")
                print(f"      Fix: export LLM_MODEL={fastest['model']}")

        print(f"   2. Disable debug logging in production")
        print(f"   3. Consider model pre-loading and caching")
        print(f"   4. Test with your actual conversation patterns")

        print("="*80)

        # Export results
        export_path = "/tmp/llm_model_performance.json"
        with open(export_path, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "test_message": self.test_message,
                "results": results
            }, f, indent=2)

        print(f"📁 Results exported to: {export_path}")

async def main():
    """Run LLM model performance testing"""
    print("🤖 LLM Model Performance Test")
    print("=" * 40)
    print("This tool tests different LLM models to identify")
    print("the source of the 40+ second latency issue.")
    print()

    tester = LLMModelTester()
    results = await tester.test_models_comparison()
    tester.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())