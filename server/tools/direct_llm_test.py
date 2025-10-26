#!/usr/bin/env python3
"""
DIRECT LLM API TEST
Tests LLM models directly via HTTP API to bypass Pipecat overhead
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Any
from loguru import logger

class DirectLLMTester:
    """Test LLM models directly via HTTP API"""

    def __init__(self):
        self.base_url = "http://127.0.0.1:1234/v1"
        self.api_key = "not-needed"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.test_message = "Hello, how are you today? Please respond briefly in one sentence."

    async def test_model_direct(self, model_name: str, timeout_seconds: int = 120) -> Dict[str, Any]:
        """Test a model directly via HTTP API"""
        logger.info(f"🧪 Testing model directly: {model_name}")

        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": self.test_message}
                ],
                "max_tokens": 50,
                "temperature": 0.7,
                "stream": False
            }

            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "model": model_name,
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "latency_ms": None,
                            "status": "❌ HTTP ERROR"
                        }

                    data = await response.json()
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000

                    # Extract response text
                    response_text = ""
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "message" in choice:
                            response_text = choice["message"].get("content", "")

                    result = {
                        "model": model_name,
                        "success": True,
                        "latency_ms": latency_ms,
                        "response": response_text[:100] if response_text else "No response",
                        "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                        "status": ""
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

    async def test_model_streaming(self, model_name: str, timeout_seconds: int = 120) -> Dict[str, Any]:
        """Test a model with streaming enabled"""
        logger.info(f"🧪 Testing model (streaming): {model_name}")

        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": self.test_message}
                ],
                "max_tokens": 50,
                "temperature": 0.7,
                "stream": True
            }

            start_time = time.time()
            first_chunk_time = None
            response_chunks = []

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "model": f"{model_name} (streaming)",
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "total_latency_ms": None,
                            "ttfb_ms": None,
                            "status": "❌ HTTP ERROR"
                        }

                    async for line in response.content:
                        if first_chunk_time is None:
                            first_chunk_time = time.time()

                        line = line.decode('utf-8').strip()
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if "choices" in data and data["choices"]:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        response_chunks.append(choice["delta"]["content"])
                            except json.JSONDecodeError:
                                continue

                    end_time = time.time()
                    total_latency_ms = (end_time - start_time) * 1000
                    ttfb_ms = (first_chunk_time - start_time) * 1000 if first_chunk_time else total_latency_ms

                    response_text = "".join(response_chunks)

                    result = {
                        "model": f"{model_name} (streaming)",
                        "success": True,
                        "total_latency_ms": total_latency_ms,
                        "ttfb_ms": ttfb_ms,
                        "response": response_text[:100] if response_text else "No response",
                        "chunks_received": len(response_chunks),
                        "status": ""
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
                "model": f"{model_name} (streaming)",
                "success": False,
                "error": f"Timeout after {timeout_seconds}s",
                "total_latency_ms": timeout_seconds * 1000,
                "ttfb_ms": None,
                "status": "🚨 TIMEOUT"
            }

        except Exception as e:
            logger.error(f"❌ {model_name} (streaming): {str(e)}")
            return {
                "model": f"{model_name} (streaming)",
                "success": False,
                "error": str(e),
                "total_latency_ms": None,
                "ttfb_ms": None,
                "status": "❌ ERROR"
            }

    async def run_comprehensive_test(self) -> List[Dict[str, Any]]:
        """Run comprehensive test of all available models"""
        logger.info("🔍 Starting comprehensive LLM model testing...")

        # First get available models
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models", timeout=10) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [model["id"] for model in models_data.get("data", [])]
                        logger.info(f"📋 Found {len(available_models)} available models")
                    else:
                        logger.error(f"❌ Failed to get models: HTTP {response.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ Failed to get models: {e}")
            return []

        # Test key models
        priority_models = [
            "llama-3.2-1b-instruct",      # Lightweight (fastest expected)
            "gemma-3n-e4b",               # Small MLX model
            "qwen3-vl-4b-instruct-mlx",  # Current problematic model
            "llama-3.2-3b-instruct",      # Medium model
            "qwen/qwen3-1.7b",           # Another lightweight option
        ]

        # Filter to only available models
        models_to_test = [m for m in priority_models if m in available_models]
        logger.info(f"🎯 Testing {len(models_to_test)} priority models: {models_to_test}")

        results = []

        for model in models_to_test:
            # Test non-streaming first
            result = await self.test_model_direct(model, timeout_seconds=120)
            results.append(result)

            # Test streaming if successful
            if result.get("success"):
                streaming_result = await self.test_model_streaming(model, timeout_seconds=120)
                results.append(streaming_result)

            # Brief pause between tests
            await asyncio.sleep(1)

        return results

    def print_results(self, results: List[Dict[str, Any]]):
        """Print comprehensive results comparison"""
        print("\n" + "="*80)
        print("🤖 DIRECT LLM API PERFORMANCE COMPARISON")
        print("="*80)

        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]

        print(f"\n📊 SUMMARY: {len(successful_results)} successful, {len(failed_results)} failed")

        if successful_results:
            print(f"\n✅ SUCCESSFUL TESTS:")
            print(f"{'Model':<35} {'Latency (ms)':<12} {'TTFB (ms)':<12} {'Status':<12} {'Tokens':<8}")
            print("-" * 85)

            # Sort by latency
            successful_results.sort(key=lambda x: x.get("total_latency_ms") or x.get("latency_ms", float('inf')))

            for result in successful_results:
                model = result["model"]
                if "total_latency_ms" in result:
                    latency = f"{result['total_latency_ms']:.1f}"
                    ttfb = f"{result.get('ttfb_ms', 0):.1f}" if result.get('ttfb_ms') else "N/A"
                else:
                    latency = f"{result['latency_ms']:.1f}"
                    ttfb = "N/A"

                status = result["status"]
                tokens = str(result.get("tokens_used", 0))
                print(f"{model:<35} {latency:<12} {ttfb:<12} {status:<12} {tokens:<8}")

                # Show response preview
                response = result.get("response", "")
                if response and len(response) > 0:
                    print(f"   Response: {response}...")

        if failed_results:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_results:
                model = result["model"]
                error = result.get("error", "Unknown error")
                status = result.get("status", "❌ ERROR")
                print(f"   {model}: {error} ({status})")

        # Analysis
        print(f"\n💡 ANALYSIS:")
        if successful_results:
            fastest = successful_results[0]
            slowest = successful_results[-1]

            fastest_latency = fastest.get("total_latency_ms") or fastest.get("latency_ms", 0)
            slowest_latency = slowest.get("total_latency_ms") or slowest.get("latency_ms", 0)

            print(f"   Fastest: {fastest['model']} ({fastest_latency:.1f}ms)")
            print(f"   Slowest: {slowest['model']} ({slowest_latency:.1f}ms)")

            # Check for problematic models
            problematic_models = []
            for result in successful_results:
                latency = result.get("total_latency_ms") or result.get("latency_ms", 0)
                if latency > 10000:  # >10 seconds
                    problematic_models.append(result)

            if problematic_models:
                print(f"\n🚨 CRITICAL ISSUES:")
                for result in problematic_models:
                    latency = result.get("total_latency_ms") or result.get("latency_ms", 0)
                    print(f"   {result['model']}: {latency/1000:.1f}s latency - UNSUITABLE for real-time!")

        print(f"\n🎯 RECOMMENDATIONS:")
        if successful_results:
            # Find best model under 1 second
            fast_models = [r for r in successful_results if (r.get("total_latency_ms") or r.get("latency_ms", 0)) < 1000]
            if fast_models:
                best = fast_models[0]
                print(f"   1. Use fastest model: {best['model']}")
                print(f"      Fix: export LLM_MODEL={best['model'].replace(' (streaming)', '')}")

        print(f"   2. Disable debug logging and verbose output")
        print(f"   3. Consider model pre-loading and caching")
        print(f"   4. Check LM Studio resource usage (CPU/RAM)")

        print("="*80)

        # Export results
        export_path = "/tmp/direct_llm_performance.json"
        with open(export_path, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "test_message": self.test_message,
                "results": results
            }, f, indent=2)

        print(f"📁 Results exported to: {export_path}")

async def main():
    """Run direct LLM API testing"""
    print("🤖 Direct LLM API Performance Test")
    print("=" * 40)
    print("This tool tests LLM models directly via HTTP API")
    print("to bypass Pipecat overhead and identify bottlenecks.")
    print()

    tester = DirectLLMTester()
    results = await tester.run_comprehensive_test()
    tester.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())