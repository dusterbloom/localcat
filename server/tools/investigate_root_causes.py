#!/usr/bin/env python3
"""
INVESTIGATE ROOT CAUSES
Test the three hypotheses: model loading, anonymous mode configuration, and vision processing
"""

import asyncio
import aiohttp
import time
import json
import os
import sys
from pathlib import Path
from loguru import logger

# Add server path for imports
sys.path.append(str(Path(__file__).parent.parent))

class RootCauseInvestigator:
    """Investigate the three root cause hypotheses"""

    def __init__(self):
        self.base_url = "http://127.0.0.1:1234/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer not-needed"
        }

    async def test_model_loading_time(self, model_name: str) -> dict:
        """Test if 40+ seconds is model loading time"""
        logger.info(f"🧪 Hypothesis 1: Testing model loading time for {model_name}")

        # Test 1: Cold start (first call)
        logger.info("   Test 1: Cold start (first call)")
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10,
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        cold_start_time = (time.time() - start_time) * 1000
                        logger.info(f"   Cold start: {cold_start_time:.1f}ms")

                        # Test 2: Warm start (subsequent calls)
                        logger.info("   Test 2: Warm start (subsequent calls)")
                        warm_times = []

                        for i in range(3):
                            start_time = time.time()
                            async with session.post(
                                f"{self.base_url}/chat/completions",
                                headers=self.headers,
                                json={
                                    "model": model_name,
                                    "messages": [{"role": "user", "content": "Hello"}],
                                    "max_tokens": 10,
                                    "stream": False
                                },
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                if response.status == 200:
                                    warm_time = (time.time() - start_time) * 1000
                                    warm_times.append(warm_time)
                                    logger.info(f"   Warm start {i+1}: {warm_time:.1f}ms")
                                await asyncio.sleep(0.5)

                        avg_warm_time = sum(warm_times) / len(warm_times) if warm_times else 0

                        # Analysis
                        if cold_start_time > 40000:  # >40 seconds
                            logger.error(f"🚨 CONFIRMED: Model loading takes {cold_start_time/1000:.1f}s!")
                            return {
                                "hypothesis_confirmed": True,
                                "cold_start_ms": cold_start_time,
                                "avg_warm_ms": avg_warm_time,
                                "analysis": "Model loading is the bottleneck"
                            }
                        elif cold_start_time > avg_warm_time * 10:
                            logger.warning(f"⚠️ Model loading takes {cold_start_time:.1f}ms vs {avg_warm_time:.1f}ms warm")
                            return {
                                "hypothesis_confirmed": True,
                                "cold_start_ms": cold_start_time,
                                "avg_warm_ms": avg_warm_time,
                                "analysis": "Model loading contributes significantly"
                            }
                        else:
                            logger.info(f"✅ Model loading is fast: {cold_start_time:.1f}ms")
                            return {
                                "hypothesis_confirmed": False,
                                "cold_start_ms": cold_start_time,
                                "avg_warm_ms": avg_warm_time,
                                "analysis": "Model loading is not the issue"
                            }
                    else:
                        return {"hypothesis_confirmed": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"   Model loading test failed: {e}")
            return {"hypothesis_confirmed": False, "error": str(e)}

    def check_anonymous_mode_config(self) -> dict:
        """Check if hotmem and SOTA classifier are properly disabled in anonymous mode"""
        logger.info("🧪 Hypothesis 2: Checking anonymous mode configuration")

        # Read current .env
        env_vars = {}
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value

        # Check anonymous mode settings
        checks = {
            "MEMORY_ENABLED": env_vars.get("MEMORY_ENABLED", "true").lower() == "true",
            "HOTMEM_USE_SOTA_CLASSIFIER": env_vars.get("HOTMEM_USE_SOTA_CLASSIFIER", "true").lower() == "true",
            "ANONYMOUS_MODE": env_vars.get("ANONYMOUS_MODE", "false").lower() == "true"
        }

        logger.info(f"   MEMORY_ENABLED: {env_vars.get('MEMORY_ENABLED', 'not_set')}")
        logger.info(f"   HOTMEM_USE_SOTA_CLASSIFIER: {env_vars.get('HOTMEM_USE_SOTA_CLASSIFIER', 'not_set')}")
        logger.info(f"   ANONYMOUS_MODE: {env_vars.get('ANONYMOUS_MODE', 'not_set')}")

        issues = []
        if checks["MEMORY_ENABLED"]:
            issues.append("Memory system enabled in anonymous mode")
        if checks["HOTMEM_USE_SOTA_CLASSIFIER"]:
            issues.append("SOTA classifier enabled in anonymous mode")

        # Check service factory code for anonymous mode handling
        try:
            from core.memory.anonymous_context import AnonymousAwareContextAggregator
            logger.info("   ✅ AnonymousAwareContextAggregator available")

            # Check if it properly disables memory in anonymous mode
            # This would require checking the actual implementation

        except ImportError:
            issues.append("AnonymousAwareContextAggregator not available")

        if issues:
            logger.error("🚨 ANONYMOUS MODE ISSUES FOUND:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return {
                "hypothesis_confirmed": True,
                "issues": issues,
                "env_vars": env_vars
            }
        else:
            logger.info("✅ Anonymous mode appears properly configured")
            return {
                "hypothesis_confirmed": False,
                "issues": [],
                "env_vars": env_vars
            }

    async def test_vision_processing_behavior(self) -> dict:
        """Test if vision processing happens when camera is off"""
        logger.info("🧪 Hypothesis 3: Testing vision processing behavior")

        current_model = os.environ.get('LLM_MODEL', 'qwen3-vl-4b-instruct-mlx')
        vision_enabled = os.environ.get('VISION_MODEL_ENABLED', 'true').lower() == 'true'

        logger.info(f"   Current model: {current_model}")
        logger.info(f"   VISION_MODEL_ENABLED: {vision_enabled}")

        if not vision_enabled:
            logger.info("✅ Vision is disabled - no unnecessary processing")
            return {"hypothesis_confirmed": False, "vision_disabled": True}

        # Test 1: Text-only request to vision model
        logger.info("   Test 1: Text-only request to vision model")
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": current_model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10,
                        "stream": False
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        text_only_time = (time.time() - start_time) * 1000
                        logger.info(f"   Text-only request: {text_only_time:.1f}ms")

                        # Test 2: Request with tiny image
                        logger.info("   Test 2: Request with tiny image")
                        start_time = time.time()

                        tiny_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

                        async with session.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json={
                                "model": current_model,
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": "What do you see?"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tiny_image_b64}"}}
                                ]}],
                                "max_tokens": 10,
                                "stream": False
                            },
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                with_image_time = (time.time() - start_time) * 1000
                                logger.info(f"   With tiny image: {with_image_time:.1f}ms")

                                # Analysis
                                vision_overhead = with_image_time - text_only_time
                                logger.info(f"   Vision overhead: {vision_overhead:.1f}ms")

                                if vision_overhead > 1000:  # >1 second overhead
                                    logger.warning(f"⚠️ High vision overhead: {vision_overhead:.1f}ms")
                                    return {
                                        "hypothesis_confirmed": True,
                                        "text_only_ms": text_only_time,
                                        "with_image_ms": with_image_time,
                                        "vision_overhead_ms": vision_overhead,
                                        "analysis": "Vision processing adds significant overhead"
                                    }
                                else:
                                    logger.info("✅ Vision overhead is reasonable")
                                    return {
                                        "hypothesis_confirmed": False,
                                        "text_only_ms": text_only_time,
                                        "with_image_ms": with_image_time,
                                        "vision_overhead_ms": vision_overhead,
                                        "analysis": "Vision overhead is acceptable"
                                    }
                            else:
                                return {"hypothesis_confirmed": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"   Vision processing test failed: {e}")
            return {"hypothesis_confirmed": False, "error": str(e)}

    async def test_vision_keyword_filtering(self) -> dict:
        """Test if vision processing only happens with vision keywords"""
        logger.info("🧪 Additional Test: Vision keyword filtering")

        current_model = os.environ.get('LLM_MODEL', '')

        if 'vl' not in current_model.lower() and 'vision' not in current_model.lower():
            logger.info("✅ Non-vision model - no filtering needed")
            return {"hypothesis_confirmed": False, "non_vision_model": True}

        # Check vision keywords from .env
        env_vars = {}
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = key

        vision_keywords = env_vars.get('VISION_KEYWORDS', '')
        vision_keyword_filter = env_vars.get('VISION_KEYWORD_FILTER', 'false').lower() == 'true'

        logger.info(f"   VISION_KEYWORD_FILTER: {vision_keyword_filter}")
        logger.info(f"   VISION_KEYWORDS: {vision_keywords}")

        if not vision_keyword_filter:
            logger.warning("⚠️ Vision keyword filtering is disabled")
            return {"hypothesis_confirmed": True, "filtering_disabled": True}
        else:
            logger.info("✅ Vision keyword filtering is enabled")
            return {"hypothesis_confirmed": False, "filtering_enabled": True}

    async def run_investigation(self) -> dict:
        """Run complete root cause investigation"""
        logger.info("🔍 Starting Root Cause Investigation")
        logger.info("=" * 50)

        results = {
            "timestamp": time.time(),
            "hypotheses": {}
        }

        # Hypothesis 1: Model loading time
        current_model = os.environ.get('LLM_MODEL', 'qwen3-vl-4b-instruct-mlx')
        results["hypotheses"]["model_loading"] = await self.test_model_loading_time(current_model)

        print()

        # Hypothesis 2: Anonymous mode configuration
        results["hypotheses"]["anonymous_mode"] = self.check_anonymous_mode_config()

        print()

        # Hypothesis 3: Vision processing behavior
        results["hypotheses"]["vision_processing"] = await self.test_vision_processing_behavior()

        print()

        # Additional test: Vision keyword filtering
        results["hypotheses"]["vision_filtering"] = await self.test_vision_keyword_filtering()

        return results

    def print_summary(self, results: dict):
        """Print investigation summary"""
        print("\n" + "="*80)
        print("🔍 ROOT CAUSE INVESTIGATION SUMMARY")
        print("="*80)

        hypotheses = results.get("hypotheses", {})

        print(f"\n📊 HYPOTHESIS RESULTS:")

        # Hypothesis 1: Model Loading
        h1 = hypotheses.get("model_loading", {})
        if h1.get("hypothesis_confirmed"):
            print(f"   🔥 MODEL LOADING: CONFIRMED")
            print(f"      Cold start: {h1.get('cold_start_ms', 0):.1f}ms")
            print(f"      Warm start: {h1.get('avg_warm_ms', 0):.1f}ms")
            print(f"      Analysis: {h1.get('analysis', '')}")
        else:
            print(f"   ✅ MODEL LOADING: Not the issue")

        # Hypothesis 2: Anonymous Mode
        h2 = hypotheses.get("anonymous_mode", {})
        if h2.get("hypothesis_confirmed"):
            print(f"   🔥 ANONYMOUS MODE: ISSUES FOUND")
            for issue in h2.get("issues", []):
                print(f"      - {issue}")
        else:
            print(f"   ✅ ANONYMOUS MODE: Properly configured")

        # Hypothesis 3: Vision Processing
        h3 = hypotheses.get("vision_processing", {})
        if h3.get("hypothesis_confirmed"):
            print(f"   🔥 VISION PROCESSING: HIGH OVERHEAD")
            print(f"      Text-only: {h3.get('text_only_ms', 0):.1f}ms")
            print(f"      With image: {h3.get('with_image_ms', 0):.1f}ms")
            print(f"      Overhead: {h3.get('vision_overhead_ms', 0):.1f}ms")
        else:
            print(f"   ✅ VISION PROCESSING: Acceptable overhead")

        # Vision Filtering
        h4 = hypotheses.get("vision_filtering", {})
        if h4.get("hypothesis_confirmed"):
            if h4.get("filtering_disabled"):
                print(f"   ⚠️ VISION FILTERING: DISABLED")
            else:
                print(f"   🔥 VISION FILTERING: ISSUES FOUND")
        else:
            print(f"   ✅ VISION FILTERING: Properly configured")

        # Overall assessment
        confirmed_issues = sum(1 for h in hypotheses.values() if h.get("hypothesis_confirmed"))

        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Confirmed issues: {confirmed_issues}/4")

        if confirmed_issues > 0:
            print(f"\n🚨 ROOT CAUSE(S) IDENTIFIED:")

            if hypotheses.get("model_loading", {}).get("hypothesis_confirmed"):
                print(f"   1. Model loading takes {hypotheses['model_loading']['cold_start_ms']/1000:.1f}s")
                print(f"      Fix: Use lighter model or enable model caching")

            if hypotheses.get("anonymous_mode", {}).get("hypothesis_confirmed"):
                print(f"   2. Anonymous mode has disabled components")
                print(f"      Fix: Ensure MEMORY_ENABLED=false and HOTMEM_USE_SOTA_CLASSIFIER=false")

            if hypotheses.get("vision_processing", {}).get("hypothesis_confirmed"):
                print(f"   3. Vision processing adds {hypotheses['vision_processing']['vision_overhead_ms']:.1f}ms")
                print(f"      Fix: Disable vision or use text-only model")

            if hypotheses.get("vision_filtering", {}).get("hypothesis_confirmed"):
                print(f"   4. Vision filtering not working")
                print(f"      Fix: Enable VISION_KEYWORD_FILTER=true")

        print(f"\n💡 RECOMMENDED FIXES:")
        print(f"   1. Switch to llama-3.2-1b-instruct (no vision)")
        print(f"   2. Ensure anonymous mode disables memory/SOTA")
        print(f"   3. Enable vision keyword filtering")
        print(f"   4. Apply all fixes from .env.performance_fixes")

        print("="*80)

async def main():
    """Run root cause investigation"""
    print("🔍 Root Cause Investigation Tool")
    print("=" * 40)
    print("Testing the three hypotheses:")
    print("1. 40+ seconds = model loading time")
    print("2. Hotmem/SOTA not disabled in anonymous mode")
    print("3. Vision processing when camera is off")
    print()

    investigator = RootCauseInvestigator()
    results = await investigator.run_investigation()
    investigator.print_summary(results)

    # Export results
    import json
    export_path = "/tmp/root_cause_investigation.json"
    with open(export_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Results exported to: {export_path}")

if __name__ == "__main__":
    asyncio.run(main())