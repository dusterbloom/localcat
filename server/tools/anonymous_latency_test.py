#!/usr/bin/env python3
"""
ANONYMOUS MODE LATENCY DIAGNOSTIC TEST
Tests voice pipeline performance in anonymous mode to isolate the 40-second LLM bottleneck
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add server path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.latency_tracer import LatencyTracer, get_tracer
from config import VoiceAgentConfig
from core.factories.service_factory import ServiceFactory
from pipecat.frames.frames import (
    AudioRawFrame,
    TranscriptionFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame
)
import numpy as np
from loguru import logger

class AnonymousLatencyTester:
    """Test voice pipeline latency in anonymous mode (minimal memory overhead)"""

    def __init__(self):
        self.tracer = LatencyTracer(enable_detailed_logging=True)
        self.config = None
        self.factory = None
        self.services = {}

        # Test parameters
        self.test_text = "Hello, how are you today?"
        self.sample_rate = 16000
        self.test_duration_ms = 2000  # 2 seconds of test audio

    async def setup_anonymous_config(self):
        """Setup configuration for anonymous mode testing"""
        # Override environment for anonymous mode
        os.environ["MEMORY_ENABLED"] = "false"  # Disable memory for pure testing
        os.environ["ANONYMOUS_MODE"] = "true"

        # Set lightweight models for testing
        os.environ["LLM_MODEL"] = "llama-3.2-1b-instruct"  # Start with lightweight model
        os.environ["VISION_MODEL_ENABLED"] = "false"  # Disable vision
        os.environ["DEBUG_MODE"] = "false"  # Remove debug overhead

        # Disable SOTA classifier
        os.environ["HOTMEM_USE_SOTA_CLASSIFIER"] = "false"

        logger.info("🔧 Anonymous mode configuration applied")

    async def initialize_services(self):
        """Initialize minimal voice agent services"""
        logger.info("🚀 Initializing services for anonymous mode testing...")

        with self.tracer.trace_sync("setup", "config_load"):
            self.config = VoiceAgentConfig.from_env()

        with self.tracer.trace_sync("setup", "factory_init"):
            self.factory = ServiceFactory(self.config)

        # Initialize core services
        service_names = ['stt', 'llm', 'tts', 'context']

        for service_name in service_names:
            with self.tracer.trace_sync("setup", f"{service_name}_init"):
                try:
                    self.services[service_name] = await self.factory.create_service(service_name)
                    logger.info(f"✅ {service_name} service initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {service_name}: {e}")
                    raise

        logger.info("🎯 All services initialized successfully")

    def generate_test_audio(self, duration_ms: int = 2000) -> AudioRawFrame:
        """Generate test audio frame (silence with a small tone)"""
        samples = int(duration_ms * self.sample_rate / 1000)

        # Generate simple test tone (440 Hz A note)
        t = np.linspace(0, duration_ms/1000, samples, False)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 0.1 amplitude

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return AudioRawFrame(
            audio=audio_bytes,
            sample_rate=self.sample_rate,
            num_channels=1
        )

    async def test_stt_latency(self) -> Dict[str, Any]:
        """Test STT (Speech-to-Text) latency"""
        logger.info("🎤 Testing STT latency...")

        test_audio = self.generate_test_audio(1000)  # 1 second

        start_time = time.time()

        with self.tracer.trace_async("stt", "transcribe"):
            # Process audio through STT service
            async for frame in self.services['stt'].process_frame(test_audio):
                if isinstance(frame, TranscriptionFrame):
                    stt_latency_ms = (time.time() - start_time) * 1000
                    logger.info(f"📝 STT Result: '{frame.text}' ({stt_latency_ms:.1f}ms)")
                    return {
                        "text": frame.text,
                        "latency_ms": stt_latency_ms,
                        "success": True
                    }

        return {"success": False, "error": "No transcription received"}

    async def test_llm_latency(self, input_text: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Test LLM inference latency"""
        logger.info(f"🤖 Testing LLM latency for: '{input_text}'")

        # Temporarily override model if specified
        original_model = None
        if model:
            original_model = os.environ.get("LLM_MODEL")
            os.environ["LLM_MODEL"] = model
            logger.info(f"🔄 Switching to model: {model}")

        try:
            start_time = time.time()

            with self.tracer.trace_async("llm", "inference", metadata={"model": model or os.environ.get("LLM_MODEL")}):
                # Create test context
                from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

                context = OpenAILLMContext()
                context.add_message({"role": "user", "content": input_text})

                # Process through LLM service
                response_chunks = []
                async for chunk in self.services['llm'].run_llm(context):
                    if hasattr(chunk, 'content'):
                        response_chunks.append(chunk.content)

                llm_latency_ms = (time.time() - start_time) * 1000
                response_text = "".join(response_chunks)

                logger.info(f"💬 LLM Response: '{response_text[:100]}...' ({llm_latency_ms:.1f}ms)")

                return {
                    "response": response_text,
                    "latency_ms": llm_latency_ms,
                    "model": model or os.environ.get("LLM_MODEL"),
                    "success": True
                }

        finally:
            # Restore original model
            if original_model:
                os.environ["LLM_MODEL"] = original_model

        return {"success": False, "error": "LLM processing failed"}

    async def test_tts_latency(self, text: str) -> Dict[str, Any]:
        """Test TTS (Text-to-Speech) latency"""
        logger.info(f"🔊 Testing TTS latency for: '{text}'")

        start_time = time.time()
        first_audio_time = None
        audio_chunks = []

        with self.tracer.trace_async("tts", "synthesize"):
            async for frame in self.services['tts'].run_tts(text):
                current_time = time.time()

                if isinstance(frame, TTSStartedFrame):
                    if first_audio_time is None:
                        first_audio_time = current_time

                elif isinstance(frame, TTSAudioRawFrame):
                    audio_chunks.append(len(frame.audio))

                elif isinstance(frame, TTSStoppedFrame):
                    break

        if first_audio_time:
            ttfb_ms = (first_audio_time - start_time) * 1000
            total_latency_ms = (time.time() - start_time) * 1000

            logger.info(f"🎵 TTS TTFB: {ttfb_ms:.1f}ms, Total: {total_latency_ms:.1f}ms, Chunks: {len(audio_chunks)}")

            return {
                "ttfb_ms": ttfb_ms,
                "total_ms": total_latency_ms,
                "chunks": len(audio_chunks),
                "success": True
            }

        return {"success": False, "error": "No audio generated"}

    async def test_end_to_end_latency(self) -> Dict[str, Any]:
        """Test complete end-to-end pipeline latency"""
        logger.info("🔄 Testing end-to-end pipeline latency...")

        pipeline_start = time.time()
        results = {}

        # Step 1: STT
        with self.tracer.trace_async("pipeline", "stt_stage"):
            stt_result = await self.test_stt_latency()
            results["stt"] = stt_result

        if not stt_result.get("success"):
            return {"success": False, "stage": "stt", "results": results}

        # Step 2: LLM
        with self.tracer.trace_async("pipeline", "llm_stage"):
            llm_result = await self.test_llm_latency(stt_result["text"])
            results["llm"] = llm_result

        if not llm_result.get("success"):
            return {"success": False, "stage": "llm", "results": results}

        # Step 3: TTS
        with self.tracer.trace_async("pipeline", "tts_stage"):
            tts_result = await self.test_tts_latency(llm_result["response"])
            results["tts"] = tts_result

        if not tts_result.get("success"):
            return {"success": False, "stage": "tts", "results": results}

        total_latency_ms = (time.time() - pipeline_start) * 1000

        logger.info(f"🎯 END-TO-END COMPLETE: {total_latency_ms:.1f}ms")

        return {
            "success": True,
            "total_latency_ms": total_latency_ms,
            "results": results
        }

    async def run_model_comparison(self) -> Dict[str, Any]:
        """Compare performance across different LLM models"""
        logger.info("🔍 Running model comparison test...")

        models_to_test = [
            "llama-3.2-1b-instruct",     # Fastest (expected)
            "gemma-3n-e4b",              # Small MLX model
            "qwen3-vl-4b-instruct-mlx", # Current model (problematic)
        ]

        results = {}

        for model in models_to_test:
            logger.info(f"🧪 Testing model: {model}")

            try:
                result = await self.test_llm_latency(self.test_text, model)
                results[model] = result

                # Check if this is the problematic model
                if result.get("latency_ms", 0) > 10000:  # >10 seconds
                    logger.error(f"🚨 CRITICAL: Model {model} shows {result['latency_ms']:.1f}ms latency!")

            except Exception as e:
                logger.error(f"❌ Model {model} failed: {e}")
                results[model] = {"success": False, "error": str(e)}

        return results

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run complete diagnostic suite"""
        logger.info("🩺 Starting comprehensive latency diagnostics...")

        diagnostic_results = {
            "timestamp": time.time(),
            "configuration": {
                "anonymous_mode": True,
                "memory_disabled": True,
                "model": os.environ.get("LLM_MODEL"),
                "vision_disabled": os.environ.get("VISION_MODEL_ENABLED") == "false"
            },
            "results": {}
        }

        try:
            # Setup
            await self.setup_anonymous_config()
            await self.initialize_services()

            # Test 1: End-to-end pipeline
            logger.info("\n" + "="*60)
            logger.info("TEST 1: End-to-End Pipeline Latency")
            logger.info("="*60)

            e2e_result = await self.test_end_to_end_latency()
            diagnostic_results["results"]["end_to_end"] = e2e_result

            # Test 2: Model comparison
            logger.info("\n" + "="*60)
            logger.info("TEST 2: Model Performance Comparison")
            logger.info("="*60)

            model_results = await self.run_model_comparison()
            diagnostic_results["results"]["model_comparison"] = model_results

            # Generate tracer report
            logger.info("\n" + "="*60)
            logger.info("LATENCY TRACER REPORT")
            logger.info("="*60)

            self.tracer.print_report()

            # Export detailed results
            export_path = "/tmp/anonymous_latency_diagnostics.json"
            diagnostic_results["tracer_data"] = {
                "measurements": len(self.tracer.measurements),
                "pipeline_summary": self.tracer.get_pipeline_summary()
            }

            with open(export_path, 'w') as f:
                import json
                json.dump(diagnostic_results, f, indent=2, default=str)

            logger.info(f"📁 Detailed results exported to: {export_path}")

        except Exception as e:
            logger.error(f"❌ Diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
            diagnostic_results["error"] = str(e)

        return diagnostic_results

    def print_summary(self, results: Dict[str, Any]):
        """Print diagnostic summary"""
        print("\n" + "="*80)
        print("🏥 ANONYMOUS MODE LATENCY DIAGNOSTIC SUMMARY")
        print("="*80)

        if "error" in results:
            print(f"❌ DIAGNOSTIC FAILED: {results['error']}")
            return

        print(f"🕒 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}")

        # Configuration
        config = results["configuration"]
        print(f"\n⚙️ CONFIGURATION:")
        print(f"   Anonymous Mode: {config['anonymous_mode']}")
        print(f"   Memory Disabled: {config['memory_disabled']}")
        print(f"   Vision Disabled: {config['vision_disabled']}")
        print(f"   LLM Model: {config['model']}")

        # End-to-end results
        if "end_to_end" in results["results"]:
            e2e = results["results"]["end_to_end"]
            if e2e.get("success"):
                print(f"\n🎯 END-TO-END PIPELINE:")
                print(f"   Total Latency: {e2e['total_latency_ms']:.1f}ms")
                print(f"   STT: {e2e['results']['stt']['latency_ms']:.1f}ms")
                print(f"   LLM: {e2e['results']['llm']['latency_ms']:.1f}ms")
                print(f"   TTS: {e2e['results']['tts']['total_ms']:.1f}ms")

                # Status
                total_ms = e2e['total_latency_ms']
                if total_ms < 1000:
                    print(f"   Status: ✅ EXCELLENT (<1s)")
                elif total_ms < 3000:
                    print(f"   Status: ⚠️ ACCEPTABLE ({total_ms/1000:.1f}s)")
                else:
                    print(f"   Status: 🔥 CRITICAL ({total_ms/1000:.1f}s - TOO SLOW)")
            else:
                print(f"\n❌ END-TO-END FAILED at stage: {e2e.get('stage')}")

        # Model comparison
        if "model_comparison" in results["results"]:
            print(f"\n🤖 MODEL COMPARISON:")
            for model, result in results["results"]["model_comparison"].items():
                if result.get("success"):
                    latency = result["latency_ms"]
                    status = "🔥 CRITICAL" if latency > 10000 else "⚠️ SLOW" if latency > 1000 else "✅ GOOD"
                    print(f"   {model:30} {latency:8.1f}ms {status}")
                else:
                    print(f"   {model:30} {'FAILED':>8} ❌")

        print("="*80)

async def main():
    """Run anonymous mode latency diagnostics"""
    tester = AnonymousLatencyTester()

    print("🩺 Anonymous Mode Latency Diagnostic Tool")
    print("=" * 50)
    print("This tool tests voice pipeline performance in anonymous mode")
    print("to identify the source of the 40+ second LLM latency issue.")
    print()

    # Run diagnostics
    results = await tester.run_diagnostics()

    # Print summary
    tester.print_summary(results)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")

    if "model_comparison" in results.get("results", {}):
        model_results = results["results"]["model_comparison"]

        # Find fastest model
        fastest_model = None
        fastest_latency = float('inf')

        for model, result in model_results.items():
            if result.get("success") and result.get("latency_ms", float('inf')) < fastest_latency:
                fastest_model = model
                fastest_latency = result["latency_ms"]

        if fastest_model and fastest_latency < 1000:
            print(f"   1. Switch to fastest model: {fastest_model} ({fastest_latency:.1f}ms)")
            print(f"      Fix: export LLM_MODEL={fastest_model}")

        # Check for problematic models
        for model, result in model_results.items():
            if result.get("success") and result.get("latency_ms", 0) > 10000:
                print(f"   2. CRITICAL: Model {model} has {result['latency_ms']/1000:.1f}s latency!")
                print(f"      This model should NOT be used for real-time applications")

    print("   3. Ensure debug logging is disabled in production")
    print("   4. Consider model pre-loading and caching")
    print("   5. Test with different LLM server configurations")

if __name__ == "__main__":
    asyncio.run(main())