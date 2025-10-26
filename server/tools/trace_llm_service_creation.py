#!/usr/bin/env python3
"""
TRACE LLM SERVICE CREATION
Trace how LLM services are created and identify why models are being reloaded
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add server path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory
from core.factories.service_factory import ServiceFactory

class LLMServiceTracer:
    """Trace LLM service creation and usage patterns"""

    def __init__(self):
        self.trace_log = []

    def log_service_creation(self, service_type: str, service_id: str, details: str = ""):
        """Log service creation events"""
        event = {
            "timestamp": time.time(),
            "type": "service_creation",
            "service_type": service_type,
            "service_id": service_id,
            "details": details
        }
        self.trace_log.append(event)
        logger.info(f"🏭 {service_type} created: {service_id} {details}")

    def log_service_usage(self, service_id: str, operation: str, details: str = ""):
        """Log service usage events"""
        event = {
            "timestamp": time.time(),
            "type": "service_usage",
            "service_id": service_id,
            "operation": operation,
            "details": details
        }
        self.trace_log.append(event)
        logger.info(f"🔧 {service_id} {operation}: {details}")

    def trace_service_factory_caching(self):
        """Trace how ServiceFactory caches LLM services"""
        logger.info("🔍 Tracing ServiceFactory caching behavior")

        # Create multiple ServiceFactory instances
        factories = []
        llm_services = []

        for i in range(3):
            logger.info(f"\n--- Factory {i+1} ---")

            config = VoiceAgentConfig.from_env()
            factory = ServiceFactory(config)
            factories.append(factory)

            # Check if factory has existing cache
            existing_cache = hasattr(factory, '_services_cache') and factory._services_cache
            logger.info(f"   Factory {i+1} existing cache: {existing_cache}")

            if existing_cache:
                cached_services = list(factory._services_cache.keys())
                logger.info(f"   Cached services: {cached_services}")

            # Create LLM service
            start_time = time.time()
            llm_service = factory.create_llm_service()
            creation_time = (time.time() - start_time) * 1000

            llm_services.append(llm_service)

            # Get service details
            service_id = id(llm_service)
            service_model = getattr(llm_service, 'model', 'unknown')

            self.log_service_creation(
                "LLM_Service",
                f"factory_{i+1}_llm_{service_id}",
                f"model={service_model}, creation_time={creation_time:.1f}ms"
            )

            # Check if service was cached or newly created
            is_cached = 'llm' in factory._services_cache and factory._services_cache['llm'] is llm_service
            logger.info(f"   Service cached: {is_cached}")

        # Check if services are the same instance
        unique_services = len(set(id(s) for s in llm_services))
        logger.info(f"\n📊 Results:")
        logger.info(f"   Total factories created: {len(factories)}")
        logger.info(f"   Total LLM services created: {len(llm_services)}")
        logger.info(f"   Unique service instances: {unique_services}")

        if unique_services == 1:
            logger.info("   ✅ All factories share the same LLM service (good)")
        else:
            logger.error(f"   🔥 CRITICAL: {unique_services} different LLM service instances!")
            logger.error("   This means the model is being loaded multiple times!")

        return unique_services

    def trace_voice_agent_factory_caching(self):
        """Trace how VoiceAgentFactory caches LLM services"""
        logger.info("🔍 Tracing VoiceAgentFactory caching behavior")

        # Create multiple VoiceAgentFactory instances
        factories = []
        llm_services = []

        for i in range(3):
            logger.info(f"\n--- VoiceAgent Factory {i+1} ---")

            config = VoiceAgentConfig.from_env()
            factory = VoiceAgentFactory(config)
            factories.append(factory)

            # Check if VoiceAgentFactory has existing cache
            existing_cache = hasattr(factory, '_services_cache') and factory._services_cache
            logger.info(f"   VoiceAgent Factory {i+1} existing cache: {existing_cache}")

            if existing_cache:
                cached_services = list(factory._services_cache.keys())
                logger.info(f"   Cached services: {cached_services}")

            # Check if internal ServiceFactory has cache
            internal_cache = hasattr(factory._service_factory, '_services_cache') and factory._service_factory._services_cache
            logger.info(f"   Internal ServiceFactory cache: {internal_cache}")

            # Create LLM service
            start_time = time.time()
            llm_service = factory.create_llm_service()
            creation_time = (time.time() - start_time) * 1000

            llm_services.append(llm_service)

            # Get service details
            service_id = id(llm_service)
            service_model = getattr(llm_service, 'model', 'unknown')

            self.log_service_creation(
                "VoiceAgent_LLM",
                f"voice_factory_{i+1}_llm_{service_id}",
                f"model={service_model}, creation_time={creation_time:.1f}ms"
            )

            # Check if service was cached or newly created
            voice_cached = 'llm' in factory._services_cache and factory._services_cache['llm'] is llm_service
            internal_cached = 'llm' in factory._service_factory._services_cache and factory._service_factory._services_cache['llm'] is llm_service

            logger.info(f"   VoiceAgent cached: {voice_cached}")
            logger.info(f"   Internal ServiceFactory cached: {internal_cached}")

        # Check if services are the same instance
        unique_services = len(set(id(s) for s in llm_services))
        logger.info(f"\n📊 VoiceAgent Results:")
        logger.info(f"   Total VoiceAgent factories created: {len(factories)}")
        logger.info(f"   Total LLM services created: {len(llm_services)}")
        logger.info(f"   Unique service instances: {unique_services}")

        if unique_services == 1:
            logger.info("   ✅ All VoiceAgent factories share the same LLM service (good)")
        else:
            logger.error(f"   🔥 CRITICAL: {unique_services} different LLM service instances!")
            logger.error("   This means the model is being loaded multiple times!")

        return unique_services

    def test_lm_studio_connection_reuse(self):
        """Test if LM Studio connections are being reused"""
        logger.info("🔍 Testing LM Studio connection reuse")

        from pipecat.services.openai.llm import OpenAILLMService

        # Create multiple LLM services with the same configuration
        services = []
        connections = []

        config = VoiceAgentConfig.from_env()
        llm_config = config.get_component_config("llm")

        for i in range(3):
            logger.info(f"\n--- LLM Service {i+1} ---")

            start_time = time.time()
            llm = OpenAILLMService(
                api_key=llm_config["api_key"],
                model=llm_config["model"],
                base_url=llm_config["base_url"],
                max_tokens=llm_config["max_tokens"],
                stream=True,
                debug=False,
                extra_body={
                    "think": False,
                    "stream": True,
                    "options": {
                        "num_predict": 768,
                        "temperature": llm_config["temperature"],
                        "top_k": 40,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "num_ctx": 4096,
                        "num_batch": 64,
                        "use_mlock": True,
                        "f16_kv": True,
                        "keep_alive": "15m"  # This should keep the model loaded
                    }
                },
            )

            creation_time = (time.time() - start_time) * 1000
            services.append(llm)

            # Try to access the underlying HTTP client/session
            if hasattr(llm, '_client'):
                client_id = id(llm._client)
                connections.append(client_id)
                logger.info(f"   Service {i+1} client ID: {client_id}")
            else:
                logger.info(f"   Service {i+1} client not accessible")

            logger.info(f"   Creation time: {creation_time:.1f}ms")

        # Check if connections are shared
        unique_connections = len(set(connections))
        logger.info(f"\n📊 Connection Results:")
        logger.info(f"   Total services created: {len(services)}")
        logger.info(f"   Unique connections: {unique_connections}")

        if unique_connections == 1:
            logger.info("   ✅ All services share the same connection (good)")
        else:
            logger.warning(f"   ⚠️ {unique_connections} different connections")

        return unique_connections

    def trace_bot_startup_process(self):
        """Trace the actual bot startup process"""
        logger.info("🔍 Tracing actual bot startup process")

        try:
            # Simulate bot.py startup
            config = VoiceAgentConfig.from_env()
            logger.info("   ✅ Configuration loaded")

            factory = VoiceAgentFactory(config)
            logger.info("   ✅ VoiceAgentFactory created")

            # Check what happens when services are created
            start_time = time.time()

            # This is what happens in bot.py line 145
            logger.info("   Creating voice agent services...")

            # Create a subset of services to trace
            llm = factory.create_llm_service()
            logger.info(f"   ✅ LLM service created in {(time.time() - start_time) * 1000:.1f}ms")

            # Check if the service has any model loading state
            if hasattr(llm, 'model'):
                logger.info(f"   Model: {llm.model}")

            if hasattr(llm, 'base_url'):
                logger.info(f"   Base URL: {llm.base_url}")

        except Exception as e:
            logger.error(f"   ❌ Startup trace failed: {e}")

    def print_trace_summary(self):
        """Print comprehensive trace summary"""
        print("\n" + "="*80)
        print("🔍 LLM SERVICE CREATION TRACE SUMMARY")
        print("="*80)

        # Group events by type
        service_creations = [e for e in self.trace_log if e["type"] == "service_creation"]
        service_usages = [e for e in self.trace_log if e["type"] == "service_usage"]

        print(f"\n📊 TRACE STATISTICS:")
        print(f"   Total service creation events: {len(service_creations)}")
        print(f"   Total service usage events: {len(service_usages)}")

        if service_creations:
            print(f"\n🏭 SERVICE CREATIONS:")
            for event in service_creations:
                print(f"   {event['timestamp']:.1f} - {event['service_type']} ({event['service_id'][:12]}...) {event['details']}")

        print(f"\n💡 ANALYSIS:")

        # Look for patterns that would cause 40-second delays
        llm_creations = [e for e in service_creations if 'LLM' in e['service_type']]

        if len(llm_creations) > 1:
            print(f"   🔥 CRITICAL: {len(llm_creations)} LLM services created!")
            print(f"      This suggests the model is being loaded multiple times")
            print(f"      Each load could take 6+ seconds, compounding to 40+ seconds")

        if len(llm_creations) == 1:
            print(f"   ✅ Only one LLM service created (good)")

        print(f"\n🎯 ROOT CAUSE HYPOTHESIS:")
        if len(llm_creations) > 1:
            print(f"   The 40-second delay is likely caused by:")
            print(f"   1. Multiple LLM service instances being created")
            print(f"   2. Each instance loading the model from scratch")
            print(f"   3. No proper connection/session reuse")
        else:
            print(f"   The single LLM service suggests the issue is elsewhere")
            print(f"   Possibly in the pipeline processing or model initialization")

        print("="*80)

async def main():
    """Run LLM service creation tracing"""
    print("🔍 LLM Service Creation Tracer")
    print("=" * 40)
    print("This tool traces how LLM services are created")
    print("to identify why models are being reloaded.")
    print()

    tracer = LLMServiceTracer()

    # Trace 1: ServiceFactory caching
    print("Trace 1: ServiceFactory Caching")
    tracer.trace_service_factory_caching()

    print("\n" + "="*60)

    # Trace 2: VoiceAgentFactory caching
    print("Trace 2: VoiceAgentFactory Caching")
    tracer.trace_voice_agent_factory_caching()

    print("\n" + "="*60)

    # Trace 3: Connection reuse
    print("Trace 3: Connection Reuse")
    tracer.test_lm_studio_connection_reuse()

    print("\n" + "="*60)

    # Trace 4: Bot startup simulation
    print("Trace 4: Bot Startup Simulation")
    tracer.trace_bot_startup_process()

    # Print summary
    tracer.print_trace_summary()

    # Export trace log
    import json
    export_path = "/tmp/llm_service_creation_trace.json"
    with open(export_path, 'w') as f:
        json.dump(tracer.trace_log, f, indent=2, default=str)
    print(f"\n📁 Trace log exported to: {export_path}")

if __name__ == "__main__":
    asyncio.run(main())