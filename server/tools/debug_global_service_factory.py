#!/usr/bin/env python3
"""
DEBUG GLOBAL SERVICE FACTORY
Debug the global ServiceFactory to ensure it's working properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import VoiceAgentConfig
from core.factory import get_global_service_factory

def debug_global_service_factory():
    """Debug the global ServiceFactory behavior"""
    print("🔍 Debugging Global ServiceFactory")
    print("=" * 50)

    # Create first VoiceAgentFactory config
    config1 = VoiceAgentConfig.from_env()
    print(f"Config 1 model: {config1.get_component_config('llm').get('model', 'unknown')}")

    # Get global ServiceFactory first time
    factory1 = get_global_service_factory(config1)
    print(f"First global factory: {id(factory1)}")

    # Check if ServiceFactory has cached LLM service
    if hasattr(factory1, '_services_cache') and 'llm' in factory1._services_cache:
        print(f"First factory has cached LLM: {id(factory1._services_cache['llm'])}")
    else:
        print("First factory no cached LLM")

    # Create second VoiceAgentFactory config
    config2 = VoiceAgentConfig.from_env()
    print(f"Config 2 model: {config2.get_component_config('llm').get('model', 'unknown')}")

    # Get global ServiceFactory second time
    factory2 = get_global_service_factory(config2)
    print(f"Second global factory: {id(factory2)}")

    # Check if same instance
    print(f"Same instance? {factory1 is factory2}")

    # Check if second factory has cached LLM service
    if hasattr(factory2, '_services_cache') and 'llm' in factory2._services_cache:
        print(f"Second factory has cached LLM: {id(factory2._services_cache['llm'])}")
        print(f"Same LLM instance? {factory1._services_cache.get('llm') is factory2._services_cache.get('llm')}")
    else:
        print("Second factory no cached LLM")

if __name__ == "__main__":
    debug_global_service_factory()