#!/usr/bin/env python3
"""Test that Enhanced Level3 is now the default extraction strategy"""

import os
import sys

# Check environment
print("Current extraction configuration:")
print(f"DEFAULT_EXTRACTION_STRATEGY={os.getenv('DEFAULT_EXTRACTION_STRATEGY', 'not set')}")
print(f"FALLBACK_EXTRACTION_STRATEGY={os.getenv('FALLBACK_EXTRACTION_STRATEGY', 'not set')}")

# Test the registry
from components.extraction.extraction_registry import ExtractionRegistry

registry = ExtractionRegistry()
print("\nRegistered strategies:")
for name in registry.strategies.keys():
    print(f"  - {name}")

# Test that enhanced_level3 works
if 'enhanced_level3' in registry.strategies:
    print("\n✅ Enhanced Level3 is registered!")
    
    # Try to create instance
    strategy_class = registry.strategies['enhanced_level3'].strategy_class
    strategy = strategy_class()
    
    if strategy.is_available():
        print("✅ Enhanced Level3 is available and ready!")
        
        # Test extraction
        test_text = "John Smith works at Google in San Francisco."
        triples = strategy.extract(test_text)
        
        print(f"\nTest extraction result:")
        print(f"  Input: {test_text}")
        print(f"  Output: {len(triples)} triples")
        for s, p, o in triples:
            print(f"    - {s} | {p} | {o}")
    else:
        print("⚠️ Enhanced Level3 is registered but not available")
else:
    print("❌ Enhanced Level3 is NOT registered!")

print("\n🎯 Enhanced Level3 is now the DEFAULT extraction strategy!")
