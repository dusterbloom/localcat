#!/usr/bin/env python3
"""
Debug GLiREL entity extraction issue
"""

import os
import sys
sys.path.append('.')

from components.memory.config import create_config
from components.extraction.memory_extractor import MemoryExtractor


def debug_glirel_entities():
    """Debug why GLiREL is getting zero entities"""

    print("🔬 Debugging GLiREL Entity Issue")
    print("=" * 50)

    # Set GLiREL enabled
    os.environ["HOTMEM_USE_GLIREL"] = "true"

    # Create config
    config = create_config()
    extractor_config = config.get_extractor_config()
    extractor = MemoryExtractor(extractor_config)

    # Test extraction with detailed logging
    text = "Steve Jobs founded Apple Inc. in Cupertino, California."

    print(f"🧪 Testing extraction with text: '{text}'")

    # Let's run extraction step by step to see where entities come from
    result = extractor.extract(text)

    print(f"\n📊 Extraction Results:")
    print(f"  • Triples: {len(result.triples)}")
    print(f"  • Entities: {len(result.entities)}")
    print(f"  • Strategies: {getattr(result, 'strategies_used', 'Unknown')}")

    print(f"\n📝 All Triples:")
    for i, triple in enumerate(result.triples):
        print(f"  {i+1}. {triple}")

    print(f"\n🏷️  All Entities:")
    for i, entity in enumerate(result.entities):
        print(f"  {i+1}. {entity}")

    # Check if GLiREL was used
    if hasattr(extractor, 'metrics'):
        print(f"\n📈 GLiREL Metrics:")
        print(f"  • GLiREL calls: {len(extractor.metrics.get('glirel_ms', []))}")
        print(f"  • GLiREL times: {extractor.metrics.get('glirel_ms', [])}")

    return True


if __name__ == "__main__":
    debug_glirel_entities()