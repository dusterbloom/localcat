#!/usr/bin/env python3
"""
Test GLiREL configuration integration with the production bot.py pipeline
"""

import os
import sys
sys.path.append('.')

from components.memory.config import create_config
from components.extraction.memory_extractor import MemoryExtractor


def test_glirel_config():
    """Test that GLiREL configuration is properly integrated"""

    print("🔬 Testing GLiREL Configuration Integration")
    print("=" * 50)

    # Test with GLiREL enabled
    os.environ["HOTMEM_USE_GLIREL"] = "true"

    # Create config and test GLiREL integration
    config = create_config()

    print(f"📊 Feature flags:")
    print(f"  • GLiNER enabled: {config.features.use_gliner}")
    print(f"  • GLiREL enabled: {config.features.use_glirel}")
    print(f"  • ReLiK enabled: {config.features.use_relik}")

    # Test extractor configuration
    extractor_config = config.get_extractor_config()
    print(f"\n🛠️  Extractor config:")
    print(f"  • use_glirel: {extractor_config.get('use_glirel', 'MISSING')}")
    print(f"  • use_gliner: {extractor_config.get('use_gliner', 'MISSING')}")
    print(f"  • use_relik: {extractor_config.get('use_relik', 'MISSING')}")

    # Test MemoryExtractor initialization
    try:
        # Use the extractor config dict instead of the config object
        extractor_config = config.get_extractor_config()
        extractor = MemoryExtractor(extractor_config)
        print(f"\n✅ MemoryExtractor initialized successfully")
        print(f"  • Has GLiREL: {hasattr(extractor, '_glirel_extractor')}")

        # Test extraction with sample text
        text = "Steve Jobs founded Apple Inc. in Cupertino."

        print(f"\n🧪 Testing extraction with text: '{text}'")

        result = extractor.extract(text)
        print(f"✅ Extraction completed")
        print(f"  • Triples: {len(result.triples)}")
        print(f"  • Entities: {len(result.entities)}")

        # Show first few triples
        if result.triples:
            print(f"\n📝 First 3 triples:")
            for i, triple in enumerate(result.triples[:3]):
                print(f"  {i+1}. {triple}")

        return True

    except Exception as e:
        print(f"❌ MemoryExtractor failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_glirel_config()
    print(f"\n🎯 Test result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)