#!/usr/bin/env python3
"""
Simple test for GLiREL integration with MemoryExtractor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.extraction.memory_extractor import MemoryExtractor

def test_glirel_integration():
    """Test GLiREL integration with MemoryExtractor"""

    print("🧪 Testing GLiREL Integration with MemoryExtractor")
    print("=" * 60)

    try:
        # Initialize MemoryExtractor with GLiREL enabled
        config = {
            'use_glirel': True,
            'use_gliner': True,
            'sqlite_path': ':memory:',
            'session_id': 'test_glirel_session'
        }

        print("📦 Initializing MemoryExtractor with GLiREL...")
        extractor = MemoryExtractor(config)

        # Test text
        text = "Steve Jobs founded Apple Inc. in Cupertino. He worked there as CEO."
        print(f"📝 Test text: '{text}'")

        print("\n🔬 Extracting memory...")
        result = extractor.extract(text)

        print(f"✅ Extraction completed!")
        print(f"📊 Results:")
        print(f"  - Entities: {len(result.entities)}")
        print(f"  - Triples: {len(result.triples)}")

        # Show extracted entities
        if result.entities:
            print("\n👥 Entities:")
            for i, entity in enumerate(result.entities[:5], 1):
                print(f"  {i}. {entity}")

        # Show triples
        if result.triples:
            print("\n📋 Triples:")
            for i, triple in enumerate(result.triples[:5], 1):
                print(f"  {i}. {triple}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_glirel_integration()
    if success:
        print("\n🎉 GLiREL integration test PASSED!")
    else:
        print("\n💥 GLiREL integration test FAILED!")
        sys.exit(1)