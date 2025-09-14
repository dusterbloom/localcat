#!/usr/bin/env python3
"""
Test GLiREL with complex sentences that have many entities and relations
"""

import os
import sys
sys.path.append('.')

from components.memory.config import create_config
from components.extraction.memory_extractor import MemoryExtractor

def test_complex_glirel():
    """Test GLiREL with complex, entity-rich sentences"""

    print("🧪 Testing GLiREL with Complex Sentences")
    print("=" * 60)

    # Set GLiREL enabled
    os.environ["HOTMEM_USE_GLIREL"] = "true"

    # Create config
    config = create_config()
    extractor_config = config.get_extractor_config()
    extractor = MemoryExtractor(extractor_config)

    # Complex test sentences with many entities and relationships
    test_sentences = [
        "Elon Musk, the CEO of Tesla and SpaceX, was born in South Africa and later moved to Silicon Valley where he founded PayPal with Peter Thiel and Max Levchin before selling it to eBay for 1.5 billion dollars.",

        "Dr. Sarah Chen, the former research director at Google DeepMind, left her position at the London office to join OpenAI in San Francisco, where she now leads the GPT-4 development team alongside Sam Altman and Greg Brockman.",

        "Microsoft Corporation acquired GitHub for 7.5 billion dollars in 2018, while Amazon Web Services competed with Google Cloud Platform and Microsoft Azure for the enterprise cloud market dominated by Jeff Bezos and Satya Nadella.",

        "The iPhone 15 Pro, manufactured by Apple Inc. in Cupertino, California, costs 999 dollars and competes with Samsung Galaxy S24 Ultra produced in South Korea by Samsung Electronics under the leadership of Jay Y. Lee."
    ]

    for i, text in enumerate(test_sentences, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"📝 Text: '{text[:100]}...'")
        print(f"📏 Length: {len(text)} chars, {len(text.split())} words")

        try:
            result = extractor.extract(text)

            print(f"✅ Extraction completed!")
            print(f"📊 Results:")
            print(f"  • Entities: {len(result.entities)}")
            print(f"  • Triples: {len(result.triples)}")

            # Show entities found
            if result.entities:
                print(f"  • Entity sample: {result.entities[:5]}")

            # Show triples
            if result.triples:
                print(f"  • Triple sample: {result.triples[:5]}")

            # Check GLiREL contribution specifically
            if hasattr(extractor, 'metrics'):
                glirel_times = extractor.metrics.get('glirel_ms', [])
                if glirel_times:
                    print(f"  • GLiREL time: {glirel_times[-1]:.1f}ms")

            print("-" * 60)

        except Exception as e:
            print(f"❌ Test {i} failed: {e}")

    return True

if __name__ == "__main__":
    test_complex_glirel()