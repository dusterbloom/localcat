#!/usr/bin/env python3
"""Test Enhanced Level3 extractor directly"""

import os
import sys

# Add server path
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')

# Lower thresholds for more extraction
os.environ['ENHANCED_LEVEL3_ENTITY_CONF'] = '0.35'
os.environ['ENHANCED_LEVEL3_RELATION_CONF'] = '0.25'

def test_enhanced_level3_direct():
    """Test Enhanced Level3 extractor directly"""
    print("🧪 Testing Enhanced Level3 extractor directly")

    try:
        from components.extraction.extraction_strategies import EnhancedLevel3ExtractionStrategy

        # Initialize extractor
        extractor = EnhancedLevel3ExtractionStrategy()

        if not extractor.is_available():
            print("❌ Enhanced Level3 extractor not available")
            return

        print("✅ Enhanced Level3 extractor available")

        # Test cases
        test_cases = [
            "My dog's name is Max",
            "I work at TechCorp",
            "I live in San Francisco",
            "I went to Stanford University for computer science"
        ]

        for text in test_cases:
            print(f"\n📝 Testing: '{text}'")
            triples = extractor.extract(text)
            print(f"   Extracted {len(triples)} triples:")
            for s, r, o in triples:
                print(f"     • ({s}, {r}, {o})")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_level3_direct()