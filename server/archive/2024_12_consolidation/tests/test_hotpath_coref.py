#!/usr/bin/env python3
"""
Quick test to verify the integrated SimpleCoreferenceResolver in HotpathTier1Extractor
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hotpath_tier1_extractor import HotPathTier1Extractor

def test_hotpath_coreference():
    """Test the integrated coreference resolution in hotpath extractor"""
    print("🧪 TESTING HOTPATH WITH INTEGRATED COREFERENCE")
    print("=" * 60)

    extractor = HotPathTier1Extractor()

    # Test cases with pronouns that should be resolved
    test_cases = [
        {
            "text": "Steve Jobs founded Apple. He was a visionary CEO.",
            "expected_pronouns": ["he"]
        },
        {
            "text": "Maria works at Google. She leads the AI team there.",
            "expected_pronouns": ["she"]
        },
        {
            "text": "The company Apple is successful. It changed the world.",
            "expected_pronouns": ["it"]
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test['text']}")
        print("-" * 40)

        # Extract with hotpath
        result = extractor.extract(test['text'])

        print(f"📊 Entities: {result.entities}")
        print(f"📈 Relations: {result.relations}")
        print(f"⏱️  Time: {result.extraction_time_ms:.1f}ms")
        print(f"🎯 Confidence: {result.confidence:.2f}")

        # Check if pronouns were resolved (not present in final relations)
        pronouns_in_relations = []
        for subj, rel, obj in result.relations:
            for pronoun in test['expected_pronouns']:
                if pronoun.lower() in [subj.lower(), obj.lower()]:
                    pronouns_in_relations.append(pronoun)

        if pronouns_in_relations:
            print(f"❌ Unresolved pronouns: {pronouns_in_relations}")
        else:
            print("✅ All pronouns resolved!")

    print(f"\n📈 Performance Stats:")
    stats = extractor.get_performance_stats()
    print(f"   Average time: {stats['average_time_ms']:.1f}ms")
    print(f"   Under 500ms: {'✅' if stats['under_500ms_guarantee'] else '❌'}")
    print(f"   Components loaded:")
    for comp, available in stats['components'].items():
        print(f"     {comp}: {'✅' if available else '❌'}")

if __name__ == "__main__":
    test_hotpath_coreference()