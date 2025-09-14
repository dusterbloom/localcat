#!/usr/bin/env python3
"""
Test semantic extraction with SRL for the critical failing example
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hotpath_tier1_extractor import HotPathTier1Extractor

def test_semantic_extraction():
    """Test semantic extraction on the critical failing example"""
    print("🧪 TESTING SEMANTIC EXTRACTION WITH SRL")
    print("=" * 60)

    extractor = HotPathTier1Extractor()

    # The critical failing case
    test_text = "The CEO announced that the company would restructure after declining profits."
    print(f"📝 Test Text: {test_text}")
    print("\nExpected semantic relations:")
    print("- (CEO, announced, restructuring_decision)")
    print("- (company, will_restructure, NULL)")
    print("- (restructuring, because_of, declining_profits)")
    print("- (profits, status, declining)")

    print(f"\n🔬 EXTRACTION WITH SRL:")
    print("-" * 40)

    result = extractor.extract(test_text)

    print(f"📊 Entities ({len(result.entities)}): {result.entities}")
    print(f"📈 Relations ({len(result.relations)}):")
    for i, (subj, rel, obj) in enumerate(result.relations, 1):
        print(f"   {i}. ({subj}, {rel}, {obj})")

    print(f"\n⏱️  Performance: {result.extraction_time_ms:.1f}ms")
    print(f"🎯 Confidence: {result.confidence:.2f}")

    # Check for expected semantic relations
    print(f"\n🧠 SEMANTIC RELATION ANALYSIS:")
    print("-" * 35)

    causal_found = any("because" in rel or "cause" in rel for _, rel, _ in result.relations)
    agent_found = any("ceo" in subj.lower() and "announce" in rel.lower() for subj, rel, _ in result.relations)
    restructure_found = any("restructure" in str(rel).lower() or "restructure" in str(obj).lower() for _, rel, obj in result.relations)

    print(f"✅ Causal relations (because_of): {'✓' if causal_found else '❌'}")
    print(f"✅ Agent relations (CEO announced): {'✓' if agent_found else '❌'}")
    print(f"✅ Action extraction (restructure): {'✓' if restructure_found else '❌'}")

    # Component status
    stats = extractor.get_performance_stats()
    print(f"\n🔧 COMPONENT STATUS:")
    print("-" * 25)
    for comp, available in stats['components'].items():
        status = '✅' if available else '❌'
        print(f"{status} {comp.replace('_', ' ').title()}")

    print(f"\n🎉 SEMANTIC INTEGRATION TEST COMPLETE!")
    return result

if __name__ == "__main__":
    test_semantic_extraction()