#!/usr/bin/env python3
"""Direct comparison: Level3 vs Enhanced Level3"""

import time
import spacy
from level3_universal_kg import UniversalKGExtractor
from enhanced_level3_extractor import QualityExtractor

def compare_extractors():
    # Test texts
    texts = [
        "John Smith works at Google in San Francisco. He manages the AI team and develops new products.",
        "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees.",
        "Yesterday, firefighters quickly responded to a small kitchen fire caused by an unattended stove."
    ]
    
    # Initialize
    level3 = UniversalKGExtractor()
    enhanced = QualityExtractor()
    nlp = spacy.load('en_core_web_sm')
    
    print("=" * 70)
    print("LEVEL3 vs ENHANCED LEVEL3 - HEAD TO HEAD")
    print("=" * 70)
    
    for i, text in enumerate(texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        print("-" * 60)
        
        # Level3 Original
        start = time.perf_counter()
        kg_original = level3.extract_universal_kg(text)
        time_original = (time.perf_counter() - start) * 1000
        
        # Enhanced Level3
        doc = nlp(text)
        start = time.perf_counter()
        kg_enhanced = enhanced.extract_quality_kg(doc)
        time_enhanced = (time.perf_counter() - start) * 1000
        
        # Results
        print(f"\n📊 LEVEL3 ORIGINAL:")
        print(f"   Time: {time_original:.2f}ms")
        print(f"   Relations: {len(kg_original.relations)}")
        print(f"   Sample output:")
        for r in kg_original.relations[:5]:
            print(f"     - {r.subject} | {r.predicate} | {r.object}")
        
        print(f"\n🎯 ENHANCED LEVEL3:")
        print(f"   Time: {time_enhanced:.2f}ms")
        print(f"   Relations: {len(kg_enhanced['relations'])}")
        if kg_enhanced['relations']:
            print(f"   Avg Confidence: {kg_enhanced['quality_metrics']['relation_avg_confidence']:.2f}")
            print(f"   Sample output:")
            for r in kg_enhanced['relations'][:5]:
                print(f"     - {r.subject} | {r.predicate} | {r.object} [conf={r.confidence:.2f}]")
        
        # Winner
        print(f"\n🏆 PERFORMANCE WINNER: ", end="")
        if time_enhanced < time_original:
            print(f"Enhanced Level3 ({time_enhanced:.2f}ms vs {time_original:.2f}ms) - {time_original/time_enhanced:.1f}x faster!")
        else:
            print(f"Level3 Original ({time_original:.2f}ms vs {time_enhanced:.2f}ms)")
            
        print(f"📈 QUANTITY WINNER: ", end="")
        if len(kg_original.relations) > len(kg_enhanced['relations']):
            print(f"Level3 Original ({len(kg_original.relations)} vs {len(kg_enhanced['relations'])} relations)")
        else:
            print(f"Enhanced Level3 ({len(kg_enhanced['relations'])} vs {len(kg_original.relations)} relations)")

if __name__ == "__main__":
    compare_extractors()
