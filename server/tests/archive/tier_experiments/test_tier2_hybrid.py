#!/usr/bin/env python3

"""
Test Tier 2 Hybrid approach
Tests the new hybrid Tier 2 that uses Tier 1 entities + JSON relationships
"""

import time
from components.extraction.tiered_extractor import TieredRelationExtractor

def test_tier2_hybrid():
    """Test the new Tier 2 hybrid approach"""
    
    extractor = TieredRelationExtractor()
    
    # Test medium complexity sentences that should trigger Tier 2
    test_sentences = [
        "Alice studied computer science at MIT and now works at Google.",
        "The company announced its quarterly earnings after the market closed.",
        "Researchers from Harvard discovered a new method for quantum computing.",
        "Despite the challenges, the team successfully completed the project ahead of schedule.",
        "The professor published her findings in a prestigious scientific journal."
    ]
    
    print("=== Tier 2 Hybrid Approach Test ===\n")
    
    results = []
    
    for i, sentence in enumerate(test_sentences):
        print(f"Test {i+1}: {sentence}")
        print("-" * 50)
        
        # First, check complexity analysis
        complexity = extractor.analyze_complexity(sentence)
        print(f"Complexity: {complexity.level.name}")
        print(f"Confidence: {complexity.confidence:.2f}")
        
        # Time the extraction
        start_time = time.perf_counter()
        
        try:
            result = extractor.extract(sentence)
            elapsed_time = (time.perf_counter() - start_time) * 1000
            
            print(f"Tier used: {result.tier_used}")
            print(f"Extraction time: {elapsed_time:.1f}ms")
            print(f"Entities: {len(result.entities)}")
            print(f"Relationships: {len(result.relationships)}")
            print(f"Confidence: {result.confidence:.2f}")
            
            if result.entities:
                print("Entities:")
                for entity in result.entities[:5]:  # Show first 5
                    print(f"  - {entity}")
            
            if result.relationships:
                print("Relationships:")
                for rel in result.relationships[:3]:  # Show first 3
                    print(f"  - {rel}")
            
            results.append({
                'sentence': sentence,
                'complexity_level': complexity.level.name,
                'tier_used': result.tier_used,
                'time_ms': elapsed_time,
                'entities': len(result.entities),
                'relationships': len(result.relationships),
                'confidence': result.confidence,
                'success': result.tier_used == 2 and len(result.relationships) > 0
            })
            
        except Exception as e:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"ERROR: {e}")
            print(f"Failed after: {elapsed_time:.1f}ms")
            
            results.append({
                'sentence': sentence,
                'complexity_level': complexity.level.name,
                'tier_used': 'ERROR',
                'time_ms': elapsed_time,
                'entities': 0,
                'relationships': 0,
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=== Tier 2 Hybrid Performance Summary ===")
    
    tier2_results = [r for r in results if r['tier_used'] == 2]
    successful_tier2 = [r for r in tier2_results if r['success']]
    
    print(f"Total tests: {len(results)}")
    print(f"Tier 2 calls: {len(tier2_results)}")
    print(f"Successful Tier 2: {len(successful_tier2)}")
    
    if tier2_results:
        times = [r['time_ms'] for r in tier2_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nTier 2 Performance:")
        print(f"Average time: {avg_time:.1f}ms")
        print(f"Min time: {min_time:.1f}ms")
        print(f"Max time: {max_time:.1f}ms")
        
        avg_entities = sum(r['entities'] for r in tier2_results) / len(tier2_results)
        avg_relationships = sum(r['relationships'] for r in tier2_results) / len(tier2_results)
        avg_confidence = sum(r['confidence'] for r in tier2_results) / len(tier2_results)
        
        print(f"Average entities: {avg_entities:.1f}")
        print(f"Average relationships: {avg_relationships:.1f}")
        print(f"Average confidence: {avg_confidence:.2f}")
    
    # Success rate
    success_rate = len(successful_tier2) / len(results) * 100 if results else 0
    print(f"\nTier 2 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 60:
        print("✅ Tier 2 hybrid approach is working well!")
    elif success_rate >= 30:
        print("⚠️  Tier 2 hybrid approach shows promise but needs improvement")
    else:
        print("❌ Tier 2 hybrid approach needs significant work")
    
    # Show what triggered other tiers
    other_tiers = [r for r in results if r['tier_used'] not in [2, 'ERROR']]
    if other_tiers:
        print(f"\nOther tiers used: {len(other_tiers)}")
        for r in other_tiers:
            print(f"  Tier {r['tier_used']}: {r['sentence'][:60]}... ({r['time_ms']:.1f}ms)")

if __name__ == "__main__":
    test_tier2_hybrid()