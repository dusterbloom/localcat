#!/usr/bin/env python3

"""
Test Tier 3 performance optimization
Measures current performance and identifies bottlenecks
"""

import time
import json
from typing import Dict, List, Tuple
from components.extraction.tiered_extractor import TieredRelationExtractor, ComplexityLevel

def test_tier3_performance():
    """Test current Tier 3 performance and identify bottlenecks"""
    
    # Initialize extractor
    extractor = TieredRelationExtractor()
    
    # Test sentences that should trigger Tier 3
    test_sentences = [
        "The sophisticated artificial intelligence system that Dr. Sarah Williams developed at MIT revolutionized how autonomous vehicles navigate complex urban environments while maintaining safety standards.",
        "Despite the challenging economic conditions and unprecedented market volatility, the innovative startup founded by recent Harvard graduates successfully secured funding from multiple venture capital firms.",
        "The comprehensive research paper published by the Stanford team demonstrated how quantum computing algorithms could potentially solve optimization problems that were previously considered intractable for classical computers.",
        "When the international climate summit concluded, representatives from 195 countries reached a historic agreement that addressed carbon emissions reduction targets while balancing economic development concerns.",
        "The groundbreaking medical treatment developed through collaboration between researchers at Oxford and Cambridge universities showed remarkable efficacy in clinical trials involving patients with previously untreatable neurological disorders."
    ]
    
    print("=== Tier 3 Performance Analysis ===\n")
    
    results = []
    
    for i, sentence in enumerate(test_sentences):
        print(f"Test {i+1}: {sentence[:80]}...")
        
        # Analyze complexity first
        complexity = extractor.analyze_complexity(sentence)
        print(f"  Complexity: {complexity.level.name}")
        print(f"  Confidence: {complexity.confidence:.2f}")
        
        # Time the extraction
        start_time = time.perf_counter()
        
        try:
            result = extractor.extract(sentence)
            elapsed_time = (time.perf_counter() - start_time) * 1000
            
            print(f"  Tier used: {result.tier_used}")
            print(f"  Extraction time: {elapsed_time:.1f}ms")
            print(f"  Entities: {len(result.entities)}")
            print(f"  Relationships: {len(result.relationships)}")
            print(f"  Confidence: {result.confidence:.2f}")
            
            if result.relationships:
                print("  Relationships:")
                for rel in result.relationships[:3]:  # Show first 3
                    print(f"    - {rel}")
            
            results.append({
                'sentence': sentence,
                'complexity_level': complexity.level.name,
                'tier_used': result.tier_used,
                'time_ms': elapsed_time,
                'entities': len(result.entities),
                'relationships': len(result.relationships),
                'confidence': result.confidence
            })
            
        except Exception as e:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"  ERROR: {e}")
            print(f"  Failed after: {elapsed_time:.1f}ms")
            
            results.append({
                'sentence': sentence,
                'complexity_level': complexity.level.name,
                'tier_used': 'ERROR',
                'time_ms': elapsed_time,
                'entities': 0,
                'relationships': 0,
                'confidence': 0.0,
                'error': str(e)
            })
        
        print()
    
    # Summary statistics
    print("=== Performance Summary ===")
    tier3_results = [r for r in results if r['tier_used'] == 3]
    
    if tier3_results:
        times = [r['time_ms'] for r in tier3_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Tier 3 calls: {len(tier3_results)}")
        print(f"Average time: {avg_time:.1f}ms")
        print(f"Min time: {min_time:.1f}ms")
        print(f"Max time: {max_time:.1f}ms")
        
        avg_entities = sum(r['entities'] for r in tier3_results) / len(tier3_results)
        avg_relationships = sum(r['relationships'] for r in tier3_results) / len(tier3_results)
        avg_confidence = sum(r['confidence'] for r in tier3_results) / len(tier3_results)
        
        print(f"Average entities: {avg_entities:.1f}")
        print(f"Average relationships: {avg_relationships:.1f}")
        print(f"Average confidence: {avg_confidence:.2f}")
    else:
        print("No Tier 3 calls made!")
    
    # Show what triggered other tiers
    other_tiers = [r for r in results if r['tier_used'] != 3]
    if other_tiers:
        print(f"\nOther tiers used: {len(other_tiers)}")
        for r in other_tiers:
            print(f"  Tier {r['tier_used']}: {r['sentence'][:60]}... ({r['time_ms']:.1f}ms)")
    
    # Analysis
    print("\n=== Bottleneck Analysis ===")
    if tier3_results:
        if avg_time > 4000:
            print("❌ Major bottleneck: Tier 3 is too slow (>4s)")
            print("   Potential issues:")
            print("   - Model loading time not cached")
            print("   - Network latency to LLM server")
            print("   - Model size too large (1B parameters)")
            print("   - Prompt too verbose")
        elif avg_time > 2000:
            print("⚠️  Moderate bottleneck: Tier 3 acceptable but could be faster")
        else:
            print("✅ Tier 3 performance is good")
    
    # Check timeout settings
    print(f"\nCurrent timeout: {extractor.llm_timeout_ms}ms")
    if extractor.llm_timeout_ms < 5000:
        print("⚠️  Timeout might be too restrictive for complex sentences")
    
    return results

if __name__ == "__main__":
    results = test_tier3_performance()