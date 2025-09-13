#!/usr/bin/env python3

"""
Comprehensive Extraction System Performance Test
Tests all three tiers with the finalized hybrid approach
"""

import time
from components.extraction.tiered_extractor import TieredRelationExtractor

def test_comprehensive_performance():
    """Test all three tiers with representative sentences"""
    
    extractor = TieredRelationExtractor()
    
    # Test sentences for each complexity level
    test_cases = [
        # Simple sentences (Tier 1)
        {
            "text": "John studied at Reed College.",
            "expected_tier": 1,
            "category": "Simple"
        },
        {
            "text": "Sarah works at Google.",
            "expected_tier": 1,
            "category": "Simple"
        },
        {
            "text": "The cat sat on the mat.",
            "expected_tier": 1,
            "category": "Simple"
        },
        
        # Medium sentences (Tier 2)
        {
            "text": "Alice studied computer science at MIT and now works at Google.",
            "expected_tier": 2,
            "category": "Medium"
        },
        {
            "text": "The company announced its quarterly earnings after the market closed.",
            "expected_tier": 2,
            "category": "Medium"
        },
        {
            "text": "Researchers from Harvard discovered a new method for quantum computing.",
            "expected_tier": 2,
            "category": "Medium"
        },
        
        # Complex sentences (Tier 3)
        {
            "text": "The sophisticated artificial intelligence system that Dr. Sarah Williams developed at MIT revolutionized how autonomous vehicles navigate complex urban environments while maintaining safety standards.",
            "expected_tier": 3,
            "category": "Complex"
        },
        {
            "text": "When the international climate summit concluded, representatives from 195 countries reached a historic agreement that addressed carbon emissions reduction targets while balancing economic development concerns.",
            "expected_tier": 3,
            "category": "Complex"
        },
        {
            "text": "Despite challenging economic conditions, the innovative startup founded by recent Harvard graduates successfully secured funding from multiple venture capital firms to develop their groundbreaking machine learning technology.",
            "expected_tier": 3,
            "category": "Complex"
        }
    ]
    
    print("=== Comprehensive Extraction System Performance Test ===\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        expected_tier = test_case["expected_tier"]
        category = test_case["category"]
        
        print(f"Test {i+1} ({category}): {text[:60]}...")
        print("-" * 70)
        
        # Analyze complexity first
        complexity = extractor.analyze_complexity(text)
        print(f"Analyzed complexity: {complexity.level.name} (confidence: {complexity.confidence:.2f})")
        print(f"Expected tier: {expected_tier}")
        
        # Time the extraction
        start_time = time.perf_counter()
        
        try:
            result = extractor.extract(text)
            elapsed_time = (time.perf_counter() - start_time) * 1000
            
            print(f"Actual tier used: {result.tier_used}")
            print(f"Extraction time: {elapsed_time:.1f}ms")
            print(f"Entities: {len(result.entities)}")
            print(f"Relationships: {len(result.relationships)}")
            print(f"Confidence: {result.confidence:.2f}")
            
            # Check if routing was correct
            routing_correct = result.tier_used == expected_tier
            print(f"Routing correct: {'✅' if routing_correct else '❌'}")
            
            if result.relationships:
                print("Sample relationships:")
                for rel in result.relationships[:2]:
                    print(f"  - {rel}")
            
            results.append({
                'text': text,
                'category': category,
                'expected_tier': expected_tier,
                'actual_tier': result.tier_used,
                'routing_correct': routing_correct,
                'time_ms': elapsed_time,
                'entities': len(result.entities),
                'relationships': len(result.relationships),
                'confidence': result.confidence,
                'success': len(result.relationships) > 0
            })
            
        except Exception as e:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"ERROR: {e}")
            print(f"Failed after: {elapsed_time:.1f}ms")
            
            results.append({
                'text': text,
                'category': category,
                'expected_tier': expected_tier,
                'actual_tier': 'ERROR',
                'routing_correct': False,
                'time_ms': elapsed_time,
                'entities': 0,
                'relationships': 0,
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Comprehensive Summary
    print("=== COMPREHENSIVE PERFORMANCE SUMMARY ===")
    
    # Overall stats
    total_tests = len(results)
    successful_tests = len([r for r in results if r['success']])
    routing_accuracy = len([r for r in results if r['routing_correct']]) / total_tests * 100
    
    print(f"Total tests: {total_tests}")
    print(f"Successful extractions: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"Routing accuracy: {routing_accuracy:.1f}%")
    
    # Performance by tier
    print("\n--- Performance by Tier ---")
    for tier in [1, 2, 3]:
        tier_results = [r for r in results if r['actual_tier'] == tier]
        if tier_results:
            avg_time = sum(r['time_ms'] for r in tier_results) / len(tier_results)
            avg_entities = sum(r['entities'] for r in tier_results) / len(tier_results)
            avg_relationships = sum(r['relationships'] for r in tier_results) / len(tier_results)
            avg_confidence = sum(r['confidence'] for r in tier_results) / len(tier_results)
            success_rate = len([r for r in tier_results if r['success']]) / len(tier_results) * 100
            
            print(f"Tier {tier}:")
            print(f"  Tests: {len(tier_results)}")
            print(f"  Avg time: {avg_time:.1f}ms")
            print(f"  Avg entities: {avg_entities:.1f}")
            print(f"  Avg relationships: {avg_relationships:.1f}")
            print(f"  Avg confidence: {avg_confidence:.2f}")
            print(f"  Success rate: {success_rate:.1f}%")
    
    # Performance by complexity
    print("\n--- Performance by Complexity ---")
    for category in ["Simple", "Medium", "Complex"]:
        cat_results = [r for r in results if r['category'] == category]
        if cat_results:
            avg_time = sum(r['time_ms'] for r in cat_results) / len(cat_results)
            avg_entities = sum(r['entities'] for r in cat_results) / len(cat_results)
            avg_relationships = sum(r['relationships'] for r in cat_results) / len(cat_results)
            
            print(f"{category}:")
            print(f"  Avg time: {avg_time:.1f}ms")
            print(f"  Avg entities: {avg_entities:.1f}")
            print(f"  Avg relationships: {avg_relationships:.1f}")
    
    # Overall assessment
    print("\n=== SYSTEM ASSESSMENT ===")
    
    if routing_accuracy >= 90 and successful_tests >= total_tests * 0.8:
        print("🎉 EXCELLENT: The three-tier extraction system is working optimally!")
        print("✅ Routing accuracy is excellent")
        print("✅ All tiers are performing well")
        print("✅ Hybrid approach successful for both Tier 2 and Tier 3")
    elif routing_accuracy >= 70 and successful_tests >= total_tests * 0.6:
        print("👍 GOOD: The extraction system is working well with minor issues")
        print("✅ Most routing is correct")
        print("✅ Most extractions successful")
    else:
        print("⚠️  NEEDS WORK: The extraction system has issues that need attention")
        print("❌ Poor routing accuracy")
        print("❌ Low success rate")

if __name__ == "__main__":
    test_comprehensive_performance()