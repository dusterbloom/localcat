#!/usr/bin/env python3
"""
Comprehensive Tier1 vs Tier2 Comparison
Full range analysis with proper GLiNER integration
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

class TierComparison:
    """Compare Tier1 vs Tier2 across the full complexity range"""
    
    def __init__(self):
        self.extractor = TieredRelationExtractor()
        
        # Test cases spanning the full complexity range
        self.test_cases = [
            {
                "name": "Tier1 - Simple Fact",
                "text": "Alice works at Tesla.",
                "complexity": "Very Low",
                "expected_entities": 2,
                "expected_relations": 1
            },
            {
                "name": "Tier1 - Basic Relation",
                "text": "Alice is an engineer at Tesla since 2020.",
                "complexity": "Low",
                "expected_entities": 3,
                "expected_relations": 2
            },
            {
                "name": "Tier1-Tier2 - Medium",
                "text": "Dr. Sarah Chen joined OpenAI as research director in 2021.",
                "complexity": "Medium",
                "expected_entities": 4,
                "expected_relations": 3
            },
            {
                "name": "Tier2 - Complex",
                "text": "Dr. Sarah Chen, who joined OpenAI as research director in 2021, previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li.",
                "complexity": "High",
                "expected_entities": 5,
                "expected_relations": 6
            },
            {
                "name": "Tier2 - Very Complex",
                "text": "Microsoft Corporation, founded by Bill Gates and Paul Allen in 1975, acquired LinkedIn Corporation for $26.2 billion in 2016 under the leadership of CEO Satya Nadella who had previously worked at Sun Microsystems before joining Microsoft in 1992.",
                "complexity": "Very High",
                "expected_entities": 8,
                "expected_relations": 8
            },
            {
                "name": "Tier2 - Extreme",
                "text": "Dr. Sarah Chen, the AI research director at OpenAI who joined the company in 2021 after completing her PhD at Stanford under the supervision of Dr. Michael Jordan, recently published a groundbreaking paper on neural architecture search that builds upon her previous work on transformer optimization done during her internship at Google Brain in 2019 where she collaborated with Dr. Fei-Fei Li before moving to Stanford University to teach machine learning courses while maintaining her research position at OpenAI.",
                "complexity": "Extreme",
                "expected_entities": 12,
                "expected_relations": 10
            }
        ]
    
    def test_tier_comparison(self):
        """Run comprehensive Tier1 vs Tier2 comparison"""
        
        print("🔍 COMPREHENSIVE TIER1 vs TIER2 COMPARISON")
        print("=" * 80)
        
        # Warm up models
        print("🔥 Warming up models...")
        self.extractor._extract_tier1("Warm up.")
        self.extractor._extract_tier2("Warm up.")
        
        results = []
        
        for test_case in self.test_cases:
            print(f"\n📝 {test_case['name']} ({test_case['complexity']})")
            print(f"Text: {test_case['text'][:100]}...")
            print(f"Expected: {test_case['expected_entities']} entities, {test_case['expected_relations']} relations")
            print("-" * 60)
            
            case_result = {
                'test_case': test_case['name'],
                'complexity': test_case['complexity'],
                'expected_entities': test_case['expected_entities'],
                'expected_relations': test_case['expected_relations']
            }
            
            # Test Tier1
            print("🏗️  TIER1:")
            try:
                start_time = time.perf_counter()
                tier1_result = self.extractor._extract_tier1(test_case['text'])
                tier1_time = (time.perf_counter() - start_time) * 1000
                
                tier1_entities = len(tier1_result.entities)
                tier1_relations = len(tier1_result.relationships)
                
                print(f"   ✅ {tier1_time:.1f}ms - {tier1_entities} entities, {tier1_relations} relations")
                
                case_result['tier1'] = {
                    'time': tier1_time,
                    'entities': tier1_entities,
                    'relations': tier1_relations,
                    'entity_ratio': tier1_entities / max(test_case['expected_entities'], 1),
                    'relation_ratio': tier1_relations / max(test_case['expected_relations'], 1)
                }
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                case_result['tier1'] = {
                    'time': float('inf'),
                    'entities': 0,
                    'relations': 0,
                    'error': str(e)
                }
            
            # Test Tier2
            print("🤖 TIER2:")
            try:
                start_time = time.perf_counter()
                tier2_result = self.extractor._extract_tier2(test_case['text'])
                tier2_time = (time.perf_counter() - start_time) * 1000
                
                tier2_entities = len(tier2_result.entities)
                tier2_relations = len(tier2_result.relationships)
                
                print(f"   ✅ {tier2_time:.1f}ms - {tier2_entities} entities, {tier2_relations} relations")
                
                case_result['tier2'] = {
                    'time': tier2_time,
                    'entities': tier2_entities,
                    'relations': tier2_relations,
                    'entity_ratio': tier2_entities / max(test_case['expected_entities'], 1),
                    'relation_ratio': tier2_relations / max(test_case['expected_relations'], 1)
                }
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                case_result['tier2'] = {
                    'time': float('inf'),
                    'entities': 0,
                    'relations': 0,
                    'error': str(e)
                }
            
            # Comparison
            if 'tier1' in case_result and 'tier2' in case_result:
                if case_result['tier1'].get('time', float('inf')) != float('inf') and case_result['tier2'].get('time', float('inf')) != float('inf'):
                    
                    tier1_stats = case_result['tier1']
                    tier2_stats = case_result['tier2']
                    
                    print(f"\n📊 COMPARISON:")
                    
                    # Speed comparison
                    if tier1_stats['time'] < tier2_stats['time']:
                        faster = "Tier1"
                        speedup = tier2_stats['time'] / tier1_stats['time']
                        print(f"   ⚡ {faster} is {speedup:.1f}x faster")
                    else:
                        faster = "Tier2"
                        speedup = tier1_stats['time'] / tier2_stats['time']
                        print(f"   ⚡ {faster} is {speedup:.1f}x faster")
                    
                    # Quality comparison
                    if tier1_stats['relations'] > tier2_stats['relations']:
                        better_quality = "Tier1"
                        quality_diff = tier1_stats['relations'] - tier2_stats['relations']
                        print(f"   🎯 {better_quality} extracted {quality_diff} more relations")
                    elif tier2_stats['relations'] > tier1_stats['relations']:
                        better_quality = "Tier2"
                        quality_diff = tier2_stats['relations'] - tier1_stats['relations']
                        print(f"   🎯 {better_quality} extracted {quality_diff} more relations")
                    else:
                        print(f"   🎯 Both extracted same number of relations")
                    
                    # Recommendation based on complexity
                    if test_case['complexity'] in ['Very Low', 'Low']:
                        recommended = "Tier1"
                    elif test_case['complexity'] in ['Medium']:
                        recommended = "Tier1 or Tier2"
                    else:
                        recommended = "Tier2"
                    
                    print(f"   💡 Recommended for {test_case['complexity']}: {recommended}")
            
            results.append(case_result)
        
        # Summary analysis
        print(f"\n🏆 SUMMARY - TIER1 vs TIER2 PERFORMANCE")
        print("=" * 80)
        
        # Analyze by complexity
        complexity_analysis = {}
        for complexity in ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']:
            complexity_results = [r for r in results if r['complexity'] == complexity]
            if complexity_results:
                complexity_analysis[complexity] = complexity_results[0]
        
        print("\n📈 PERFORMANCE BY COMPLEXITY:")
        for complexity, result in complexity_analysis.items():
            print(f"\n   {complexity}:")
            if 'tier1' in result and 'error' not in result['tier1']:
                t1 = result['tier1']
                print(f"     Tier1: {t1['time']:.0f}ms, {t1['entities']} entities, {t1['relations']} relations")
            if 'tier2' in result and 'error' not in result['tier2']:
                t2 = result['tier2']
                print(f"     Tier2: {t2['time']:.0f}ms, {t2['entities']} entities, {t2['relations']} relations")
        
        # Overall recommendations
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        
        tier1_wins = 0
        tier2_wins = 0
        ties = 0
        
        for result in results:
            if 'tier1' in result and 'tier2' in result:
                t1 = result['tier1']
                t2 = result['tier2']
                
                if 'error' not in t1 and 'error' not in t2:
                    # Simple scoring: speed + quality
                    t1_score = (1/t1['time'] * 1000) + t1['relations']
                    t2_score = (1/t2['time'] * 1000) + t2['relations']
                    
                    if t1_score > t2_score:
                        tier1_wins += 1
                    elif t2_score > t1_score:
                        tier2_wins += 1
                    else:
                        ties += 1
        
        print(f"   🥊 Tier1 wins: {tier1_wins}")
        print(f"   🤖 Tier2 wins: {tier2_wins}")
        print(f"   🤝 Ties: {ties}")
        
        # Complexity-based routing recommendation
        print(f"\n   🎯 ROUTING STRATEGY:")
        print(f"   • Very Low/Low complexity → Use Tier1 (faster, simpler)")
        print(f"   • Medium complexity → Use Tier1 or Tier2 (similar performance)")
        print(f"   • High/Very High/Extreme complexity → Use Tier2 (better quality)")
        
        return results

if __name__ == "__main__":
    comparison = TierComparison()
    results = comparison.test_tier_comparison()