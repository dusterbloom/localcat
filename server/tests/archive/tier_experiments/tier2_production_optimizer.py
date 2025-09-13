#!/usr/bin/env python3
"""
Production Tier2 Optimization Strategy
Focus on speed vs quality trade-offs for voice assistant use cases
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

class Tier2Optimizer:
    """Optimize Tier2 for production voice assistant scenarios"""
    
    def __init__(self):
        self.test_cases = [
            {
                "name": "Simple Conversation",
                "text": "Alice works at Tesla.",
                "expected_entities": 2,
                "expected_relations": 1
            },
            {
                "name": "Medium Conversation", 
                "text": "Dr. Sarah Chen joined OpenAI as research director in 2021.",
                "expected_entities": 3,
                "expected_relations": 3
            },
            {
                "name": "Complex Conversation",
                "text": "Dr. Sarah Chen, who joined OpenAI as research director in 2021, previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li.",
                "expected_entities": 4,
                "expected_relations": 6
            }
        ]
    
    def test_production_configs(self):
        """Test different production configurations"""
        
        print("🚀 PRODUCTION TIER2 OPTIMIZATION")
        print("=" * 60)
        
        # Production-focused configurations
        configs = [
            {
                "name": "MAX SPEED (SRL Only)",
                "config": {"enable_srl": True, "enable_coref": False, "enable_gliner": False},
                "target": "< 1000ms"
            },
            {
                "name": "BALANCED (No GLiNER)",
                "config": {"enable_srl": True, "enable_coref": True, "enable_gliner": False},
                "target": "< 1500ms"
            },
            {
                "name": "MAX QUALITY (Full)",
                "config": {"enable_srl": True, "enable_coref": True, "enable_gliner": True},
                "target": "< 2000ms"
            }
        ]
        
        # Warm up models first
        print("🔥 Warming up models...")
        warmup_extractor = TieredRelationExtractor()
        warmup_extractor._extract_tier2("Warm up test.")
        
        results = {}
        
        for config in configs:
            print(f"\n🧪 {config['name']} (Target: {config['target']})")
            print("-" * 50)
            
            config_results = []
            
            for test_case in self.test_cases:
                try:
                    # Create extractor with specific config
                    extractor = TieredRelationExtractor(**config['config'])
                    
                    # Test multiple times for consistency
                    times = []
                    relation_counts = []
                    
                    for _ in range(3):
                        start = time.perf_counter()
                        result = extractor._extract_tier2(test_case['text'])
                        elapsed = (time.perf_counter() - start) * 1000
                        
                        times.append(elapsed)
                        relation_counts.append(len(result.relationships))
                    
                    avg_time = sum(times) / len(times)
                    avg_relations = sum(relation_counts) / len(relation_counts)
                    
                    meets_target = avg_time < float(config['target'].replace('< ', '').replace('ms', ''))
                    
                    print(f"  {test_case['name']}: {avg_time:.0f}ms, {avg_relations:.0f} relations {'✅' if meets_target else '❌'}")
                    
                    config_results.append({
                        'test_case': test_case['name'],
                        'time': avg_time,
                        'relations': avg_relations,
                        'meets_target': meets_target
                    })
                    
                except Exception as e:
                    print(f"  {test_case['name']}: ERROR - {e}")
                    config_results.append({
                        'test_case': test_case['name'],
                        'time': float('inf'),
                        'relations': 0,
                        'error': str(e)
                    })
            
            # Calculate config performance
            valid_results = [r for r in config_results if 'error' not in r]
            if valid_results:
                avg_time = sum(r['time'] for r in valid_results) / len(valid_results)
                avg_relations = sum(r['relations'] for r in valid_results) / len(valid_results)
                success_rate = sum(1 for r in valid_results if r['meets_target']) / len(valid_results)
                
                target_met = avg_time < float(config['target'].replace('< ', '').replace('ms', ''))
                
                print(f"\n  📊 Config Summary:")
                print(f"     Avg Time: {avg_time:.0f}ms")
                print(f"     Avg Relations: {avg_relations:.0f}")
                print(f"     Target Met: {'✅' if target_met else '❌'}")
                print(f"     Success Rate: {success_rate*100:.0f}%")
                
                results[config['name']] = {
                    'avg_time': avg_time,
                    'avg_relations': avg_relations,
                    'target': config['target'],
                    'target_met': target_met,
                    'success_rate': success_rate
                }
        
        # Find optimal configuration
        print(f"\n🏆 PRODUCTION RECOMMENDATIONS")
        print("=" * 60)
        
        # Rank by target success rate, then by speed
        ranked_configs = sorted(results.items(), 
                               key=lambda x: (-x[1]['success_rate'], x[1]['avg_time']))
        
        for i, (name, stats) in enumerate(ranked_configs, 1):
            print(f"{i}. {name}")
            print(f"   ⏱️  {stats['avg_time']:.0f}ms (target: {stats['target']})")
            print(f"   🎯 {stats['avg_relations']:.0f} avg relations")
            print(f"   ✅ {stats['success_rate']*100:.0f}% target success rate")
            print()
        
        # Voice assistant specific recommendations
        best_config = ranked_configs[0]
        
        print("💡 VOICE ASSISTANT RECOMMENDATIONS:")
        print(f"   Use: {best_config[0]}")
        print(f"   Expected response time: ~{best_config[1]['avg_time']:.0f}ms + processing")
        print(f"   Quality level: {best_config[1]['avg_relations']:.0f} relations avg")
        
        if best_config[1]['avg_time'] < 1000:
            print("   ✅ Suitable for real-time conversation")
        elif best_config[1]['avg_time'] < 1500:
            print("   ⚠️  Acceptable for conversation with slight delay")
        else:
            print("   ❌ May be too slow for real-time use")
        
        return results

if __name__ == "__main__":
    optimizer = Tier2Optimizer()
    results = optimizer.test_production_configs()