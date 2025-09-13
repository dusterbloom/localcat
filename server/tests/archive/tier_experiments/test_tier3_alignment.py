#!/usr/bin/env python3
"""
Test Tier 3 prompt alignment and inconsistency issues
"""

import json
import time
from components.extraction.tiered_extractor import TieredRelationExtractor

class Tier3PromptTester:
    """Test Tier 3 prompt alignment and consistency"""
    
    def __init__(self):
        self.extractor = TieredRelationExtractor()
        
    def test_prompt_alignment(self):
        """Check if all prompts are aligned and consistent"""
        
        print("🔍 PROMPT ALIGNMENT CHECK")
        print("=" * 60)
        
        # Find all prompt-related methods
        import inspect
        
        # Check _extract_tier3 method prompts
        source = inspect.getsource(self.extractor._extract_tier3)
        
        print("📝 EXTRACT TIER 3 METHOD PROMPTS:")
        print("-" * 40)
        
        # Extract system prompts
        system_prompts = []
        lines = source.split('\n')
        in_prompt = False
        current_prompt = ""
        
        for line in lines:
            if 'system_prompt = ' in line and not in_prompt:
                in_prompt = True
                current_prompt = line.split('system_prompt = ')[1]
                if current_prompt.strip().endswith('"""'):
                    # Single line prompt
                    system_prompts.append(current_prompt.strip().strip('"""'))
                    in_prompt = False
            elif in_prompt:
                current_prompt += "\n" + line
                if '"""' in line:
                    # End of multi-line prompt
                    system_prompts.append(current_prompt.strip().strip('"""'))
                    in_prompt = False
        
        for i, prompt in enumerate(system_prompts):
            print(f"\nSystem Prompt {i+1}:")
            print(f"{prompt[:200]}..." if len(prompt) > 200 else prompt)
        
        # Check _extract_tier3_full method prompts
        try:
            full_source = inspect.getsource(self.extractor._extract_tier3_full)
            print(f"\n📝 EXTRACT TIER 3 FULL PROMPTS:")
            print("-" * 40)
            
            full_prompts = []
            lines = full_source.split('\n')
            in_prompt = False
            current_prompt = ""
            
            for line in lines:
                if 'system_prompt = ' in line and not in_prompt:
                    in_prompt = True
                    current_prompt = line.split('system_prompt = ')[1]
                    if current_prompt.strip().endswith('"""'):
                        full_prompts.append(current_prompt.strip().strip('"""'))
                        in_prompt = False
                elif in_prompt:
                    current_prompt += "\n" + line
                    if '"""' in line:
                        full_prompts.append(current_prompt.strip().strip('"""'))
                        in_prompt = False
            
            for i, prompt in enumerate(full_prompts):
                print(f"\nFull Method Prompt {i+1}:")
                print(f"{prompt[:200]}..." if len(prompt) > 200 else prompt)
        except:
            print("\n❌ Could not get _extract_tier3_full source")
    
    def test_inconsistency_cases(self):
        """Test what causes Tier 3 inconsistency"""
        
        print(f"\n🔍 INCONSISTENCY TEST")
        print("=" * 60)
        
        # Test cases with different complexity levels
        test_cases = [
            {
                "name": "Simple (should work)",
                "text": "Alice works at Tesla.",
                "expected_performance": "good"
            },
            {
                "name": "Medium (mixed results)",
                "text": "Alice, who works at Tesla as an engineer, drives a Model 3 and reports to Bob who is the CTO.",
                "expected_performance": "mixed"
            },
            {
                "name": "Complex (problematic)",
                "text": "Dr. Sarah Chen, the AI research director at OpenAI who joined the company in 2021 after completing her PhD at Stanford under the supervision of Dr. Michael Jordan, recently published a groundbreaking paper on neural architecture search that builds upon her previous work on transformer optimization done during her internship at Google Brain in 2019.",
                "expected_performance": "poor"
            },
            {
                "name": "Business (problematic)",
                "text": "Microsoft, founded by Bill Gates and Paul Allen in 1975, acquired LinkedIn for $26.2 billion in 2016. Satya Nadella, who became CEO in 2014, led this acquisition and previously worked at Sun Microsystems before joining Microsoft in 1992.",
                "expected_performance": "poor"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n📝 {test_case['name']}")
            print(f"Expected: {test_case['expected_performance']}")
            print(f"Text: {test_case['text'][:100]}...")
            print("-" * 50)
            
            # Test multiple runs to check consistency
            run_results = []
            
            for run in range(3):  # Test 3 times each
                try:
                    start_time = time.perf_counter()
                    
                    # Get Tier 1 entities first
                    tier1_result = self.extractor._extract_tier1(test_case['text'])
                    entities = [str(ent) for ent in tier1_result.entities]
                    
                    # Test Tier 3
                    tier3_result = self.extractor._extract_tier3(test_case['text'])
                    extraction_time = (time.perf_counter() - start_time) * 1000
                    
                    run_results.append({
                        'run': run + 1,
                        'entity_count': len(tier3_result.entities),
                        'relation_count': len(tier3_result.relationships),
                        'extraction_time': extraction_time,
                        'tier1_entities': len(entities),
                        'success_ratio': len(tier3_result.relationships) / max(len(tier3_result.entities), 1)
                    })
                    
                    print(f"   Run {run + 1}: {len(tier3_result.entities)} entities → {len(tier3_result.relationships)} relations ({extraction_time:.1f}ms)")
                    
                except Exception as e:
                    print(f"   Run {run + 1}: ERROR - {e}")
                    run_results.append({'run': run + 1, 'error': str(e)})
                
                # Small delay between runs
                time.sleep(0.5)
            
            # Analyze consistency
            successful_runs = [r for r in run_results if 'error' not in r]
            if successful_runs:
                avg_entities = sum(r['entity_count'] for r in successful_runs) / len(successful_runs)
                avg_relations = sum(r['relation_count'] for r in successful_runs) / len(successful_runs)
                avg_success_ratio = sum(r['success_ratio'] for r in successful_runs) / len(successful_runs)
                
                # Check consistency
                entity_variance = sum((r['entity_count'] - avg_entities) ** 2 for r in successful_runs) / len(successful_runs)
                relation_variance = sum((r['relation_count'] - avg_relations) ** 2 for r in successful_runs) / len(successful_runs)
                
                print(f"\n   📊 AVERAGES: {avg_entities:.1f} entities, {avg_relations:.1f} relations")
                print(f"   📈 Success Ratio: {avg_success_ratio:.2f}")
                print(f"   🎯 Consistency: {'HIGH' if entity_variance < 1 and relation_variance < 1 else 'LOW'}")
                print(f"   📊 Entity Variance: {entity_variance:.2f}")
                print(f"   📊 Relation Variance: {relation_variance:.2f}")
                
                results.append({
                    'test_case': test_case['name'],
                    'expected': test_case['expected_performance'],
                    'avg_entities': avg_entities,
                    'avg_relations': avg_relations,
                    'success_ratio': avg_success_ratio,
                    'consistency': 'HIGH' if entity_variance < 1 and relation_variance < 1 else 'LOW',
                    'runs': run_results
                })
            else:
                print(f"\n   ❌ ALL RUNS FAILED")
                results.append({
                    'test_case': test_case['name'],
                    'expected': test_case['expected_performance'],
                    'error': 'All runs failed'
                })
        
        # Summary
        print(f"\n🏆 INCONSISTENCY SUMMARY")
        print("=" * 60)
        
        for result in results:
            if 'error' not in result:
                status = "✅ GOOD" if result['success_ratio'] > 0.3 else "⚠️ POOR"
                consistency = result['consistency']
                print(f"{result['test_case']}: {status} ({result['success_ratio']:.2f} ratio, {consistency} consistency)")
            else:
                print(f"{result['test_case']}: ❌ FAILED")
        
        return results
    
    def test_entity_overload(self):
        """Test if entity count affects performance"""
        
        print(f"\n🔍 ENTITY OVERLOAD TEST")
        print("=" * 60)
        
        # Create test cases with increasing entity counts
        test_cases = [
            {
                "name": "2 entities",
                "text": "Alice works at Tesla."
            },
            {
                "name": "4 entities", 
                "text": "Alice works at Tesla. Bob manages Alice. Tesla is in California."
            },
            {
                "name": "6 entities",
                "text": "Alice works at Tesla in California. Bob manages Alice. Tesla was founded by Elon Musk. Elon Musk also founded SpaceX."
            },
            {
                "name": "8+ entities",
                "text": "Alice works at Tesla in California. Bob manages Alice. Tesla was founded by Elon Musk. Elon Musk also founded SpaceX. SpaceX launched rockets from Cape Canaveral. Cape Canaveral is in Florida. NASA works with SpaceX."
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📝 {test_case['name']}")
            print(f"Text: {test_case['text']}")
            print("-" * 40)
            
            try:
                # Test Tier 1 entity extraction
                tier1_result = self.extractor._extract_tier1(test_case['text'])
                entities = [str(ent) for ent in tier1_result.entities]
                
                print(f"   Tier 1 entities: {len(entities)}")
                print(f"   Entities: {entities}")
                
                # Test Tier 3
                tier3_result = self.extractor._extract_tier3(test_case['text'])
                
                print(f"   Tier 3 result: {len(tier3_result.entities)} entities → {len(tier3_result.relationships)} relations")
                print(f"   Success ratio: {len(tier3_result.relationships) / max(len(tier3_result.entities), 1):.2f}")
                
                # Show actual relationships
                if tier3_result.relationships:
                    print(f"   Relationships: {tier3_result.relationships[:3]}")  # Show first 3
                
            except Exception as e:
                print(f"   ERROR: {e}")

if __name__ == "__main__":
    tester = Tier3PromptTester()
    
    # Check prompt alignment
    tester.test_prompt_alignment()
    
    # Test inconsistency
    tester.test_inconsistency_cases()
    
    # Test entity overload
    tester.test_entity_overload()