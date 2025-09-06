#!/usr/bin/env python3
"""
Confidence Scoring Fix Testing
=============================

Tests different approaches to fix the confidence scoring system that's filtering out valid facts.
The current system is marking facts like "(you, has, cat)" as 0.00 confidence.

Approaches to test:
1. Lower confidence threshold (0.3 → 0.0) 
2. Bypass confidence for basic patterns
3. Fix confidence algorithm itself
4. Use different quality scoring
"""

import os
import sys
import time
import tempfile
from typing import List, Tuple, Dict, Any

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

# Test sentences that are currently failing
FAILING_SENTENCES = [
    "I also have a cat named Whiskers who is 2 years old and loves to play with yarn.",
    "My favorite color is purple and my lucky number is 7.",
    "I drive a Tesla Model 3 that I bought last year.",
    "Luna is 3 years old and very smart.",
    "My brother Tom teaches philosophy at Reed College in Portland.",
]

class ConfidenceFixTester:
    def __init__(self):
        self.temp_dir = None
        self.original_env = {}
        
    def setup(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        print(f"🏗️  Test storage created at: {self.temp_dir}")
        
        # Store original env values
        self.original_env = {
            'HOTMEM_CONFIDENCE_THRESHOLD': os.getenv('HOTMEM_CONFIDENCE_THRESHOLD'),
            'HOTMEM_EXTRA_CONFIDENCE': os.getenv('HOTMEM_EXTRA_CONFIDENCE'),
        }
    
    def cleanup(self):
        """Clean up test data and restore environment"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
        
        # Restore original env
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
                
        print(f"🧹 Cleaned up test environment")
    
    def create_hot_memory(self) -> HotMemory:
        """Create fresh HotMemory instance"""
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'test_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'test_graph.lmdb')
        )
        store = MemoryStore(paths)
        hot_memory = HotMemory(store)
        hot_memory.prewarm('en')
        return hot_memory
    
    def test_fix_1_lower_threshold(self) -> Dict[str, Any]:
        """Fix 1: Lower confidence threshold from 0.3 to 0.0"""
        print("\n🔧 FIX 1: Lower Confidence Threshold (0.3 → 0.0)")
        print("-" * 50)
        
        # Set very low threshold
        os.environ['HOTMEM_CONFIDENCE_THRESHOLD'] = '0.0'
        os.environ['HOTMEM_EXTRA_CONFIDENCE'] = 'false'
        
        hot_memory = self.create_hot_memory()
        results = []
        
        for i, sentence in enumerate(FAILING_SENTENCES, 1):
            print(f"\n🧪 Test {i}: {sentence[:50]}...")
            
            start_time = time.perf_counter()
            bullets, triples = hot_memory.process_turn(sentence, session_id="fix1", turn_id=i)
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Extracted {len(triples)} facts in {extraction_time:.1f}ms")
            
            # Show extracted facts
            if triples:
                print("✅ Facts extracted:")
                for j, (s, r, d) in enumerate(triples[:3]):
                    print(f"  {j+1}. ({s}) -[{r}]-> ({d})")
                if len(triples) > 3:
                    print(f"  ... and {len(triples)-3} more")
            else:
                print("❌ No facts extracted")
            
            results.append({
                'sentence': sentence,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'extraction_time_ms': extraction_time,
                'facts': triples[:5]  # Store sample facts
            })
        
        return {
            'method': 'Lower Threshold (0.0)',
            'total_facts': sum(r['facts_count'] for r in results),
            'avg_time_ms': sum(r['extraction_time_ms'] for r in results) / len(results),
            'success_rate': len([r for r in results if r['facts_count'] > 0]) / len(results),
            'results': results
        }
    
    def test_fix_2_bypass_basic_patterns(self) -> Dict[str, Any]:
        """Fix 2: Test with EXTRA_CONFIDENCE disabled"""
        print("\n🔧 FIX 2: Disable Extra Confidence Scoring")
        print("-" * 50)
        
        # Keep normal threshold but disable extra confidence
        os.environ['HOTMEM_CONFIDENCE_THRESHOLD'] = '0.3'
        os.environ['HOTMEM_EXTRA_CONFIDENCE'] = 'false'
        
        hot_memory = self.create_hot_memory()
        results = []
        
        for i, sentence in enumerate(FAILING_SENTENCES, 1):
            print(f"\n🧪 Test {i}: {sentence[:50]}...")
            
            start_time = time.perf_counter()
            bullets, triples = hot_memory.process_turn(sentence, session_id="fix2", turn_id=i)
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Extracted {len(triples)} facts in {extraction_time:.1f}ms")
            
            # Show extracted facts
            if triples:
                print("✅ Facts extracted:")
                for j, (s, r, d) in enumerate(triples[:3]):
                    print(f"  {j+1}. ({s}) -[{r}]-> ({d})")
                if len(triples) > 3:
                    print(f"  ... and {len(triples)-3} more")
            else:
                print("❌ No facts extracted")
            
            results.append({
                'sentence': sentence,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'extraction_time_ms': extraction_time,
                'facts': triples[:5]
            })
        
        return {
            'method': 'Disable Extra Confidence',
            'total_facts': sum(r['facts_count'] for r in results),
            'avg_time_ms': sum(r['extraction_time_ms'] for r in results) / len(results),
            'success_rate': len([r for r in results if r['facts_count'] > 0]) / len(results),
            'results': results
        }
    
    def test_fix_3_basic_fact_bypass(self) -> Dict[str, Any]:
        """Fix 3: Test the new basic fact bypass logic"""
        print("\n🔧 FIX 3: Basic Fact Bypass (New Code)")
        print("-" * 50)
        
        # Use the new bypass functionality  
        os.environ['HOTMEM_CONFIDENCE_THRESHOLD'] = '0.3'  # Keep normal threshold
        os.environ['HOTMEM_EXTRA_CONFIDENCE'] = 'true'     # Keep extra confidence
        os.environ['HOTMEM_BYPASS_CONFIDENCE_FOR_BASIC'] = 'true'  # Enable bypass
        os.environ['HOTMEM_CONFIDENCE_FLOOR_BASIC'] = '0.6'  # Floor for basic facts
        
        hot_memory = self.create_hot_memory()
        results = []
        
        for i, sentence in enumerate(FAILING_SENTENCES, 1):
            print(f"\n🧪 Test {i}: {sentence[:50]}...")
            
            start_time = time.perf_counter()
            bullets, triples = hot_memory.process_turn(sentence, session_id="fix3", turn_id=i)
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Extracted {len(triples)} facts in {extraction_time:.1f}ms")
            
            # Show extracted facts
            if triples:
                print("✅ Facts extracted:")
                for j, (s, r, d) in enumerate(triples[:3]):
                    print(f"  {j+1}. ({s}) -[{r}]-> ({d})")
                if len(triples) > 3:
                    print(f"  ... and {len(triples)-3} more")
            else:
                print("❌ No facts extracted")
            
            results.append({
                'sentence': sentence,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'extraction_time_ms': extraction_time,
                'facts': triples[:5]
            })
        
        return {
            'method': 'Basic Fact Bypass (New)',
            'total_facts': sum(r['facts_count'] for r in results),
            'avg_time_ms': sum(r['extraction_time_ms'] for r in results) / len(results),
            'success_rate': len([r for r in results if r['facts_count'] > 0]) / len(results),
            'results': results
        }
    
    def test_fix_4_mixed_approach(self) -> Dict[str, Any]:
        """Fix 3: Lower threshold + disable extra confidence"""
        print("\n🔧 FIX 3: Mixed Approach (Low Threshold + No Extra Confidence)")
        print("-" * 50)
        
        # Combine approaches
        os.environ['HOTMEM_CONFIDENCE_THRESHOLD'] = '0.1'  # Lower but not zero
        os.environ['HOTMEM_EXTRA_CONFIDENCE'] = 'false'
        
        hot_memory = self.create_hot_memory()
        results = []
        
        for i, sentence in enumerate(FAILING_SENTENCES, 1):
            print(f"\n🧪 Test {i}: {sentence[:50]}...")
            
            start_time = time.perf_counter()
            bullets, triples = hot_memory.process_turn(sentence, session_id="fix3", turn_id=i)
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Extracted {len(triples)} facts in {extraction_time:.1f}ms")
            
            # Show extracted facts
            if triples:
                print("✅ Facts extracted:")
                for j, (s, r, d) in enumerate(triples[:3]):
                    print(f"  {j+1}. ({s}) -[{r}]-> ({d})")
                if len(triples) > 3:
                    print(f"  ... and {len(triples)-3} more")
            else:
                print("❌ No facts extracted")
            
            results.append({
                'sentence': sentence,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'extraction_time_ms': extraction_time,
                'facts': triples[:5]
            })
        
        return {
            'method': 'Mixed Approach (0.1 threshold)',
            'total_facts': sum(r['facts_count'] for r in results),
            'avg_time_ms': sum(r['extraction_time_ms'] for r in results) / len(results),
            'success_rate': len([r for r in results if r['facts_count'] > 0]) / len(results),
            'results': results
        }
    
    def test_baseline_current(self) -> Dict[str, Any]:
        """Baseline: Current broken configuration"""
        print("\n📊 BASELINE: Current Configuration (Broken)")
        print("-" * 50)
        
        # Current settings
        os.environ['HOTMEM_CONFIDENCE_THRESHOLD'] = '0.3'
        os.environ['HOTMEM_EXTRA_CONFIDENCE'] = 'true'
        
        hot_memory = self.create_hot_memory()
        results = []
        
        for i, sentence in enumerate(FAILING_SENTENCES, 1):
            print(f"\n🧪 Test {i}: {sentence[:50]}...")
            
            start_time = time.perf_counter()
            bullets, triples = hot_memory.process_turn(sentence, session_id="baseline", turn_id=i)
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Extracted {len(triples)} facts in {extraction_time:.1f}ms")
            
            if triples:
                print("✅ Facts extracted:")
                for j, (s, r, d) in enumerate(triples[:3]):
                    print(f"  {j+1}. ({s}) -[{r}]-> ({d})")
            else:
                print("❌ No facts extracted")
            
            results.append({
                'sentence': sentence,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'extraction_time_ms': extraction_time,
                'facts': triples[:5]
            })
        
        return {
            'method': 'Current (Broken)',
            'total_facts': sum(r['facts_count'] for r in results),
            'avg_time_ms': sum(r['extraction_time_ms'] for r in results) / len(results),
            'success_rate': len([r for r in results if r['facts_count'] > 0]) / len(results),
            'results': results
        }
    
    def print_comparison_results(self, all_results: List[Dict[str, Any]]):
        """Print comprehensive comparison of all fixes"""
        print("\n📈 CONFIDENCE FIX COMPARISON RESULTS")
        print("=" * 60)
        
        # Summary table
        print("\n📊 Performance Summary:")
        print(f"{'Method':<25} {'Facts':<8} {'Success%':<10} {'Avg Time':<10}")
        print("-" * 55)
        
        for result in all_results:
            success_pct = result['success_rate'] * 100
            print(f"{result['method']:<25} {result['total_facts']:<8} {success_pct:<10.1f} {result['avg_time_ms']:<10.1f}ms")
        
        # Find best approach
        best_by_facts = max(all_results, key=lambda x: x['total_facts'])
        best_by_success = max(all_results, key=lambda x: x['success_rate'])
        
        print(f"\n🏆 Best Results:")
        print(f"  Most Facts Extracted: {best_by_facts['method']} ({best_by_facts['total_facts']} facts)")
        print(f"  Highest Success Rate: {best_by_success['method']} ({best_by_success['success_rate']*100:.1f}%)")
        
        # Detailed breakdown
        print(f"\n🔍 Per-Sentence Breakdown:")
        for i, sentence in enumerate(FAILING_SENTENCES):
            print(f"\n  Sentence {i+1}: '{sentence[:40]}...'")
            for result in all_results:
                facts = result['results'][i]['facts_count']
                print(f"    {result['method']:<25}: {facts} facts")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_by_facts['total_facts'] > 0:
            print(f"  ✅ USE: {best_by_facts['method']}")
            print(f"     - Extracts {best_by_facts['total_facts']} total facts")
            print(f"     - {best_by_facts['success_rate']*100:.1f}% success rate")
            print(f"     - {best_by_facts['avg_time_ms']:.1f}ms average time")
        else:
            print("  ❌ All approaches failed - deeper algorithm fix needed")

def main():
    """Run all confidence fix tests"""
    tester = ConfidenceFixTester()
    
    try:
        tester.setup()
        
        # Run all tests
        all_results = []
        
        # Baseline first
        all_results.append(tester.test_baseline_current())
        
        # Then the fixes
        all_results.append(tester.test_fix_1_lower_threshold())
        all_results.append(tester.test_fix_2_bypass_basic_patterns())
        all_results.append(tester.test_fix_3_basic_fact_bypass())
        all_results.append(tester.test_fix_4_mixed_approach())
        
        # Print comparison
        tester.print_comparison_results(all_results)
        
    finally:
        tester.cleanup()

if __name__ == '__main__':
    main()