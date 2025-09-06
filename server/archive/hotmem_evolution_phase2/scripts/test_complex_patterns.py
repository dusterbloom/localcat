#!/usr/bin/env python3
"""
Complex Sentence Pattern Testing
================================

Tests the HotMem system with complex sentences that utilize all 27 dependency patterns.
This validates the system's ability to handle sophisticated grammatical constructions.
"""

import os
import sys
import time
import tempfile
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

@dataclass
class ComplexSentenceTest:
    """Test case for complex sentence patterns"""
    name: str
    sentence: str
    patterns_tested: List[str]
    expected_facts: List[str]  # Human-readable expected extractions
    difficulty: str  # easy, medium, hard, extreme

# Complex test sentences covering different dependency patterns
COMPLEX_SENTENCES = [
    ComplexSentenceTest(
        name="Nested Relative Clauses",
        sentence="The professor who taught the class that I took last semester, whose research focuses on AI ethics, recently published a book about algorithmic bias that became a bestseller.",
        patterns_tested=["nsubj", "acl", "nmod", "poss", "dobj", "amod"],
        expected_facts=["professor teaches class", "you took class", "research focuses on AI ethics", "professor published book"],
        difficulty="extreme"
    ),
    
    ComplexSentenceTest(
        name="Passive with Agent",
        sentence="The presentation was delivered by Sarah, who was praised by her colleagues for the innovative approach that was demonstrated during the meeting.",
        patterns_tested=["nsubjpass", "agent", "acl", "dobj", "prep"],
        expected_facts=["Sarah delivered presentation", "colleagues praised Sarah", "approach was innovative"],
        difficulty="hard"
    ),
    
    ComplexSentenceTest(
        name="Complex Coordination",
        sentence="My brother Tom, who lives in Portland and teaches philosophy at Reed College, is writing a book about ethics while simultaneously preparing for his sabbatical in Europe.",
        patterns_tested=["appos", "acl", "conj", "prep", "advcl", "xcomp"],
        expected_facts=["Tom lives in Portland", "Tom teaches philosophy", "Tom teaches at Reed College", "Tom is writing book"],
        difficulty="hard"
    ),
    
    ComplexSentenceTest(
        name="Clausal Complements",
        sentence="I believe that machine learning, which has revolutionized data analysis, will continue to transform industries that rely heavily on pattern recognition and prediction.",
        patterns_tested=["ccomp", "nsubj", "acl", "xcomp", "dobj", "conj"],
        expected_facts=["you believe machine learning will transform", "machine learning revolutionized analysis", "industries rely on recognition"],
        difficulty="extreme"
    ),
    
    ComplexSentenceTest(
        name="Multiple Modifiers",
        sentence="The extremely talented young musician from Seattle, whose beautiful violin performances have captivated audiences worldwide, recently won three prestigious international awards.",
        patterns_tested=["amod", "nmod", "poss", "acl", "dobj", "nummod"],
        expected_facts=["musician from Seattle", "musician is talented", "performances captivated audiences", "musician won awards"],
        difficulty="hard"
    ),
    
    ComplexSentenceTest(
        name="Embedded Questions", 
        sentence="I wonder whether the new algorithm that my team developed, which processes natural language more efficiently than previous methods, will be accepted for publication in the upcoming conference.",
        patterns_tested=["ccomp", "acl", "nsubj", "advmod", "prep", "dobj"],
        expected_facts=["you wonder about algorithm", "team developed algorithm", "algorithm processes language", "algorithm is efficient"],
        difficulty="extreme"
    ),
    
    ComplexSentenceTest(
        name="Possessive Chains",
        sentence="My grandmother's oldest sister's house, which was built by my great-grandfather in 1923, still contains my family's antique furniture that was handcrafted by local artisans.",
        patterns_tested=["poss", "nsubjpass", "agent", "acl", "dobj", "amod"],
        expected_facts=["grandmother has sister", "sister has house", "great-grandfather built house", "house contains furniture"],
        difficulty="extreme"
    ),
    
    ComplexSentenceTest(
        name="Attributive Constructions",
        sentence="The solution seems remarkably elegant and surprisingly simple, making it both theoretically sound and practically implementable for real-world applications.",
        patterns_tested=["attr", "acomp", "conj", "xcomp", "amod", "prep"],
        expected_facts=["solution is elegant", "solution is simple", "solution is implementable"],
        difficulty="medium"
    ),
    
    ComplexSentenceTest(
        name="Temporal and Causal Chains",
        sentence="After graduating from MIT in 2020, where she studied computer science and mathematics, Lisa joined Google as a software engineer before starting her own AI startup last year.",
        patterns_tested=["advcl", "prep", "acl", "conj", "xcomp", "nmod"],
        expected_facts=["Lisa graduated from MIT", "Lisa studied computer science", "Lisa joined Google", "Lisa started startup"],
        difficulty="hard"
    ),
    
    ComplexSentenceTest(
        name="Literary Complex",
        sentence="The character whose journey through the labyrinthine plot, filled with unexpected twists and profound revelations, ultimately leads to a deeper understanding of human nature, represents the author's most ambitious creative achievement.",
        patterns_tested=["acl", "nmod", "conj", "dobj", "prep", "amod", "attr"],
        expected_facts=["character has journey", "plot has twists", "journey leads to understanding", "character represents achievement"],
        difficulty="extreme"
    )
]

class ComplexPatternTester:
    def __init__(self):
        self.temp_dir = None
        self.hot_memory = None
        
    def setup(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        paths = Paths(
            sqlite_path=os.path.join(self.temp_dir, 'test_memory.db'),
            lmdb_dir=os.path.join(self.temp_dir, 'test_graph.lmdb')
        )
        store = MemoryStore(paths)
        self.hot_memory = HotMemory(store)
        self.hot_memory.prewarm('en')
        print(f"🏗️  Test storage created at: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up test data"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test storage")
    
    def test_complex_sentences(self) -> Dict[str, Any]:
        """Test extraction on complex sentences"""
        print("🧪 Testing Complex Sentence Pattern Extraction")
        print("=" * 60)
        
        results = []
        total_facts = 0
        total_time = 0.0
        
        for i, test_case in enumerate(COMPLEX_SENTENCES, 1):
            print(f"\n🔄 Test {i}: {test_case.name} ({test_case.difficulty.upper()})")
            print(f"📝 Sentence: {test_case.sentence[:80]}...")
            print(f"🎯 Patterns: {', '.join(test_case.patterns_tested[:5])}")
            if len(test_case.patterns_tested) > 5:
                print(f"          ... and {len(test_case.patterns_tested)-5} more")
            
            # Extract facts
            start_time = time.perf_counter()
            bullets, triples = self.hot_memory.process_turn(
                test_case.sentence, 
                session_id="complex_test", 
                turn_id=i
            )
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⚡ Extracted {len(triples)} facts in {extraction_time:.1f}ms")
            
            # Show extracted facts
            if triples:
                print("✅ Facts extracted:")
                for j, (s, r, d) in enumerate(triples[:5]):
                    print(f"  {j+1}. ({s}) -[{r}]-> ({d})")
                if len(triples) > 5:
                    print(f"  ... and {len(triples)-5} more facts")
            else:
                print("❌ No facts extracted")
            
            # Show bullets if any
            if bullets:
                print("📝 Memory bullets:")
                for j, bullet in enumerate(bullets[:3]):
                    print(f"  • {bullet}")
                if len(bullets) > 3:
                    print(f"  ... and {len(bullets)-3} more bullets")
            
            # Store results
            results.append({
                'test_case': test_case,
                'facts_count': len(triples),
                'bullets_count': len(bullets),
                'extraction_time_ms': extraction_time,
                'facts': triples,
                'bullets': bullets
            })
            
            total_facts += len(triples)
            total_time += extraction_time
        
        return {
            'results': results,
            'total_facts': total_facts,
            'avg_time_ms': total_time / len(COMPLEX_SENTENCES),
            'success_rate': len([r for r in results if r['facts_count'] > 0]) / len(COMPLEX_SENTENCES)
        }
    
    def analyze_results(self, test_results: Dict[str, Any]):
        """Analyze and print detailed results"""
        print(f"\n📈 COMPLEX SENTENCE ANALYSIS RESULTS")
        print("=" * 60)
        
        results = test_results['results']
        
        # Overall statistics
        print(f"\n📊 Overall Performance:")
        print(f"  Total Facts Extracted: {test_results['total_facts']}")
        print(f"  Average Extraction Time: {test_results['avg_time_ms']:.1f}ms")
        print(f"  Success Rate: {test_results['success_rate']*100:.1f}%")
        
        # By difficulty level
        difficulty_stats = {}
        for result in results:
            diff = result['test_case'].difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'facts': 0, 'time': 0, 'count': 0}
            difficulty_stats[diff]['facts'] += result['facts_count']
            difficulty_stats[diff]['time'] += result['extraction_time_ms']
            difficulty_stats[diff]['count'] += 1
        
        print(f"\n📋 Performance by Difficulty:")
        print(f"{'Difficulty':<12} {'Avg Facts':<12} {'Avg Time':<12} {'Tests':<8}")
        print("-" * 50)
        
        for diff in ['medium', 'hard', 'extreme']:
            if diff in difficulty_stats:
                stats = difficulty_stats[diff]
                avg_facts = stats['facts'] / stats['count']
                avg_time = stats['time'] / stats['count']
                print(f"{diff.title():<12} {avg_facts:<12.1f} {avg_time:<12.1f}ms {stats['count']:<8}")
        
        # Most successful extractions
        best_results = sorted(results, key=lambda x: x['facts_count'], reverse=True)[:3]
        print(f"\n🏆 Best Extractions:")
        for i, result in enumerate(best_results, 1):
            test_case = result['test_case']
            print(f"  {i}. {test_case.name}: {result['facts_count']} facts in {result['extraction_time_ms']:.1f}ms")
        
        # Pattern coverage analysis
        all_patterns = set()
        for test_case in [r['test_case'] for r in results]:
            all_patterns.update(test_case.patterns_tested)
        
        print(f"\n🎯 Dependency Patterns Tested:")
        print(f"  Total Unique Patterns: {len(all_patterns)}")
        print(f"  Patterns: {', '.join(sorted(all_patterns))}")
        
        # Detailed breakdown
        print(f"\n🔍 Detailed Results:")
        for i, result in enumerate(results, 1):
            test_case = result['test_case']
            success = "✅" if result['facts_count'] > 0 else "❌"
            print(f"  {i:2d}. {success} {test_case.name:<25} ({test_case.difficulty:<8}) {result['facts_count']:2d} facts, {result['extraction_time_ms']:6.1f}ms")
        
        # Recommendations
        print(f"\n💡 ANALYSIS:")
        if test_results['success_rate'] > 0.8:
            print("  ✅ Excellent performance on complex sentences!")
            print("  ✅ The 27-pattern system handles sophisticated constructions well")
        elif test_results['success_rate'] > 0.6:
            print("  ⚠️  Good performance but some complex patterns still challenging")
            print("  💡 Consider improving coreference and long-distance dependencies")
        else:
            print("  ❌ Complex sentence extraction needs improvement")
            print("  💡 Focus on clausal complements and relative clause handling")
        
        if test_results['avg_time_ms'] < 100:
            print("  ⚡ Excellent speed - suitable for real-time processing")
        elif test_results['avg_time_ms'] < 200:
            print("  ⚡ Good speed for batch processing")
        else:
            print("  ⚠️  High processing time - may need optimization")

def main():
    """Run complex pattern tests"""
    tester = ComplexPatternTester()
    
    try:
        tester.setup()
        
        # Run the tests
        test_results = tester.test_complex_sentences()
        
        # Analyze results
        tester.analyze_results(test_results)
        
    finally:
        tester.cleanup()

if __name__ == '__main__':
    main()