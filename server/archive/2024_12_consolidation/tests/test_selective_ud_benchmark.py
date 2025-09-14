#!/usr/bin/env python3
"""
Selective UD Patterns Benchmark Test
Performance comparison: Full 27 patterns vs Selective approach

Expected results:
- Essential (8 patterns): ~80ms, 10% accuracy drop
- Essential + Connectivity (15 patterns): ~120ms, 5% accuracy drop
- Full system (27 patterns): ~250ms, baseline accuracy

Target: <300ms for realtime graph intelligence
"""

import time
import sys
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics

# Add server to path
sys.path.insert(0, '.')

import spacy
from services.selective_ud_patterns import SelectiveUDPatterns, PatternTier, extract_priority_patterns
from services.ud_utils import UDPatternMatcher

@dataclass
class BenchmarkResult:
    approach: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    relations_count: int
    patterns_used: int
    accuracy_vs_full: float


class SelectiveUDBenchmark:
    """Comprehensive benchmark of selective UD pattern system"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.selective_patterns = SelectiveUDPatterns()
        self.full_patterns = UDPatternMatcher()

        # Test sentences with varying complexity
        self.test_sentences = [
            # Simple sentences (should use Essential tier only)
            "John runs fast.",
            "The cat sleeps.",
            "Apple Inc. exists.",
            "She works hard.",

            # Medium complexity (should use Essential + Connectivity)
            "Steve Jobs founded Apple Inc. in Cupertino, California.",
            "The meeting is scheduled for 3 PM tomorrow in conference room A.",
            "Dr. Smith teaches artificial intelligence at Stanford University.",
            "Microsoft acquired GitHub for billions of dollars.",

            # Complex sentences (might need Optional tier)
            "Elon Musk founded SpaceX and currently serves as CEO while Tesla produces electric vehicles.",
            "After graduating from MIT, Sarah joined Tesla where she led the autopilot team before founding her own startup.",
            "The company announced that its revenue increased by 25% last quarter, which exceeded analysts' expectations significantly.",
            "When artificial intelligence systems become more sophisticated, they will likely transform industries in ways we haven't fully anticipated yet."
        ]

    def benchmark_approach(self, approach_name: str, extraction_func, sentences: List[str]) -> BenchmarkResult:
        """Benchmark a specific extraction approach"""
        timings = []
        relation_counts = []

        print(f"\n🔄 Benchmarking {approach_name}...")

        for i, sentence in enumerate(sentences, 1):
            doc = self.nlp(sentence)

            start_time = time.perf_counter()
            result = extraction_func(doc)
            execution_time = (time.perf_counter() - start_time) * 1000

            timings.append(execution_time)

            # Count relations
            if isinstance(result, dict):
                relations_count = len(result.get('relations', []))
                patterns_used = result.get('patterns_count', 0)
            else:
                relations_count = len(result) if result else 0
                patterns_used = 27  # Assume full system

            relation_counts.append(relations_count)

            print(f"  Test {i:2d}: {execution_time:6.1f}ms, {relations_count:2d} relations - {sentence[:50]}{'...' if len(sentence) > 50 else ''}")

        avg_relations = statistics.mean(relation_counts) if relation_counts else 0

        return BenchmarkResult(
            approach=approach_name,
            avg_time_ms=statistics.mean(timings),
            min_time_ms=min(timings),
            max_time_ms=max(timings),
            relations_count=int(avg_relations),
            patterns_used=patterns_used if 'patterns_used' in locals() else 0,
            accuracy_vs_full=1.0  # Will be calculated later
        )

    def run_full_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark of all approaches"""

        print("🚀 SELECTIVE UD PATTERNS BENCHMARK")
        print("=" * 80)
        print(f"Testing {len(self.test_sentences)} sentences of varying complexity")
        print("Target: <300ms for realtime graph intelligence")

        results = {}

        # 1. Essential patterns only (8 patterns, ~80ms target)
        results['essential'] = self.benchmark_approach(
            "Essential (8 patterns)",
            lambda doc: self.selective_patterns.extract_selective_patterns(doc, PatternTier.ESSENTIAL),
            self.test_sentences
        )

        # 2. Essential + Connectivity (15 patterns, ~120ms target)
        results['connectivity'] = self.benchmark_approach(
            "Essential + Connectivity (15 patterns)",
            lambda doc: self.selective_patterns.extract_selective_patterns(doc, PatternTier.CONNECTIVITY),
            self.test_sentences
        )

        # 3. Full selective system (adaptive tier selection)
        results['adaptive'] = self.benchmark_approach(
            "Adaptive Tier Selection",
            lambda doc: self.selective_patterns.extract_selective_patterns(doc),
            self.test_sentences
        )

        # 4. Priority patterns with time budget (120ms max)
        results['priority'] = self.benchmark_approach(
            "Priority Patterns (120ms budget)",
            lambda doc: extract_priority_patterns(doc, max_execution_time_ms=120.0),
            self.test_sentences
        )

        # TODO: Add full 27-pattern system comparison when UDPatternMatcher is working
        # results['full_system'] = self.benchmark_approach(
        #     "Full System (27 patterns)",
        #     lambda doc: self._extract_full_patterns(doc),
        #     self.test_sentences
        # )

        return results

    def analyze_results(self, results: Dict[str, BenchmarkResult]) -> None:
        """Analyze and display benchmark results"""

        print("\n" + "=" * 80)
        print("📊 BENCHMARK RESULTS")
        print("=" * 80)

        # Performance comparison table
        print(f"{'Approach':<30} {'Avg Time':<12} {'Min/Max':<15} {'Relations':<10} {'Patterns':<9}")
        print("-" * 80)

        for name, result in results.items():
            print(f"{result.approach:<30} {result.avg_time_ms:>8.1f}ms   "
                  f"{result.min_time_ms:>5.1f}/{result.max_time_ms:<5.1f}ms   "
                  f"{result.relations_count:>6}     {result.patterns_used:>7}")

        print("\n" + "=" * 80)
        print("⚡ PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Find fastest approach
        fastest = min(results.values(), key=lambda x: x.avg_time_ms)
        print(f"🏆 Fastest: {fastest.approach} at {fastest.avg_time_ms:.1f}ms")

        # Realtime assessment (target <300ms)
        realtime_approaches = [r for r in results.values() if r.avg_time_ms < 300]
        print(f"✅ Realtime capable (<300ms): {len(realtime_approaches)}/{len(results)} approaches")

        for approach in realtime_approaches:
            margin = 300 - approach.avg_time_ms
            print(f"   • {approach.approach}: {approach.avg_time_ms:.1f}ms ({margin:.1f}ms margin)")

        # Quality vs Performance trade-off
        print(f"\n📈 QUALITY vs PERFORMANCE TRADE-OFF:")
        baseline_relations = max(r.relations_count for r in results.values())

        for name, result in results.items():
            quality_ratio = result.relations_count / baseline_relations if baseline_relations > 0 else 1.0
            quality_drop = (1 - quality_ratio) * 100

            print(f"   • {result.approach}:")
            print(f"     Time: {result.avg_time_ms:.1f}ms, Relations: {result.relations_count}, Quality: {quality_ratio:.1%} (-{quality_drop:.1f}%)")

        # Recommendation
        print(f"\n🎯 RECOMMENDATIONS:")

        # Find best balance of speed and quality
        balanced_approaches = [r for r in results.values()
                              if r.avg_time_ms < 200 and r.relations_count >= baseline_relations * 0.8]

        if balanced_approaches:
            best_balanced = min(balanced_approaches, key=lambda x: x.avg_time_ms)
            print(f"✨ Best balanced approach: {best_balanced.approach}")
            print(f"   Performance: {best_balanced.avg_time_ms:.1f}ms")
            print(f"   Relations: {best_balanced.relations_count} ({best_balanced.relations_count/baseline_relations:.1%} of max)")

        # Performance target assessment
        target_approaches = [r for r in results.values() if r.avg_time_ms <= 120]
        if target_approaches:
            print(f"🚀 Target performance (<120ms): {len(target_approaches)} approaches qualify")
        else:
            print(f"⚠️  No approaches meet <120ms target. Best: {fastest.avg_time_ms:.1f}ms")

    def complexity_analysis(self) -> None:
        """Analyze how sentence complexity affects pattern selection"""

        print("\n" + "=" * 80)
        print("🧠 SENTENCE COMPLEXITY ANALYSIS")
        print("=" * 80)

        complexity_stats = {"simple": [], "normal": [], "complex": []}

        for sentence in self.test_sentences:
            doc = self.nlp(sentence)
            complexity = self.selective_patterns.analyze_sentence_complexity(doc)

            # Test adaptive extraction
            result = self.selective_patterns.extract_selective_patterns(doc)

            complexity_stats[complexity].append({
                'sentence': sentence[:60] + "..." if len(sentence) > 60 else sentence,
                'tier_used': result['tier_used'],
                'time_ms': result['execution_time_ms'],
                'relations': len(result['relations']),
                'patterns': result['patterns_count']
            })

        for complexity, stats in complexity_stats.items():
            if not stats:
                continue

            avg_time = statistics.mean([s['time_ms'] for s in stats])
            avg_relations = statistics.mean([s['relations'] for s in stats])

            print(f"\n{complexity.upper()} sentences ({len(stats)} samples):")
            print(f"  Average time: {avg_time:.1f}ms")
            print(f"  Average relations: {avg_relations:.1f}")

            for stat in stats:
                print(f"    {stat['tier_used']:12s} {stat['time_ms']:6.1f}ms {stat['relations']:2d} rels - {stat['sentence']}")


async def run_benchmark():
    """Main benchmark execution"""
    benchmark = SelectiveUDBenchmark()

    # Run all benchmarks
    results = benchmark.run_full_benchmark()

    # Analyze results
    benchmark.analyze_results(results)

    # Complexity analysis
    benchmark.complexity_analysis()

    print("\n" + "=" * 80)
    print("✅ SELECTIVE UD PATTERNS BENCHMARK COMPLETE")
    print("=" * 80)

    # Performance gain summary
    patterns = SelectiveUDPatterns()
    perf_gain = patterns.estimate_performance_gain()

    print("\n📊 ESTIMATED PERFORMANCE GAINS:")
    print(f"  Full system baseline: {perf_gain['full_system_ms']}ms")
    print(f"  Essential only: {perf_gain['essential_only_ms']:.1f}ms ({perf_gain['speedup_essential']} faster)")
    print(f"  Essential + Connectivity: {perf_gain['essential_plus_connectivity_ms']:.1f}ms ({perf_gain['speedup_connectivity']} faster)")

    print(f"\n🎯 REALTIME READINESS:")
    if perf_gain['essential_plus_connectivity_ms'] < 120:
        print(f"✅ Ready for <300ms realtime extraction!")
    else:
        print(f"⚠️  Needs more optimization for <300ms target")


if __name__ == "__main__":
    print("🚀 Starting Selective UD Patterns Benchmark...")
    asyncio.run(run_benchmark())