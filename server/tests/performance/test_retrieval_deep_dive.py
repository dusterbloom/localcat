#!/usr/bin/env python3.12
"""
Retrieval Performance Deep Dive Analysis
========================================

Comprehensive analysis of memory retrieval performance, bottlenecks, and optimization opportunities.
"""

import time
import json
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import cProfile
import pstats
import io
import sys
import os

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.retrieval.memory_retriever import MemoryRetriever, RetrievalResult
from components.retrieval.memory_retriever_optimized import MemoryRetrieverOptimized
from components.memory.memory_timing_tracer import MemoryTimingTracer, get_memory_tracer
from components.memory.memory_store import MemoryStore
from components.session.session_store import SessionStore
from components.memory.hotmemory_facade import HotMemoryFacade
from components.context.memory_config import MemoryConfig

@dataclass
class RetrievalBenchmark:
    """Single retrieval benchmark result"""
    query: str
    entities: List[str]
    expected_bullets: int
    actual_bullets: int
    duration_ms: float
    timing_breakdown: Dict[str, float]
    memory_stats: Dict[str, Any]

@dataclass
class RetrievalProfile:
    """Retrieval performance profile"""
    operation: str
    total_time_ms: float
    expansion_time_ms: float
    gathering_time_ms: float
    mmr_time_ms: float
    bullet_count: int
    candidate_count: int
    entity_count: int
    expansion_count: int

class RetrievalPerformanceAnalyzer:
    """Comprehensive retrieval performance analyzer"""

    def __init__(self):
        self.config = MemoryConfig()
        self.memory_store = MemoryStore()
        self.session_store = SessionStore()
        self.facade = HotMemoryFacade(self.memory_store, self.session_store)
        self.tracer = get_memory_tracer()

        # Test queries with varying complexity
        self.test_queries = [
            # Simple single-entity queries
            {
                "query": "Where does Michael work?",
                "entities": ["michael"],
                "expected_complexity": "low"
            },
            {
                "query": "Tell me about Sarah Chen",
                "entities": ["sarah chen"],
                "expected_complexity": "low"
            },

            # Multi-entity queries
            {
                "query": "Where do Michael and Sarah live?",
                "entities": ["michael", "sarah chen"],
                "expected_complexity": "medium"
            },
            {
                "query": "Tell me about Michael's job and Sarah's education",
                "entities": ["michael", "sarah chen"],
                "expected_complexity": "medium"
            },

            # Complex multi-hop queries
            {
                "query": "Where do people who work at Apple live?",
                "entities": ["apple"],
                "expected_complexity": "high"
            },
            {
                "query": "Tell me about the family members of people who work in tech",
                "entities": ["tech"],
                "expected_complexity": "high"
            },

            # Factual queries requiring retrieval
            {
                "query": "Where does Michael work and what city does he live in?",
                "entities": ["michael"],
                "expected_complexity": "medium"
            },
            {
                "query": "What do you know about Michael and Sarah's relationship?",
                "entities": ["michael", "sarah chen"],
                "expected_complexity": "high"
            }
        ]

        self.profiles = []
        self.benchmarks = []

    def setup_test_data(self):
        """Set up test memory data for benchmarking"""
        print("📝 Setting up test memory data...")

        # Clear existing data
        self.memory_store.clear()
        self.session_store.clear()

        # Create test session
        session_id = "retrieval_benchmark_deep_dive"

        # Add test data
        test_turns = [
            {
                "text": "I work at Google as a software engineer and my name is Sarah Chen",
                "expected_entities": ["I", "Google", "software engineer", "Sarah Chen"]
            },
            {
                "text": "Michael works at Apple as a designer and lives in San Francisco",
                "expected_entities": ["Michael", "Apple", "designer", "San Francisco"]
            },
            {
                "text": "I live with my husband Michael and our two kids in New York",
                "expected_entities": ["I", "husband Michael", "kids", "New York"]
            },
            {
                "text": "My parents live in Seattle and Michael's parents live in Boston",
                "expected_entities": ["parents", "Seattle", "Michael parents", "Boston"]
            },
            {
                "text": "Sarah studied computer science at Stanford and graduated in 2017",
                "expected_entities": ["Sarah", "computer science", "Stanford", "2017"]
            },
            {
                "text": "Michael studied graphic design at RISD and graduated in 2015",
                "expected_entities": ["Michael", "graphic design", "RISD", "2015"]
            }
        ]

        # Process test turns to build memory
        for i, turn in enumerate(test_turns):
            print(f"   Processing turn {i+1}/{len(test_turns)}...")

            self.facade.process_turn(
                session_id=session_id,
                text=turn["text"],
                speaker="user",
                turn_id=i
            )

        print(f"✅ Setup complete: {len(test_turns)} turns processed")

    def profile_retrieval_components(self):
        """Profile individual retrieval components with detailed timing"""
        print("\n🔍 Profiling retrieval components...")

        # Build entity index from memory store
        entity_index = self._build_entity_index()

        # Initialize retrievers
        original_retriever = MemoryRetriever(
            store=self.memory_store,
            entity_index=entity_index,
            config={
                'use_leann': False,  # Disable for clean profiling
                'retrieval_fusion': True,
                'leann_complexity': 16
            }
        )

        optimized_retriever = MemoryRetrieverOptimized(
            store=self.memory_store,
            entity_index=entity_index,
            config={
                'use_leann': False,
                'retrieval_fusion': True,
                'leann_complexity': 16
            }
        )

        retrievers = {
            'original': original_retriever,
            'optimized': optimized_retriever
        }

        results = {}

        for retriever_name, retriever in retrievers.items():
            print(f"\n📊 Profiling {retriever_name} retriever...")

            retriever_results = []

            for i, test_case in enumerate(self.test_queries):
                print(f"   Test {i+1}: {test_case['query'][:30]}...")

                # Profile with cProfile
                pr = cProfile.Profile()
                pr.enable()

                # Time the retrieval
                start_time = time.perf_counter()

                result = retriever.retrieve_context(
                    query=test_case['query'],
                    entities=test_case['entities'],
                    turn_id=i
                )

                end_time = time.perf_counter()

                pr.disable()

                # Get profiling stats
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                profile_stats = s.getvalue()

                # Create profile
                profile = RetrievalProfile(
                    operation=f"{retriever_name}_retrieval",
                    total_time_ms=(end_time - start_time) * 1000,
                    expansion_time_ms=result.retrieval_stats.get('expand_ms', 0),
                    gathering_time_ms=result.retrieval_stats.get('gather_ms', 0),
                    mmr_time_ms=result.retrieval_stats.get('mmr_ms', 0),
                    bullet_count=len(result.bullets),
                    candidate_count=result.retrieval_stats.get('candidates', 0),
                    entity_count=len(test_case['entities']),
                    expansion_count=len(result.expanded_entities)
                )

                retriever_results.append({
                    'profile': profile,
                    'test_case': test_case,
                    'profile_stats': profile_stats,
                    'result': result
                })

                print(f"      Time: {profile.total_time_ms:.1f}ms, "
                      f"Bullets: {profile.bullet_count}, "
                      f"Candidates: {profile.candidate_count}")

            results[retriever_name] = retriever_results

        return results

    def analyze_timing_bottlenecks(self, profiling_results):
        """Analyze timing bottlenecks in detail"""
        print("\n🔍 Analyzing timing bottlenecks...")

        bottleneck_analysis = {
            'original': {},
            'optimized': {}
        }

        for retriever_name, results in profiling_results.items():
            print(f"\n📊 {retriever_name.upper()} RETRIEVER BOTTLENECK ANALYSIS:")

            # Collect timing data
            expansion_times = []
            gathering_times = []
            mmr_times = []
            total_times = []

            for test_result in results:
                profile = test_result['profile']
                expansion_times.append(profile.expansion_time_ms)
                gathering_times.append(profile.gathering_time_ms)
                mmr_times.append(profile.mmr_time_ms)
                total_times.append(profile.total_time_ms)

            # Calculate statistics
            def calculate_stats(times, component_name):
                if not times:
                    return {"mean": 0, "median": 0, "p95": 0, "max": 0}

                return {
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "p95": statistics.quantiles(times, n=20)[18] if len(times) > 1 else times[0],
                    "max": max(times),
                    "min": min(times)
                }

            exp_stats = calculate_stats(expansion_times, "expansion")
            gath_stats = calculate_stats(gathering_times, "gathering")
            mmr_stats = calculate_stats(mmr_times, "mmr")
            total_stats = calculate_stats(total_times, "total")

            analysis = {
                "expansion": exp_stats,
                "gathering": gath_stats,
                "mmr": mmr_stats,
                "total": total_stats,
                "breakdown_percentages": {
                    "expansion": (exp_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0,
                    "gathering": (gath_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0,
                    "mmr": (mmr_stats["mean"] / total_stats["mean"] * 100) if total_stats["mean"] > 0 else 0
                }
            }

            bottleneck_analysis[retriever_name] = analysis

            # Print analysis
            print(f"   Total Time: {total_stats['mean']:.1f}ms (p95: {total_stats['p95']:.1f}ms)")
            print(f"   - Expansion: {exp_stats['mean']:.1f}ms ({analysis['breakdown_percentages']['expansion']:.1f}%)")
            print(f"   - Gathering: {gath_stats['mean']:.1f}ms ({analysis['breakdown_percentages']['gathering']:.1f}%)")
            print(f"   - MMR: {mmr_stats['mean']:.1f}ms ({analysis['breakdown_percentages']['mmr']:.1f}%)")

            # Identify bottlenecks
            components = [
                ("Expansion", exp_stats["mean"], analysis['breakdown_percentages']['expansion']),
                ("Gathering", gath_stats["mean"], analysis['breakdown_percentages']['gathering']),
                ("MMR", mmr_stats["mean"], analysis['breakdown_percentages']['mmr'])
            ]

            bottleneck = max(components, key=lambda x: x[1])
            print(f"   🚨 Main bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}ms, {bottleneck[2]:.1f}%)")

        return bottleneck_analysis

    def analyze_scaling_performance(self):
        """Analyze how retrieval performance scales with data size"""
        print("\n📈 Analyzing scaling performance...")

        # Test with increasing data sizes
        data_sizes = [10, 50, 100, 200, 500]  # Number of memory triples
        scaling_results = []

        for size in data_sizes:
            print(f"\n📊 Testing with {size} memory triples...")

            # Create test data with specified size
            self._create_scaled_test_data(size)
            entity_index = self._build_entity_index()

            # Test retriever performance
            retriever = MemoryRetriever(
                store=self.memory_store,
                entity_index=entity_index,
                config={'use_leann': False, 'retrieval_fusion': True}
            )

            test_times = []
            for test_case in self.test_queries[:3]:  # Use first 3 test cases
                start_time = time.perf_counter()

                result = retriever.retrieve_context(
                    query=test_case['query'],
                    entities=test_case['entities'],
                    turn_id=0
                )

                end_time = time.perf_counter()
                test_times.append((end_time - start_time) * 1000)

            avg_time = statistics.mean(test_times)
            scaling_results.append({
                'data_size': size,
                'avg_time_ms': avg_time,
                'throughput_queries_per_sec': 1000 / avg_time if avg_time > 0 else 0
            })

            print(f"   Average retrieval time: {avg_time:.1f}ms")
            print(f"   Throughput: {1000 / avg_time:.1f} queries/second")

        return scaling_results

    def analyze_memory_access_patterns(self):
        """Analyze memory access patterns and their impact"""
        print("\n🔍 Analyzing memory access patterns...")

        # Build entity index and analyze access patterns
        entity_index = self._build_entity_index()

        access_patterns = {
            'entity_frequencies': defaultdict(int),
            'relation_frequencies': defaultdict(int),
            'entity_connection_counts': defaultdict(int),
            'memory_footprint': 0
        }

        # Analyze entity index
        for entity, triples in entity_index.items():
            access_patterns['entity_frequencies'][entity] += len(triples)
            access_patterns['entity_connection_counts'][entity] = len(triples)

            for triple in triples:
                if isinstance(triple, (tuple, list)) and len(triple) >= 3:
                    s, r, d = triple[:3]
                    access_patterns['relation_frequencies'][r] += 1

        # Calculate memory footprint (rough estimate)
        access_patterns['memory_footprint'] = (
            sum(len(str(k)) + len(str(v)) for k, v in entity_index.items()) +
            sum(len(str(t)) for triples in entity_index.values() for t in triples)
        )

        # Print analysis
        print(f"   Memory footprint: ~{access_patterns['memory_footprint'] / 1024:.1f}KB")
        print(f"   Unique entities: {len(access_patterns['entity_frequencies'])}")
        print(f"   Unique relations: {len(access_patterns['relation_frequencies'])}")
        print(f"   Total triples: {sum(access_patterns['entity_frequencies'].values())}")

        # Most frequent entities and relations
        top_entities = sorted(access_patterns['entity_frequencies'].items(),
                            key=lambda x: x[1], reverse=True)[:5]
        top_relations = sorted(access_patterns['relation_frequencies'].items(),
                             key=lambda x: x[1], reverse=True)[:5]

        print(f"   Top entities: {[(e, c) for e, c in top_entities]}")
        print(f"   Top relations: {[(r, c) for r, c in top_relations]}")

        return access_patterns

    def identify_optimization_opportunities(self, profiling_results, bottleneck_analysis):
        """Identify specific optimization opportunities"""
        print("\n💡 Identifying optimization opportunities...")

        opportunities = []

        # Analyze bottlenecks
        for retriever_name, analysis in bottleneck_analysis.items():
            breakdown = analysis['breakdown_percentages']

            # Entity expansion optimizations
            if breakdown['expansion'] > 30:
                opportunities.append({
                    'area': 'Entity Expansion',
                    'issue': f'Slow expansion ({breakdown["expansion"]:.1f}% of time)',
                    'suggestions': [
                        'Implement early termination in multi-hop expansion',
                        'Cache expansion results for common entities',
                        'Limit expansion depth based on query complexity'
                    ],
                    'priority': 'high' if breakdown['expansion'] > 50 else 'medium'
                })

            # Candidate gathering optimizations
            if breakdown['gathering'] > 40:
                opportunities.append({
                    'area': 'Candidate Gathering',
                    'issue': f'Slow gathering ({breakdown["gathering"]:.1f}% of time)',
                    'suggestions': [
                        'Pre-filter entities by relevance before scoring',
                        'Implement batch scoring for similar entities',
                        'Use more efficient similarity calculation'
                    ],
                    'priority': 'high' if breakdown['gathering'] > 60 else 'medium'
                })

            # MMR optimizations
            if breakdown['mmr'] > 40:
                opportunities.append({
                    'area': 'MMR Selection',
                    'issue': f'Slow MMR selection ({breakdown["mmr"]:.1f}% of time)',
                    'suggestions': [
                        'Limit candidate pool size before MMR',
                        'Optimize similarity calculations',
                        'Implement early termination in MMR loop'
                    ],
                    'priority': 'high' if breakdown['mmr'] > 60 else 'medium'
                })

        # Memory usage optimizations
        opportunities.append({
            'area': 'Memory Usage',
            'issue': 'High memory footprint for entity index',
            'suggestions': [
                'Use more memory-efficient data structures',
                'Implement LRU caching for frequently accessed entities',
                'Consider database-backed indexing for large datasets'
            ],
            'priority': 'medium'
        })

        # Algorithmic optimizations
        opportunities.append({
            'area': 'Algorithm Selection',
            'issue': 'Current algorithms may not be optimal for all query types',
            'suggestions': [
                'Use different strategies based on query complexity',
                'Implement fallback to simpler algorithms for time-sensitive queries',
                'Add query-specific optimization paths'
            ],
            'priority': 'medium'
        })

        # Print opportunities
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{i}. {opp['area']} ({opp['priority'].upper()} priority)")
            print(f"   Issue: {opp['issue']}")
            print(f"   Suggestions:")
            for suggestion in opp['suggestions']:
                print(f"     - {suggestion}")

        return opportunities

    def generate_comprehensive_report(self, profiling_results, bottleneck_analysis,
                                    scaling_results, access_patterns, optimization_opportunities):
        """Generate comprehensive performance report"""
        print("\n📋 GENERATING COMPREHENSIVE RETRIEVAL PERFORMANCE REPORT")
        print("=" * 80)

        report = {
            'timestamp': time.time(),
            'test_configuration': {
                'queries_tested': len(self.test_queries),
                'retrievers_tested': list(profiling_results.keys()),
                'memory_data_size': scaling_results[-1]['data_size'] if scaling_results else 0
            },
            'performance_summary': bottleneck_analysis,
            'scaling_analysis': scaling_results,
            'access_patterns': access_patterns,
            'optimization_opportunities': optimization_opportunities,
            'recommendations': []
        }

        # Executive Summary
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 40)

        original_perf = bottleneck_analysis['original']['total']
        optimized_perf = bottleneck_analysis['optimized']['total']

        print(f"Original retriever avg time: {original_perf['mean']:.1f}ms (p95: {original_perf['p95']:.1f}ms)")
        print(f"Optimized retriever avg time: {optimized_perf['mean']:.1f}ms (p95: {optimized_perf['p95']:.1f}ms)")

        if original_perf['mean'] > 0:
            improvement = (original_perf['mean'] - optimized_perf['mean']) / original_perf['mean'] * 100
            print(f"Performance improvement: {improvement:.1f}%")

        # Scaling Analysis
        print(f"\n📈 SCALING ANALYSIS")
        print("-" * 40)
        if scaling_results:
            smallest = scaling_results[0]
            largest = scaling_results[-1]
            scaling_factor = largest['avg_time_ms'] / smallest['avg_time_ms'] if smallest['avg_time_ms'] > 0 else 0
            data_growth_factor = largest['data_size'] / smallest['data_size']

            print(f"Data growth: {smallest['data_size']} → {largest['data_size']} triples ({data_growth_factor:.1f}x)")
            print(f"Time growth: {smallest['avg_time_ms']:.1f}ms → {largest['avg_time_ms']:.1f}ms ({scaling_factor:.1f}x)")

            if scaling_factor > data_growth_factor * 1.5:
                print("⚠️  Retrieval time growing faster than data size - needs optimization")
            else:
                print("✅ Good scaling characteristics")

        # Top Recommendations
        print(f"\n🎯 TOP RECOMMENDATIONS")
        print("-" * 40)

        high_priority = [opp for opp in optimization_opportunities if opp['priority'] == 'high']
        for i, opp in enumerate(high_priority[:3], 1):
            print(f"{i}. {opp['area']}: {opp['suggestions'][0]}")

        # Save detailed report
        report_file = f"retrieval_performance_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📁 Detailed report saved to: {report_file}")

        return report

    def _build_entity_index(self) -> Dict[str, set]:
        """Build entity index from memory store"""
        entity_index = defaultdict(set)

        # This is a simplified version - in practice, you'd extract from memory store
        # For now, we'll use the test data we've been building
        if hasattr(self.memory_store, 'get_all_triples'):
            triples = self.memory_store.get_all_triples()
            for triple in triples:
                if isinstance(triple, (tuple, list)) and len(triple) >= 3:
                    s, r, d = triple[:3]
                    entity_index[s].add(triple)
                    entity_index[d].add(triple)

        return dict(entity_index)

    def _create_scaled_test_data(self, num_triples: int):
        """Create test data with specified number of triples"""
        self.memory_store.clear()

        entities = [f"entity_{i}" for i in range(num_triples // 10)]
        relations = ["works_at", "lives_in", "studied_at", "has", "knows", "married_to"]

        triples_created = 0
        for i in range(num_triples):
            if i >= len(entities):
                break

            s = entities[i]
            r = relations[i % len(relations)]
            d = entities[(i + 1) % len(entities)]

            # Add to memory store (simplified)
            self.memory_store.add_triple(s, r, d, confidence=0.7)
            triples_created += 1

        print(f"   Created {triples_created} triples with {len(entities)} entities")

    def run_comprehensive_analysis(self):
        """Run complete retrieval performance analysis"""
        print("🚀 Starting comprehensive retrieval performance analysis...")

        # Setup test data
        self.setup_test_data()

        # Profile retrieval components
        profiling_results = self.profile_retrieval_components()

        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_timing_bottlenecks(profiling_results)

        # Analyze scaling
        scaling_results = self.analyze_scaling_performance()

        # Analyze access patterns
        access_patterns = self.analyze_memory_access_patterns()

        # Identify optimization opportunities
        optimization_opportunities = self.identify_optimization_opportunities(
            profiling_results, bottleneck_analysis
        )

        # Generate comprehensive report
        report = self.generate_comprehensive_report(
            profiling_results, bottleneck_analysis, scaling_results,
            access_patterns, optimization_opportunities
        )

        print("\n✅ Comprehensive retrieval performance analysis complete!")
        return report


if __name__ == "__main__":
    analyzer = RetrievalPerformanceAnalyzer()
    report = analyzer.run_comprehensive_analysis()