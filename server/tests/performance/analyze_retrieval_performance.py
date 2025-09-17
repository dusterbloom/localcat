#!/usr/bin/env python3.12
"""
Focused Retrieval Performance Analysis
=======================================

Deep dive analysis of retrieval performance, timing, and bottlenecks.
"""

import time
import json
import statistics
from typing import List, Dict, Any, Optional, Set, Tuple
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
from components.memory.memory_timing_tracer import MemoryTimingTracer

@dataclass
class RetrievalTimingProfile:
    """Detailed timing profile for retrieval operation"""
    operation_name: str
    total_time_ms: float
    expansion_time_ms: float
    gathering_time_ms: float
    mmr_time_ms: float
    query_length: int
    entity_count: int
    expanded_entity_count: int
    candidate_count: int
    bullet_count: int
    complexity_level: str

@dataclass
class RetrievalBenchmark:
    """Retrieval benchmark result"""
    query: str
    entities: List[str]
    retriever_type: str
    timing_profile: RetrievalTimingProfile
    bullets: List[str]
    relevant_triples: List[Tuple[str, str, str]]

class RetrievalPerformanceAnalyzer:
    """Focused retrieval performance analyzer"""

    def __init__(self):
        # Test queries with varying complexity
        self.test_queries = [
            # Simple queries
            {
                "query": "Where does Michael work?",
                "entities": ["michael"],
                "complexity": "low"
            },
            {
                "query": "Tell me about Sarah",
                "entities": ["sarah"],
                "complexity": "low"
            },

            # Medium complexity queries
            {
                "query": "Where do Michael and Sarah live?",
                "entities": ["michael", "sarah"],
                "complexity": "medium"
            },
            {
                "query": "What do you know about Michael's job and where he lives?",
                "entities": ["michael"],
                "complexity": "medium"
            },

            # Complex queries
            {
                "query": "Tell me about people who work at Apple and where they live",
                "entities": ["apple"],
                "complexity": "high"
            },
            {
                "query": "What do you know about Michael's education and career history?",
                "entities": ["michael"],
                "complexity": "high"
            },
            {
                "query": "Where do people who studied computer science work?",
                "entities": ["computer science"],
                "complexity": "high"
            }
        ]

        # Create sample entity index with test data
        self.entity_index = self._create_test_entity_index()
        self.mock_store = MockMemoryStore()

    def _create_test_entity_index(self) -> Dict[str, Set]:
        """Create test entity index with sample data"""
        entity_index = defaultdict(set)

        # Sample knowledge graph
        test_triples = [
            # Michael's information
            ("michael", "works_at", "apple"),
            ("michael", "lives_in", "san francisco"),
            ("michael", "studied_at", "risd"),
            ("michael", "graduated_in", "2015"),
            ("michael", "has_profession", "designer"),
            ("michael", "also_known_as", "mike"),
            ("michael", "married_to", "sarah"),

            # Sarah's information
            ("sarah", "works_at", "google"),
            ("sarah", "lives_in", "new york"),
            ("sarah", "studied_at", "stanford"),
            ("sarah", "graduated_in", "2017"),
            ("sarah", "has_profession", "software engineer"),
            ("sarah", "married_to", "michael"),

            # Company information
            ("apple", "located_in", "cupertino"),
            ("apple", "industry", "technology"),
            ("google", "located_in", "mountain view"),
            ("google", "industry", "technology"),

            # Educational institutions
            ("stanford", "located_in", "palo alto"),
            ("stanford", "type", "university"),
            ("risd", "located_in", "rhode island"),
            ("risd", "type", "design school"),

            # Location information
            ("san francisco", "located_in", "california"),
            ("new york", "located_in", "new york state"),
            ("cupertino", "located_in", "california"),
            ("mountain view", "located_in", "california"),
            ("palo alto", "located_in", "california"),

            # Family relationships
            ("michael parents", "live_in", "seattle"),
            ("sarah parents", "live_in", "boston"),
            ("parents", "have_children", "michael"),
            ("parents", "have_children", "sarah")
        ]

        # Add metadata for scoring
        edge_metadata = {}
        current_time = int(time.time() * 1000)

        for i, (s, r, d) in enumerate(test_triples):
            # Add to entity index
            entity_index[s].add((s, r, d))
            entity_index[d].add((s, r, d))

            # Add metadata with realistic timestamps
            edge_metadata[(s, r, d)] = {
                'ts': current_time - (i * 86400000),  # Spread over days
                'weight': 0.7 + (i % 5) * 0.06,  # Varying confidence
                'source': 'test'
            }

        # Add some aliases
        entity_index["mike"].add(("mike", "also_known_as", "michael"))
        entity_index["michael"].add(("michael", "also_known_as", "mike"))

        return dict(entity_index), edge_metadata

    def profile_retrieval_performance(self):
        """Profile both original and optimized retrievers"""
        print("🔍 Profiling retrieval performance...")

        entity_index, edge_metadata = self.entity_index

        # Initialize retrievers
        original_retriever = MemoryRetriever(
            store=self.mock_store,
            entity_index=entity_index,
            config={
                'use_leann': False,
                'retrieval_fusion': False,  # Disable for clean profiling
                'leann_complexity': 16
            }
        )
        original_retriever.edge_meta = edge_metadata

        optimized_retriever = MemoryRetrieverOptimized(
            store=self.mock_store,
            entity_index=entity_index,
            config={
                'use_leann': False,
                'retrieval_fusion': False,
                'leann_complexity': 16
            }
        )
        optimized_retriever.edge_meta = edge_metadata

        results = {
            'original': [],
            'optimized': []
        }

        # Test each query
        for i, test_case in enumerate(self.test_queries):
            print(f"\n📊 Test {i+1}: {test_case['query']} (complexity: {test_case['complexity']})")

            # Test original retriever
            print("   Testing original retriever...")
            original_result = self._profile_single_retrieval(
                original_retriever, test_case, 'original'
            )
            results['original'].append(original_result)

            # Test optimized retriever
            print("   Testing optimized retriever...")
            optimized_result = self._profile_single_retrieval(
                optimized_retriever, test_case, 'optimized'
            )
            results['optimized'].append(optimized_result)

            # Compare results
            improvement = 0
            if original_result.timing_profile.total_time_ms > 0:
                improvement = (original_result.timing_profile.total_time_ms - optimized_result.timing_profile.total_time_ms) / original_result.timing_profile.total_time_ms * 100

            print(f"   Comparison: Original {original_result.timing_profile.total_time_ms:.1f}ms → "
                  f"Optimized {optimized_result.timing_profile.total_time_ms:.1f}ms "
                  f"({improvement:+.1f}%)")

        return results

    def _profile_single_retrieval(self, retriever, test_case, retriever_type):
        """Profile a single retrieval operation with detailed timing"""
        # Use cProfile for detailed profiling
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.perf_counter()

        # Perform retrieval
        result = retriever.retrieve_context(
            query=test_case['query'],
            entities=test_case['entities'],
            turn_id=0
        )

        end_time = time.perf_counter()

        pr.disable()

        # Get profile stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        profile_stats = s.getvalue()

        # Create timing profile
        timing_profile = RetrievalTimingProfile(
            operation_name=f"{retriever_type}_retrieval",
            total_time_ms=(end_time - start_time) * 1000,
            expansion_time_ms=result.retrieval_stats.get('expand_ms', 0),
            gathering_time_ms=result.retrieval_stats.get('gather_ms', 0),
            mmr_time_ms=result.retrieval_stats.get('mmr_ms', 0),
            query_length=len(test_case['query']),
            entity_count=len(test_case['entities']),
            expanded_entity_count=len(result.expanded_entities),
            candidate_count=result.retrieval_stats.get('candidates', 0),
            bullet_count=len(result.bullets),
            complexity_level=test_case['complexity']
        )

        return RetrievalBenchmark(
            query=test_case['query'],
            entities=test_case['entities'],
            retriever_type=retriever_type,
            timing_profile=timing_profile,
            bullets=result.bullets,
            relevant_triples=result.relevant_triples
        )

    def analyze_bottlenecks(self, results):
        """Analyze performance bottlenecks in detail"""
        print("\n🔍 Analyzing performance bottlenecks...")

        analysis = {
            'original': self._analyze_retriever_results(results['original']),
            'optimized': self._analyze_retriever_results(results['optimized'])
        }

        # Compare performance
        print("\n📊 PERFORMANCE COMPARISON:")
        print("-" * 50)

        for complexity in ['low', 'medium', 'high']:
            orig_complexity = [r for r in results['original'] if r.timing_profile.complexity_level == complexity]
            opt_complexity = [r for r in results['optimized'] if r.timing_profile.complexity_level == complexity]

            if orig_complexity and opt_complexity:
                orig_avg = statistics.mean(r.timing_profile.total_time_ms for r in orig_complexity)
                opt_avg = statistics.mean(r.timing_profile.total_time_ms for r in opt_complexity)

                print(f"\n{complexity.upper()} Complexity Queries:")
                print(f"   Original: {orig_avg:.1f}ms")
                print(f"   Optimized: {opt_avg:.1f}ms")
                print(f"   Improvement: {((orig_avg - opt_avg) / orig_avg * 100):+.1f}%")

        # Identify bottlenecks by component
        print("\n🚨 BOTTLENECK ANALYSIS:")
        print("-" * 50)

        for retriever_type, analyzer in analysis.items():
            print(f"\n{retriever_type.upper()} Retrievers:")

            components = [
                ('Entity Expansion', 'expansion_time_ms'),
                ('Candidate Gathering', 'gathering_time_ms'),
                ('MMR Selection', 'mmr_time_ms')
            ]

            for component_name, time_field in components:
                avg_time = statistics.mean(getattr(profile, time_field) for profile in analyzer['profiles'])
                percentage = analyzer['breakdown_percentages'][component_name]

                print(f"   {component_name}: {avg_time:.1f}ms ({percentage:.1f}%)")

                # Highlight bottlenecks
                if percentage > 50:
                    print(f"   🚨 MAJOR BOTTLENECK: {component_name} takes {percentage:.1f}% of time")
                elif percentage > 30:
                    print(f"   ⚠️  Moderate bottleneck: {component_name} takes {percentage:.1f}% of time")

        return analysis

    def _analyze_retriever_results(self, results):
        """Analyze results for a single retriever type"""
        if not results:
            return {}

        profiles = [r.timing_profile for r in results]

        # Calculate timing breakdown percentages
        total_times = [p.total_time_ms for p in profiles]
        expansion_times = [p.expansion_time_ms for p in profiles]
        gathering_times = [p.gathering_time_ms for p in profiles]
        mmr_times = [p.mmr_time_ms for p in profiles]

        avg_total = statistics.mean(total_times)

        breakdown_percentages = {
            'Entity Expansion': statistics.mean(expansion_times) / avg_total * 100 if avg_total > 0 else 0,
            'Candidate Gathering': statistics.mean(gathering_times) / avg_total * 100 if avg_total > 0 else 0,
            'MMR Selection': statistics.mean(mmr_times) / avg_total * 100 if avg_total > 0 else 0
        }

        return {
            'profiles': profiles,
            'breakdown_percentages': breakdown_percentages,
            'avg_total_time': avg_total,
            'p95_total_time': statistics.quantiles(total_times, n=20)[18] if len(total_times) > 1 else total_times[0]
        }

    def identify_optimization_opportunities(self, analysis):
        """Identify specific optimization opportunities"""
        print("\n💡 OPTIMIZATION OPPORTUNITIES:")
        print("-" * 50)

        opportunities = []

        # Analyze bottlenecks for both retrievers
        for retriever_type, analyzer in analysis.items():
            breakdown = analyzer['breakdown_percentages']

            # Entity expansion optimizations
            if breakdown['Entity Expansion'] > 30:
                opportunities.append({
                    'component': 'Entity Expansion',
                    'issue': f'Slow expansion ({breakdown["Entity Expansion"]:.1f}% of time)',
                    'severity': 'high' if breakdown['Entity Expansion'] > 50 else 'medium',
                    'suggestions': [
                        'Implement early termination in multi-hop expansion',
                        'Limit expansion depth based on query complexity',
                        'Cache expansion results for common entities'
                    ]
                })

            # Candidate gathering optimizations
            if breakdown['Candidate Gathering'] > 40:
                opportunities.append({
                    'component': 'Candidate Gathering',
                    'issue': f'Slow candidate gathering ({breakdown["Candidate Gathering"]:.1f}% of time)',
                    'severity': 'high' if breakdown['Candidate Gathering'] > 60 else 'medium',
                    'suggestions': [
                        'Pre-filter entities before scoring',
                        'Implement batch scoring operations',
                        'Use more efficient similarity calculations'
                    ]
                })

            # MMR optimizations
            if breakdown['MMR Selection'] > 40:
                opportunities.append({
                    'component': 'MMR Selection',
                    'issue': f'Slow MMR selection ({breakdown["MMR Selection"]:.1f}% of time)',
                    'severity': 'high' if breakdown['MMR Selection'] > 60 else 'medium',
                    'suggestions': [
                        'Limit candidate pool size before MMR',
                        'Optimize similarity calculations',
                        'Implement early termination in MMR loop'
                    ]
                })

        # Additional optimization opportunities
        opportunities.extend([
            {
                'component': 'Memory Access',
                'issue': 'Linear search through entity index',
                'severity': 'medium',
                'suggestions': [
                    'Implement better indexing for entities',
                    'Use more efficient data structures',
                    'Consider caching frequently accessed entities'
                ]
            },
            {
                'component': 'Algorithm Selection',
                'issue': 'One-size-fits-all approach',
                'severity': 'medium',
                'suggestions': [
                    'Use different strategies based on query complexity',
                    'Implement fallback to simpler algorithms',
                    'Add query-specific optimization paths'
                ]
            }
        ])

        # Print opportunities
        for i, opp in enumerate(opportunities, 1):
            severity_icon = "🔥" if opp['severity'] == 'high' else "⚠️"
            print(f"\n{severity_icon} {i}. {opp['component']} ({opp['severity'].upper()})")
            print(f"   Issue: {opp['issue']}")
            print(f"   Suggestions:")
            for suggestion in opp['suggestions']:
                print(f"     • {suggestion}")

        return opportunities

    def generate_performance_report(self, results, analysis, opportunities):
        """Generate comprehensive performance report"""
        print("\n📋 RETRIEVAL PERFORMANCE REPORT")
        print("=" * 60)

        # Performance summary
        original_times = [r.timing_profile.total_time_ms for r in results['original']]
        optimized_times = [r.timing_profile.total_time_ms for r in results['optimized']]

        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Original retriever: {statistics.mean(original_times):.1f}ms avg, {statistics.quantiles(original_times, n=20)[18]:.1f}ms p95")
        print(f"   Optimized retriever: {statistics.mean(optimized_times):.1f}ms avg, {statistics.quantiles(optimized_times, n=20)[18]:.1f}ms p95")

        improvement = (statistics.mean(original_times) - statistics.mean(optimized_times)) / statistics.mean(original_times) * 100
        print(f"   Overall improvement: {improvement:+.1f}%")

        # Complexity-based analysis
        print(f"\n📈 COMPLEXITY-BASED ANALYSIS:")
        for complexity in ['low', 'medium', 'high']:
            orig_results = [r for r in results['original'] if r.timing_profile.complexity_level == complexity]
            opt_results = [r for r in results['optimized'] if r.timing_profile.complexity_level == complexity]

            if orig_results and opt_results:
                orig_avg = statistics.mean(r.timing_profile.total_time_ms for r in orig_results)
                opt_avg = statistics.mean(r.timing_profile.total_time_ms for r in opt_results)
                complexity_improvement = (orig_avg - opt_avg) / orig_avg * 100

                print(f"   {complexity.upper()}: {orig_avg:.1f}ms → {opt_avg:.1f}ms ({complexity_improvement:+.1f}%)")

        # Key bottlenecks
        print(f"\n🚨 KEY BOTTLENECKS:")
        for retriever_type, analyzer in analysis.items():
            breakdown = analyzer['breakdown_percentages']
            worst_component = max(breakdown.items(), key=lambda x: x[1])
            print(f"   {retriever_type.upper()}: {worst_component[0]} ({worst_component[1]:.1f}%)")

        # Top recommendations
        high_priority = [opp for opp in opportunities if opp['severity'] == 'high']
        if high_priority:
            print(f"\n🎯 TOP RECOMMENDATIONS:")
            for i, opp in enumerate(high_priority[:3], 1):
                print(f"   {i}. {opp['suggestions'][0]}")

        return {
            'performance_summary': {
                'original_avg_ms': statistics.mean(original_times),
                'optimized_avg_ms': statistics.mean(optimized_times),
                'improvement_percent': improvement
            },
            'complexity_analysis': analysis,
            'bottlenecks': {retriever_type: max(analyzer['breakdown_percentages'].items(), key=lambda x: x[1])
                          for retriever_type, analyzer in analysis.items()},
            'recommendations': opportunities
        }

    def run_analysis(self):
        """Run complete retrieval performance analysis"""
        print("🚀 Starting retrieval performance analysis...")

        # Profile retrieval performance
        results = self.profile_retrieval_performance()

        # Analyze bottlenecks
        analysis = self.analyze_bottlenecks(results)

        # Identify optimization opportunities
        opportunities = self.identify_optimization_opportunities(analysis)

        # Generate performance report
        report = self.generate_performance_report(results, analysis, opportunities)

        # Save detailed results
        detailed_results = {
            'timestamp': time.time(),
            'test_queries': self.test_queries,
            'results': self._serialize_results(results),
            'analysis': analysis,
            'opportunities': opportunities,
            'report': report
        }

        with open('retrieval_analysis_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        print(f"\n📁 Detailed results saved to: retrieval_analysis_results.json")
        print("\n✅ Retrieval performance analysis complete!")

        return report

    def _serialize_results(self, results):
        """Serialize results for JSON storage"""
        serialized = {}
        for retriever_type, benchmarks in results.items():
            serialized[retriever_type] = [
                {
                    'query': b.query,
                    'entities': b.entities,
                    'retriever_type': b.retriever_type,
                    'timing_profile': {
                        'operation_name': b.timing_profile.operation_name,
                        'total_time_ms': b.timing_profile.total_time_ms,
                        'expansion_time_ms': b.timing_profile.expansion_time_ms,
                        'gathering_time_ms': b.timing_profile.gathering_time_ms,
                        'mmr_time_ms': b.timing_profile.mmr_time_ms,
                        'query_length': b.timing_profile.query_length,
                        'entity_count': b.timing_profile.entity_count,
                        'expanded_entity_count': b.timing_profile.expanded_entity_count,
                        'candidate_count': b.timing_profile.candidate_count,
                        'bullet_count': b.timing_profile.bullet_count,
                        'complexity_level': b.timing_profile.complexity_level
                    },
                    'bullet_count': len(b.bullets),
                    'triple_count': len(b.relevant_triples)
                }
                for b in benchmarks
            ]
        return serialized


class MockMemoryStore:
    """Mock memory store for testing"""
    def search_fts_detailed(self, query, limit=10):
        # Return empty list for FTS (disabled in our test)
        return []


if __name__ == "__main__":
    analyzer = RetrievalPerformanceAnalyzer()
    report = analyzer.run_analysis()