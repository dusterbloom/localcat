#!/usr/bin/env python3.12
"""
Comprehensive A/B Test for Retrieval Quality and Performance
========================================================

Compares MemoryRetriever vs MemoryRetrieverOptimized across:
- Performance metrics (latency, throughput)
- Quality metrics (relevance, coverage, diversity)
- Edge case handling
- Memory usage
"""

import time
import json
import statistics
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import cProfile
import pstats
import io
import sys
import os
import hashlib

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.retrieval.memory_retriever import MemoryRetriever, RetrievalResult
from components.retrieval.memory_retriever_optimized import MemoryRetrieverOptimized

@dataclass
class TestCase:
    """Single test case for retrieval comparison"""
    query: str
    entities: List[str]
    expected_bullets: int
    expected_entities: List[str]
    complexity: str  # 'low', 'medium', 'high'
    category: str  # 'direct_lookup', 'multi_entity', 'semantic_inference', 'edge_case'

@dataclass
class RetrievalResultAnalysis:
    """Analysis of a single retrieval result"""
    bullets: List[str]
    bullet_count: int
    entity_coverage: float  # % of expected entities found
    relevance_score: float  # 0-1 based on query relevance
    diversity_score: float  # 0-1 based on information diversity
    execution_time_ms: float
    timing_breakdown: Dict[str, float]
    candidate_count: int
    expansion_count: int
    error_occurred: bool
    error_message: str = ""

@dataclass
class ABTestResult:
    """A/B test comparison result"""
    test_case: TestCase
    original_result: RetrievalResultAnalysis
    optimized_result: RetrievalResultAnalysis
    performance_improvement: float  # % improvement
    quality_delta: float  # quality difference (-1 to 1)
    winner: str  # 'original', 'optimized', 'tie'

class RetrievalQualityEvaluator:
    """Evaluates the quality of retrieval results"""

    def __init__(self):
        # Quality scoring weights
        self.relevance_weight = 0.4
        self.coverage_weight = 0.3
        self.diversity_weight = 0.3

    def calculate_relevance_score(self, query: str, bullets: List[str]) -> float:
        """Calculate relevance score based on query-bullet match"""
        if not bullets:
            return 0.0

        query_tokens = set(self._tokenize_text(query.lower()))
        total_relevance = 0.0

        for bullet in bullets:
            bullet_tokens = set(self._tokenize_text(bullet.lower()))
            if query_tokens and bullet_tokens:
                # Calculate Jaccard similarity
                intersection = len(query_tokens & bullet_tokens)
                union = len(query_tokens | bullet_tokens)
                similarity = intersection / union if union > 0 else 0
                total_relevance += similarity

        return min(1.0, total_relevance / len(bullets))

    def calculate_entity_coverage(self, expected_entities: List[str], bullets: List[str]) -> float:
        """Calculate coverage of expected entities in results"""
        if not expected_entities or not bullets:
            return 0.0

        found_entities = set()
        for bullet in bullets:
            for entity in expected_entities:
                if entity.lower() in bullet.lower():
                    found_entities.add(entity)

        return len(found_entities) / len(expected_entities)

    def calculate_diversity_score(self, bullets: List[str]) -> float:
        """Calculate information diversity score"""
        if len(bullets) <= 1:
            return 1.0

        # Calculate pairwise bullet similarity
        similarities = []
        for i in range(len(bullets)):
            for j in range(i + 1, len(bullets)):
                tokens_i = set(self._tokenize_text(bullets[i].lower()))
                tokens_j = set(self._tokenize_text(bullets[j].lower()))

                if tokens_i and tokens_j:
                    intersection = len(tokens_i & tokens_j)
                    union = len(tokens_i | tokens_j)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)

        # Diversity = 1 - average similarity
        avg_similarity = statistics.mean(similarities) if similarities else 0
        return 1.0 - avg_similarity

    def calculate_quality_score(self, query: str, expected_entities: List[str],
                              bullets: List[str]) -> float:
        """Calculate overall quality score"""
        relevance = self.calculate_relevance_score(query, bullets)
        coverage = self.calculate_entity_coverage(expected_entities, bullets)
        diversity = self.calculate_diversity_score(bullets)

        quality_score = (
            self.relevance_weight * relevance +
            self.coverage_weight * coverage +
            self.diversity_weight * diversity
        )

        return min(1.0, quality_score)

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple text tokenization"""
        import re
        return re.findall(r'\b\w+\b', text.lower())

class RetrievalABTester:
    """A/B tester for retrieval quality and performance"""

    def __init__(self):
        self.evaluator = RetrievalQualityEvaluator()
        self.test_cases = self._create_comprehensive_test_cases()
        self.mock_store = MockMemoryStore()
        entity_index, edge_metadata = self._create_test_entity_index()

        # Initialize both retrievers
        config = {
            'use_leann': False,
            'retrieval_fusion': False,
            'leann_complexity': 16
        }

        self.original_retriever = MemoryRetriever(
            store=self.mock_store,
            entity_index=entity_index,
            config=config
        )
        self.original_retriever.edge_meta = edge_metadata

        self.optimized_retriever = MemoryRetrieverOptimized(
            store=self.mock_store,
            entity_index=entity_index,
            config=config
        )
        self.optimized_retriever.edge_meta = edge_metadata

    def _create_comprehensive_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering different scenarios"""
        test_cases = []

        # Direct lookup queries (LOW complexity)
        test_cases.extend([
            TestCase(
                query="Where does Michael work?",
                entities=["michael"],
                expected_bullets=1,
                expected_entities=["michael", "apple"],
                complexity="low",
                category="direct_lookup"
            ),
            TestCase(
                query="Tell me about Sarah Chen",
                entities=["sarah chen"],
                expected_bullets=2,
                expected_entities=["sarah", "google", "stanford"],
                complexity="low",
                category="direct_lookup"
            ),
            TestCase(
                query="Where does Sarah live?",
                entities=["sarah"],
                expected_bullets=1,
                expected_entities=["sarah", "new york"],
                complexity="low",
                category="direct_lookup"
            )
        ])

        # Multi-entity queries (MEDIUM complexity)
        test_cases.extend([
            TestCase(
                query="Where do Michael and Sarah live?",
                entities=["michael", "sarah"],
                expected_bullets=2,
                expected_entities=["michael", "sarah", "san francisco", "new york"],
                complexity="medium",
                category="multi_entity"
            ),
            TestCase(
                query="What do you know about Michael's job and education?",
                entities=["michael"],
                expected_bullets=3,
                expected_entities=["michael", "apple", "designer", "risd"],
                complexity="medium",
                category="multi_entity"
            ),
            TestCase(
                query="Tell me about people who work at tech companies",
                entities=["apple", "google"],
                expected_bullets=4,
                expected_entities=["michael", "sarah", "apple", "google"],
                complexity="medium",
                category="multi_entity"
            )
        ])

        # Semantic inference queries (HIGH complexity)
        test_cases.extend([
            TestCase(
                query="Tell me about people who work at Apple and where they live",
                entities=["apple"],
                expected_bullets=3,
                expected_entities=["apple", "michael", "san francisco"],
                complexity="high",
                category="semantic_inference"
            ),
            TestCase(
                query="Where do people who studied computer science work?",
                entities=["computer science", "stanford"],
                expected_bullets=2,
                expected_entities=["sarah", "google"],
                complexity="high",
                category="semantic_inference"
            ),
            TestCase(
                query="What do you know about Michael's family background?",
                entities=["michael"],
                expected_bullets=3,
                expected_entities=["michael", "parents", "seattle", "sarah"],
                complexity="high",
                category="semantic_inference"
            )
        ])

        # Edge cases
        test_cases.extend([
            TestCase(
                query="Tell me about someone who doesn't exist",
                entities=["nonexistent_person"],
                expected_bullets=0,
                expected_entities=[],
                complexity="low",
                category="edge_case"
            ),
            TestCase(
                query="What do you know about X?",
                entities=["x"],  # Single letter entity
                expected_bullets=0,
                expected_entities=[],
                complexity="low",
                category="edge_case"
            ),
            TestCase(
                query="",
                entities=[],
                expected_bullets=0,
                expected_entities=[],
                complexity="low",
                category="edge_case"
            )
        ])

        return test_cases

    def _create_test_entity_index(self) -> Tuple[Dict[str, Set], Dict]:
        """Create comprehensive test entity index"""
        entity_index = defaultdict(set)
        edge_metadata = {}

        # Rich knowledge graph
        test_triples = [
            # Michael's complete profile
            ("michael", "works_at", "apple"),
            ("michael", "lives_in", "san francisco"),
            ("michael", "studied_at", "risd"),
            ("michael", "graduated_in", "2015"),
            ("michael", "has_profession", "designer"),
            ("michael", "also_known_as", "mike"),
            ("michael", "married_to", "sarah"),
            ("michael", "has_parents", "michael parents"),
            ("michael", "has_salary", "120000"),
            ("michael", "works_in_department", "design"),

            # Sarah's complete profile
            ("sarah", "works_at", "google"),
            ("sarah", "lives_in", "new york"),
            ("sarah", "studied_at", "stanford"),
            ("sarah", "graduated_in", "2017"),
            ("sarah", "has_profession", "software engineer"),
            ("sarah", "married_to", "michael"),
            ("sarah", "has_parents", "sarah parents"),
            ("sarah", "has_salary", "150000"),
            ("sarah", "works_in_department", "engineering"),
            ("sarah", "specializes_in", "computer science"),
            ("sarah", "also_known_as", "sarah chen"),

            # Company information
            ("apple", "located_in", "cupertino"),
            ("apple", "industry", "technology"),
            ("apple", "founded_by", "steve jobs"),
            ("apple", "has_employees", "150000"),
            ("google", "located_in", "mountain view"),
            ("google", "industry", "technology"),
            ("google", "founded_by", "larry page"),
            ("google", "has_employees", "180000"),

            # Educational institutions
            ("stanford", "located_in", "palo alto"),
            ("stanford", "type", "university"),
            ("stanford", "known_for", "computer science"),
            ("stanford", "founded_in", "1885"),
            ("risd", "located_in", "rhode island"),
            ("risd", "type", "design school"),
            ("risd", "known_for", "art and design"),
            ("risd", "founded_in", "1877"),

            # Location hierarchy
            ("san francisco", "located_in", "california"),
            ("san francisco", "type", "city"),
            ("san francisco", "has_population", "874961"),
            ("new york", "located_in", "new york state"),
            ("new york", "type", "city"),
            ("new york", "has_population", "8336817"),
            ("cupertino", "located_in", "california"),
            ("cupertino", "type", "city"),
            ("mountain view", "located_in", "california"),
            ("mountain view", "type", "city"),
            ("palo alto", "located_in", "california"),
            ("palo alto", "type", "city"),
            ("california", "located_in", "united states"),
            ("california", "type", "state"),
            ("seattle", "located_in", "washington"),
            ("seattle", "type", "city"),
            ("boston", "located_in", "massachusetts"),
            ("boston", "type", "city"),

            # Family relationships
            ("michael parents", "live_in", "seattle"),
            ("michael parents", "have_child", "michael"),
            ("michael parents", "type", "parents"),
            ("sarah parents", "live_in", "boston"),
            ("sarah parents", "have_child", "sarah"),
            ("sarah parents", "type", "parents"),
            ("parents", "live_in", "seattle"),  # Duplicate for testing
            ("parents", "have_children", "michael"),
            ("parents", "have_children", "sarah"),

            # Additional relationships for complexity
            ("michael", "has_skill", "design"),
            ("michael", "has_skill", "prototyping"),
            ("michael", "has_experience", "ui design"),
            ("sarah", "has_skill", "programming"),
            ("sarah", "has_skill", "machine learning"),
            ("sarah", "has_experience", "software development"),
            ("apple", "specializes_in", "consumer electronics"),
            ("google", "specializes_in", "internet services"),
            ("stanford", "offers_program", "computer science"),
            ("risd", "offers_program", "graphic design"),
        ]

        # Add metadata with realistic timestamps and weights
        current_time = int(time.time() * 1000)
        for i, (s, r, d) in enumerate(test_triples):
            # Add to entity index
            entity_index[s].add((s, r, d))
            entity_index[d].add((s, r, d))

            # Add metadata with varying confidence and timestamps
            edge_metadata[(s, r, d)] = {
                'ts': current_time - (i * 86400000),  # Spread over days
                'weight': 0.5 + (i % 10) * 0.05,  # Varying confidence (0.5-0.95)
                'source': 'test_data',
                'extraction_method': 'manual'
            }

        return dict(entity_index), edge_metadata

    def run_single_test(self, test_case: TestCase) -> ABTestResult:
        """Run A/B test for a single test case"""
        print(f"\n🧪 Testing: {test_case.query}")
        print(f"   Entities: {test_case.entities}")
        print(f"   Complexity: {test_case.complexity}, Category: {test_case.category}")

        # Test original retriever
        print("   Testing original retriever...")
        original_analysis = self._test_retriever(
            self.original_retriever, test_case, "original"
        )

        # Test optimized retriever
        print("   Testing optimized retriever...")
        optimized_analysis = self._test_retriever(
            self.optimized_retriever, test_case, "optimized"
        )

        # Calculate metrics
        performance_improvement = self._calculate_performance_improvement(
            original_analysis, optimized_analysis
        )

        quality_delta = optimized_analysis.relevance_score - original_analysis.relevance_score

        # Determine winner
        winner = self._determine_winner(original_analysis, optimized_analysis)

        return ABTestResult(
            test_case=test_case,
            original_result=original_analysis,
            optimized_result=optimized_analysis,
            performance_improvement=performance_improvement,
            quality_delta=quality_delta,
            winner=winner
        )

    def _test_retriever(self, retriever, test_case: TestCase, retriever_name: str) -> RetrievalResultAnalysis:
        """Test a single retriever and analyze results"""
        try:
            start_time = time.perf_counter()

            result = retriever.retrieve_context(
                query=test_case.query,
                entities=test_case.entities,
                turn_id=0
            )

            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            # Calculate quality metrics
            relevance_score = self.evaluator.calculate_relevance_score(
                test_case.query, result.bullets
            )
            entity_coverage = self.evaluator.calculate_entity_coverage(
                test_case.expected_entities, result.bullets
            )
            diversity_score = self.evaluator.calculate_diversity_score(result.bullets)

            return RetrievalResultAnalysis(
                bullets=result.bullets,
                bullet_count=len(result.bullets),
                entity_coverage=entity_coverage,
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                execution_time_ms=execution_time_ms,
                timing_breakdown=result.retrieval_stats,
                candidate_count=result.retrieval_stats.get('candidates', 0),
                expansion_count=len(result.expanded_entities),
                error_occurred=False
            )

        except Exception as e:
            return RetrievalResultAnalysis(
                bullets=[],
                bullet_count=0,
                entity_coverage=0.0,
                relevance_score=0.0,
                diversity_score=0.0,
                execution_time_ms=0.0,
                timing_breakdown={},
                candidate_count=0,
                expansion_count=0,
                error_occurred=True,
                error_message=str(e)
            )

    def _calculate_performance_improvement(self, original: RetrievalResultAnalysis,
                                         optimized: RetrievalResultAnalysis) -> float:
        """Calculate percentage performance improvement"""
        if original.execution_time_ms == 0:
            return 0.0
        improvement = (original.execution_time_ms - optimized.execution_time_ms) / original.execution_time_ms * 100
        return improvement

    def _determine_winner(self, original: RetrievalResultAnalysis,
                        optimized: RetrievalResultAnalysis) -> str:
        """Determine winner based on performance and quality"""
        if optimized.error_occurred and not original.error_occurred:
            return "original"
        if original.error_occurred and not optimized.error_occurred:
            return "optimized"

        # Performance threshold: 20% improvement
        perf_threshold = 20.0
        perf_improvement = (original.execution_time_ms - optimized.execution_time_ms) / original.execution_time_ms * 100

        # Quality threshold: 5% difference
        quality_threshold = 0.05
        quality_diff = optimized.relevance_score - original.relevance_score

        if perf_improvement > perf_threshold and abs(quality_diff) < quality_threshold:
            return "optimized"
        elif quality_diff > quality_threshold:
            return "optimized"
        elif abs(quality_diff) < quality_threshold and abs(perf_improvement) < perf_threshold:
            return "tie"
        else:
            return "original"

    def run_comprehensive_ab_test(self) -> Dict[str, Any]:
        """Run comprehensive A/B test across all scenarios"""
        print("🚀 Starting Comprehensive A/B Test...")
        print(f"📊 Testing {len(self.test_cases)} scenarios across both retrievers")

        results = []

        for i, test_case in enumerate(self.test_cases):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(self.test_cases)}")
            print(f"{'='*60}")

            result = self.run_single_test(test_case)
            results.append(result)

            # Print immediate results
            self._print_test_result(result)

        # Analyze overall results
        analysis = self._analyze_ab_results(results)

        # Generate report
        report = self._generate_ab_report(results, analysis)

        return {
            'results': [self._serialize_result(r) for r in results],
            'analysis': analysis,
            'report': report
        }

    def _print_test_result(self, result: ABTestResult):
        """Print individual test result"""
        test_case = result.test_case
        orig = result.original_result
        opt = result.optimized_result

        print(f"\n📊 Results for: '{test_case.query}'")
        print(f"   Original:  {orig.execution_time_ms:.2f}ms, {orig.bullet_count} bullets, "
              f"relevance={orig.relevance_score:.2f}")
        print(f"   Optimized: {opt.execution_time_ms:.2f}ms, {opt.bullet_count} bullets, "
              f"relevance={opt.relevance_score:.2f}")
        print(f"   Improvement: {result.performance_improvement:+.1f}%, "
              f"Quality Δ: {result.quality_delta:+.2f}")
        print(f"   Winner: {result.winner.upper()}")

        if orig.error_occurred:
            print(f"   ❌ Original error: {orig.error_message}")
        if opt.error_occurred:
            print(f"   ❌ Optimized error: {opt.error_message}")

    def _analyze_ab_results(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Analyze A/B test results"""
        analysis = {
            'total_tests': len(results),
            'performance_comparison': {},
            'quality_comparison': {},
            'winner_distribution': {'original': 0, 'optimized': 0, 'tie': 0},
            'category_analysis': defaultdict(lambda: {'original': 0, 'optimized': 0, 'tie': 0}),
            'complexity_analysis': defaultdict(lambda: {'original': 0, 'optimized': 0, 'tie': 0}),
            'error_analysis': {'original_errors': 0, 'optimized_errors': 0}
        }

        # Performance metrics
        original_times = [r.original_result.execution_time_ms for r in results if not r.original_result.error_occurred]
        optimized_times = [r.optimized_result.execution_time_ms for r in results if not r.optimized_result.error_occurred]

        if original_times and optimized_times:
            analysis['performance_comparison'] = {
                'original_avg_ms': statistics.mean(original_times),
                'original_p95_ms': statistics.quantiles(original_times, n=20)[18],
                'optimized_avg_ms': statistics.mean(optimized_times),
                'optimized_p95_ms': statistics.quantiles(optimized_times, n=20)[18],
                'avg_improvement_percent': statistics.mean([r.performance_improvement for r in results]),
                'tests_with_significant_improvement': len([r for r in results if r.performance_improvement > 20])
            }

        # Quality metrics
        original_quality = [r.original_result.relevance_score for r in results if not r.original_result.error_occurred]
        optimized_quality = [r.optimized_result.relevance_score for r in results if not r.optimized_result.error_occurred]

        if original_quality and optimized_quality:
            analysis['quality_comparison'] = {
                'original_avg_quality': statistics.mean(original_quality),
                'optimized_avg_quality': statistics.mean(optimized_quality),
                'quality_delta': statistics.mean([r.quality_delta for r in results]),
                'tests_with_quality_improvement': len([r for r in results if r.quality_delta > 0.05]),
                'tests_with_quality_degradation': len([r for r in results if r.quality_delta < -0.05])
            }

        # Winner distribution
        for result in results:
            analysis['winner_distribution'][result.winner] += 1
            analysis['category_analysis'][result.test_case.category][result.winner] += 1
            analysis['complexity_analysis'][result.test_case.complexity][result.winner] += 1

            if result.original_result.error_occurred:
                analysis['error_analysis']['original_errors'] += 1
            if result.optimized_result.error_occurred:
                analysis['error_analysis']['optimized_errors'] += 1

        return analysis

    def _generate_ab_report(self, results: List[ABTestResult], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive A/B test report"""
        report = []

        report.append("🔍 RETRIEVAL A/B TEST REPORT")
        report.append("=" * 60)

        # Executive summary
        perf_comp = analysis['performance_comparison']
        quality_comp = analysis['quality_comparison']

        if perf_comp:
            report.append(f"\n📊 PERFORMANCE SUMMARY:")
            report.append(f"   Original:  {perf_comp['original_avg_ms']:.2f}ms avg, {perf_comp['original_p95_ms']:.2f}ms p95")
            report.append(f"   Optimized: {perf_comp['optimized_avg_ms']:.2f}ms avg, {perf_comp['optimized_p95_ms']:.2f}ms p95")
            report.append(f"   Improvement: {perf_comp['avg_improvement_percent']:.1f}% average")

        if quality_comp:
            report.append(f"\n🎯 QUALITY SUMMARY:")
            report.append(f"   Original:  {quality_comp['original_avg_quality']:.2f} avg relevance")
            report.append(f"   Optimized: {quality_comp['optimized_avg_quality']:.2f} avg relevance")
            report.append(f"   Quality Δ: {quality_comp['quality_delta']:+.3f}")

        # Winner distribution
        winners = analysis['winner_distribution']
        total_tests = sum(winners.values())
        report.append(f"\n🏆 WINNER DISTRIBUTION:")
        for winner, count in winners.items():
            percentage = (count / total_tests * 100) if total_tests > 0 else 0
            report.append(f"   {winner.upper()}: {count}/{total_tests} ({percentage:.1f}%)")

        # Error analysis
        errors = analysis['error_analysis']
        if errors['original_errors'] > 0 or errors['optimized_errors'] > 0:
            report.append(f"\n⚠️ ERROR ANALYSIS:")
            report.append(f"   Original errors: {errors['original_errors']}")
            report.append(f"   Optimized errors: {errors['optimized_errors']}")

        # Category breakdown
        report.append(f"\n📈 CATEGORY BREAKDOWN:")
        for category, counts in analysis['category_analysis'].items():
            total = sum(counts.values())
            report.append(f"   {category}:")
            for winner, count in counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                report.append(f"     {winner}: {count}/{total} ({percentage:.1f}%)")

        # Recommendation
        report.append(f"\n💡 RECOMMENDATION:")
        optimized_wins = winners.get('optimized', 0)
        total_tests = sum(winners.values())

        if optimized_wins / total_tests > 0.7 and errors['optimized_errors'] == 0:
            report.append("   ✅ SWITCH TO OPTIMIZED - Clear performance win with maintained quality")
        elif optimized_wins / total_tests > 0.5:
            report.append("   ⚠️ CONSIDER SWITCHING - Performance gains outweigh minor quality differences")
        else:
            report.append("   ❌ KEEP ORIGINAL - Quality or reliability concerns with optimized version")

        return "\n".join(report)

    def _serialize_result(self, result: ABTestResult) -> Dict[str, Any]:
        """Serialize result for JSON storage"""
        return {
            'test_case': asdict(result.test_case),
            'original_result': asdict(result.original_result),
            'optimized_result': asdict(result.optimized_result),
            'performance_improvement': result.performance_improvement,
            'quality_delta': result.quality_delta,
            'winner': result.winner
        }

class MockMemoryStore:
    """Mock memory store for testing"""
    def search_fts_detailed(self, query, limit=10):
        return []

if __name__ == "__main__":
    tester = RetrievalABTester()
    report_data = tester.run_comprehensive_ab_test()

    # Save detailed results
    with open('retrieval_ab_test_results.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n📁 Detailed A/B test results saved to: retrieval_ab_test_results.json")
    print("\n✅ A/B testing complete!")