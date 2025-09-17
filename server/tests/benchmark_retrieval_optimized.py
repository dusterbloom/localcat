#!/usr/bin/env python3
"""
Benchmark comparison between original and optimized MemoryRetriever.
Shows performance improvements from quick win optimizations.
"""

import os
import sys
import time
import sqlite3
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
from components.retrieval.memory_retriever import MemoryRetriever
from components.retrieval.memory_retriever_optimized import MemoryRetrieverOptimized
from components.memory.memory_store import MemoryStore
from components.memory.memory_store import Paths


class RetrievalBenchmark:
    """Benchmark retrieval performance comparison"""

    def __init__(self, db_path: str = None):
        """Initialize benchmark with test database"""
        self.db_path = db_path or "/tmp/hotmem_benchmark.db"
        self.setup_test_data()

    def setup_test_data(self):
        """Set up test database with realistic data"""
        logger.info(f"Setting up benchmark data at {self.db_path}")

        # Initialize memory store
        paths = Paths()
        paths.sqlite_path = self.db_path
        paths.lmdb_dir = self.db_path + ".lmdb"
        self.store = MemoryStore(paths)

        # Create test data with varying sizes
        now_ts = int(time.time() * 1000)

        # Core edges (user profile)
        core_edges = [
            ("you", "lives_in", "San Francisco", 0.9, now_ts - 1000),
            ("you", "works_at", "Anthropic", 0.85, now_ts - 5000),
            ("you", "name", "Alex Chen", 0.95, now_ts - 10000),
            ("you", "has", "dog", 0.8, now_ts - 50000),
            ("dog", "name", "Max", 0.9, now_ts - 60000),
            ("you", "likes", "coffee", 0.7, now_ts - 70000),
            ("you", "born_in", "Seattle", 0.8, now_ts - 100000),
            ("you", "studied_at", "MIT", 0.85, now_ts - 200000),
        ]

        # Add relationships
        relationships = [
            ("San Francisco", "is", "city", 0.7, now_ts - 20000),
            ("San Francisco", "located_in", "California", 0.8, now_ts - 21000),
            ("Anthropic", "is", "AI company", 0.8, now_ts - 30000),
            ("Anthropic", "founded_by", "Dario Amodei", 0.7, now_ts - 31000),
            ("MIT", "is", "university", 0.8, now_ts - 210000),
            ("MIT", "located_in", "Cambridge", 0.8, now_ts - 211000),
        ]

        # Add noise data for realistic load
        noise_edges = []
        entities = ["entity", "person", "place", "thing", "concept", "idea", "object", "item"]
        relations = ["knows", "likes", "has", "works_with", "visited", "created", "owns", "uses"]

        for i in range(200):  # Add 200 noise edges
            s = f"{entities[i % len(entities)]}_{i // 10}"
            r = relations[i % len(relations)]
            d = f"object_{i}"
            conf = 0.4 + (i % 5) * 0.1
            ts = now_ts - (i * 5000)
            noise_edges.append((s, r, d, conf, ts))

        # Store all edges
        all_edges = core_edges + relationships + noise_edges
        for s, r, d, conf, ts in all_edges:
            self.store.observe_edge(s, r, d, conf, ts)
            # Add to FTS
            self.store.enqueue_mention(s, f"{s} {r} {d}", ts, "benchmark_session", 0)
            self.store.enqueue_mention(d, f"{s} {r} {d}", ts, "benchmark_session", 0)

        # Add some summaries
        summaries = [
            ("You mentioned living in San Francisco and working at Anthropic on AI safety", now_ts - 1000),
            ("You have a dog named Max who likes to play fetch in the park", now_ts - 50000),
            ("You studied computer science at MIT before moving to San Francisco", now_ts - 200000),
        ]

        for text, ts in summaries:
            self.store.enqueue_mention(f"summary:{ts}", text, ts, "benchmark_session", 0)

        self.store.flush()
        logger.info(f"Test data setup complete: {len(all_edges)} edges")

        # Build entity index
        self.entity_index = defaultdict(set)
        self.edge_meta = {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT src, rel, dst, weight, updated_at FROM edge")

        for s, r, d, w, ts in cursor.fetchall():
            triple = (s, r, d)
            self.entity_index[s].add(triple)
            self.entity_index[d].add(triple)
            self.edge_meta[triple] = {'weight': w, 'ts': ts}

        conn.close()
        logger.info(f"Entity index built: {len(self.entity_index)} entities")

    def benchmark_retriever(self, retriever_class, name: str, test_cases: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
        """Benchmark a specific retriever implementation"""
        config = {
            'use_leann': False,  # Disable LEANN for pure performance comparison
            'retrieval_fusion': True,
            'use_leann_summaries': False,
        }

        # Create retriever instance
        if name == "Optimized":
            retriever = retriever_class(self.store, self.entity_index, config)
            retriever.edge_meta = self.edge_meta
        else:
            retriever = retriever_class(self.store, self.entity_index, config)
            retriever.edge_meta = self.edge_meta

        # Warm-up run
        for query, entities in test_cases[:2]:
            retriever.retrieve_context(query, entities, turn_id=0)

        # Benchmark runs
        timings = defaultdict(list)
        results_count = []

        num_runs = 10  # Number of runs per test case

        for run in range(num_runs):
            for query, entities in test_cases:
                start = time.perf_counter()
                result = retriever.retrieve_context(query, entities, turn_id=run)
                elapsed_ms = (time.perf_counter() - start) * 1000

                timings[query].append(elapsed_ms)
                results_count.append(len(result.bullets))

        # Calculate statistics
        stats = {}
        for query, times in timings.items():
            stats[query] = {
                'mean_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
                'min_ms': min(times),
                'max_ms': max(times),
            }

        overall_times = [t for times in timings.values() for t in times]
        stats['overall'] = {
            'mean_ms': statistics.mean(overall_times),
            'median_ms': statistics.median(overall_times),
            'stdev_ms': statistics.stdev(overall_times) if len(overall_times) > 1 else 0,
            'min_ms': min(overall_times),
            'max_ms': max(overall_times),
            'avg_bullets': statistics.mean(results_count),
        }

        return stats

    def run_comparison(self):
        """Run the benchmark comparison"""
        test_cases = [
            ("Where do I live?", ["you"]),
            ("Tell me about my work", ["you", "Anthropic"]),
            ("What's my dog's name?", ["you", "dog"]),
            ("Where was I born?", ["you"]),
            ("What did I study?", ["you", "MIT"]),
            ("Who do I work with at Anthropic?", ["you", "Anthropic", "Dario"]),
        ]

        logger.info("=" * 80)
        logger.info("MEMORY RETRIEVAL OPTIMIZATION BENCHMARK")
        logger.info("=" * 80)
        logger.info(f"Test cases: {len(test_cases)}")
        logger.info(f"Runs per case: 10")
        logger.info("")

        # Benchmark original
        logger.info("Benchmarking ORIGINAL implementation...")
        original_stats = self.benchmark_retriever(MemoryRetriever, "Original", test_cases)

        # Benchmark optimized
        logger.info("Benchmarking OPTIMIZED implementation...")
        optimized_stats = self.benchmark_retriever(MemoryRetrieverOptimized, "Optimized", test_cases)

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)

        # Per-query comparison
        logger.info("\nPer-Query Performance (median times):")
        logger.info("-" * 60)

        for query in test_cases:
            q_text = query[0]
            orig = original_stats[q_text]['median_ms']
            opt = optimized_stats[q_text]['median_ms']
            improvement = ((orig - opt) / orig) * 100 if orig > 0 else 0

            logger.info(f"\n'{q_text[:40]}...'")
            logger.info(f"  Original:  {orig:8.2f}ms")
            logger.info(f"  Optimized: {opt:8.2f}ms")
            logger.info(f"  Speedup:   {improvement:+6.1f}% {'🚀' if improvement > 30 else '✓' if improvement > 0 else '❌'}")

        # Overall comparison
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL PERFORMANCE SUMMARY")
        logger.info("=" * 60)

        orig_overall = original_stats['overall']
        opt_overall = optimized_stats['overall']

        mean_improvement = ((orig_overall['mean_ms'] - opt_overall['mean_ms']) / orig_overall['mean_ms']) * 100
        median_improvement = ((orig_overall['median_ms'] - opt_overall['median_ms']) / orig_overall['median_ms']) * 100

        logger.info(f"\nOriginal Implementation:")
        logger.info(f"  Mean:    {orig_overall['mean_ms']:.2f}ms ± {orig_overall['stdev_ms']:.2f}ms")
        logger.info(f"  Median:  {orig_overall['median_ms']:.2f}ms")
        logger.info(f"  Range:   {orig_overall['min_ms']:.2f}ms - {orig_overall['max_ms']:.2f}ms")
        logger.info(f"  Bullets: {orig_overall['avg_bullets']:.1f} avg")

        logger.info(f"\nOptimized Implementation:")
        logger.info(f"  Mean:    {opt_overall['mean_ms']:.2f}ms ± {opt_overall['stdev_ms']:.2f}ms")
        logger.info(f"  Median:  {opt_overall['median_ms']:.2f}ms")
        logger.info(f"  Range:   {opt_overall['min_ms']:.2f}ms - {opt_overall['max_ms']:.2f}ms")
        logger.info(f"  Bullets: {opt_overall['avg_bullets']:.1f} avg")

        logger.info(f"\n🎯 Performance Improvement:")
        logger.info(f"  Mean speedup:   {mean_improvement:+.1f}%")
        logger.info(f"  Median speedup: {median_improvement:+.1f}%")

        if median_improvement > 50:
            logger.info("\n🚀 EXCELLENT: >50% improvement achieved!")
        elif median_improvement > 30:
            logger.info("\n✨ GREAT: >30% improvement achieved!")
        elif median_improvement > 10:
            logger.info("\n✓ GOOD: >10% improvement achieved!")
        else:
            logger.info("\n⚠️ MINIMAL: <10% improvement, more optimization needed")

        # Identify remaining bottlenecks
        if opt_overall['median_ms'] > 100:
            logger.info("\n" + "=" * 60)
            logger.info("NEXT OPTIMIZATION TARGETS")
            logger.info("=" * 60)

            if opt_overall['median_ms'] > 500:
                logger.info("❗ Still >500ms median - Major optimizations needed:")
                logger.info("  1. Implement proper caching layer for entity scoring")
                logger.info("  2. Move to graph database (Neo4j) for complex traversals")
                logger.info("  3. Use vector embeddings instead of lexical matching")
            elif opt_overall['median_ms'] > 200:
                logger.info("⚠️ Still >200ms median - Consider:")
                logger.info("  1. Pre-compute common query patterns")
                logger.info("  2. Implement async/parallel scoring")
                logger.info("  3. Add result caching with TTL")
            else:
                logger.info("✓ Under 200ms - Further optimizations:")
                logger.info("  1. Fine-tune early termination thresholds")
                logger.info("  2. Optimize FTS queries")
                logger.info("  3. Profile remaining hot spots")

        return original_stats, optimized_stats


def main():
    """Main entry point"""
    logger.info("Starting retrieval optimization benchmark...")

    benchmark = RetrievalBenchmark()
    original_stats, optimized_stats = benchmark.run_comparison()

    logger.info("\n✅ Benchmark complete!")
    logger.info("\nRecommended next steps based on results:")
    logger.info("1. If improvement <30%: Focus on caching and pre-computation")
    logger.info("2. If improvement 30-50%: Consider architectural changes")
    logger.info("3. If improvement >50%: Quick wins successful, proceed to medium-term fixes")


if __name__ == "__main__":
    main()