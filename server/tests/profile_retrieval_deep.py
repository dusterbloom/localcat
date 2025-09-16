#!/usr/bin/env python3
"""
Deep profiling of memory retrieval to understand the 1.5s bottleneck.
Breaks down retrieval into component operations with detailed timing.
"""

import os
import sys
import time
import json
import sqlite3
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import cProfile
import pstats
import io

# Add server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from loguru import logger
from components.retrieval.memory_retriever import MemoryRetriever
from components.memory.memory_store import MemoryStore
from components.memory.config import MemoryConfig

# Set environment for detailed debugging
os.environ['HOTMEM_RETRIEVAL_DEBUG'] = '1'
os.environ['LOGURU_LEVEL'] = 'DEBUG'

@dataclass
class TimingResult:
    """Result of a timed operation"""
    name: str
    duration_ms: float
    details: Dict[str, Any]


class RetrievalProfiler:
    """Deep profiler for memory retrieval operations"""

    def __init__(self, db_path: str = None):
        # Use test database or create temporary one
        self.db_path = db_path or "/tmp/hotmem_profile.db"
        self.setup_test_data()

    def setup_test_data(self):
        """Set up test database with realistic data"""
        logger.info(f"Setting up test data at {self.db_path}")

        # Initialize memory store
        config = MemoryConfig()
        config.db_path = self.db_path
        self.store = MemoryStore(config)
        self.store.initialize()

        # Create realistic test data
        now_ts = int(time.time() * 1000)

        # Add edges with varying recency
        test_edges = [
            ("you", "lives_in", "San Francisco", 0.9, now_ts - 1000),
            ("you", "works_at", "Anthropic", 0.85, now_ts - 5000),
            ("you", "name", "Alex", 0.95, now_ts - 10000),
            ("San Francisco", "is", "city", 0.7, now_ts - 20000),
            ("Anthropic", "is", "AI company", 0.8, now_ts - 30000),
            ("you", "interested_in", "AI", 0.75, now_ts - 40000),
            ("you", "has", "dog", 0.8, now_ts - 50000),
            ("dog", "name", "Max", 0.9, now_ts - 60000),
            ("you", "likes", "coffee", 0.7, now_ts - 70000),
            ("coffee", "from", "Starbucks", 0.6, now_ts - 80000),
        ]

        # Add more edges for realistic graph size
        for i in range(100):
            s = f"entity_{i % 10}"
            r = ["knows", "likes", "has", "works_with", "visited"][i % 5]
            d = f"object_{i}"
            conf = 0.5 + (i % 5) * 0.1
            ts = now_ts - (i * 1000)
            test_edges.append((s, r, d, conf, ts))

        # Store edges
        for s, r, d, conf, ts in test_edges:
            self.store.observe_edge(s, r, d, conf, ts)
            # Add to FTS
            self.store.enqueue_mention(s, f"{s} {r} {d}", ts, "test_session", 0)
            self.store.enqueue_mention(d, f"{s} {r} {d}", ts, "test_session", 0)

        # Add summaries to FTS
        summaries = [
            ("You mentioned living in San Francisco and working at Anthropic", now_ts - 1000),
            ("You have a dog named Max who likes to play fetch", now_ts - 50000),
            ("You enjoy drinking coffee from Starbucks in the morning", now_ts - 70000),
        ]

        for text, ts in summaries:
            self.store.enqueue_mention(f"summary:{ts}", text, ts, "test_session", 0)

        self.store.flush()
        logger.info(f"Test data setup complete: {len(test_edges)} edges")

        # Build entity index
        self.entity_index = defaultdict(set)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT subject, relation, destination FROM edges")
        for s, r, d in cursor.fetchall():
            self.entity_index[s].add((s, r, d))
            self.entity_index[d].add((s, r, d))
        conn.close()

        logger.info(f"Entity index built: {len(self.entity_index)} entities")

    def profile_retrieval_components(self, query: str, entities: List[str]) -> Dict[str, Any]:
        """Profile individual retrieval components"""
        results = {}

        # Initialize retriever
        config = {
            'use_leann': False,  # Disable LEANN for now to isolate other bottlenecks
            'retrieval_fusion': True,
            'use_leann_summaries': False,
        }

        retriever = MemoryRetriever(self.store, self.entity_index, config)

        # 1. Profile entity expansion
        start = time.perf_counter()
        expanded_entities = retriever._expand_query_entities(entities, query)
        results['entity_expansion_ms'] = (time.perf_counter() - start) * 1000
        results['expanded_count'] = len(expanded_entities)

        # 2. Profile multi-hop expansion separately
        start = time.perf_counter()
        base_set = set(entities)
        multi_hop = retriever._multi_hop_expansion(base_set, query)
        results['multi_hop_ms'] = (time.perf_counter() - start) * 1000
        results['multi_hop_count'] = len(multi_hop)

        # 3. Profile entity triple scoring
        start = time.perf_counter()
        now_ms = int(time.time() * 1000)
        recency_T_ms = 7 * 24 * 60 * 60 * 1000
        entity_candidates = []
        for entity in expanded_entities[:3]:  # Sample first 3
            entity_start = time.perf_counter()
            if entity in self.entity_index:
                candidates = retriever._score_entity_triples(entity, query, now_ms, recency_T_ms)
                entity_candidates.extend(candidates)
            entity_time = (time.perf_counter() - entity_start) * 1000
            results[f'entity_scoring_{entity}_ms'] = entity_time

        results['entity_scoring_total_ms'] = (time.perf_counter() - start) * 1000
        results['entity_candidates'] = len(entity_candidates)

        # 4. Profile FTS search
        start = time.perf_counter()
        fts_results = retriever._search_fts_summaries(query)
        results['fts_search_ms'] = (time.perf_counter() - start) * 1000
        results['fts_results'] = len(fts_results)

        # 5. Profile MMR selection
        all_candidates = entity_candidates + fts_results
        start = time.perf_counter()
        bullets = retriever._apply_mmr_selection(query, all_candidates, turn_id=1)
        results['mmr_selection_ms'] = (time.perf_counter() - start) * 1000
        results['bullets_selected'] = len(bullets)

        # 6. Profile complete retrieval
        start = time.perf_counter()
        result = retriever.retrieve_context(query, entities, turn_id=1)
        results['total_retrieval_ms'] = (time.perf_counter() - start) * 1000

        return results

    def profile_database_operations(self) -> Dict[str, Any]:
        """Profile raw database operations"""
        results = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Profile edge count query
        start = time.perf_counter()
        cursor.execute("SELECT COUNT(*) FROM edges")
        count = cursor.fetchone()[0]
        results['edge_count_query_ms'] = (time.perf_counter() - start) * 1000
        results['total_edges'] = count

        # 2. Profile entity lookup
        start = time.perf_counter()
        cursor.execute("SELECT * FROM edges WHERE subject = ? OR destination = ?", ("you", "you"))
        rows = cursor.fetchall()
        results['entity_lookup_ms'] = (time.perf_counter() - start) * 1000
        results['entity_rows'] = len(rows)

        # 3. Profile FTS search
        start = time.perf_counter()
        cursor.execute("""
            SELECT eid, text, ts
            FROM mentions_fts
            WHERE text MATCH ?
            ORDER BY ts DESC
            LIMIT 10
        """, ("lives OR San Francisco*",))
        fts_rows = cursor.fetchall()
        results['fts_query_ms'] = (time.perf_counter() - start) * 1000
        results['fts_matches'] = len(fts_rows)

        # 4. Profile index usage
        start = time.perf_counter()
        cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM edges WHERE subject = ?", ("you",))
        plan = cursor.fetchall()
        results['explain_plan_ms'] = (time.perf_counter() - start) * 1000
        results['query_plan'] = str(plan)

        conn.close()
        return results

    def run_full_profile(self) -> Dict[str, Any]:
        """Run complete profiling suite"""
        logger.info("=" * 80)
        logger.info("MEMORY RETRIEVAL DEEP PROFILING")
        logger.info("=" * 80)

        # Test queries
        test_cases = [
            ("Where do I live?", ["you"]),
            ("Tell me about my work", ["you", "Anthropic"]),
            ("What's my dog's name?", ["you", "dog"]),
        ]

        all_results = {}

        # Profile database operations first
        logger.info("\n1. DATABASE OPERATIONS PROFILE")
        logger.info("-" * 40)
        db_results = self.profile_database_operations()
        for key, value in db_results.items():
            if key.endswith('_ms'):
                logger.info(f"  {key}: {value:.2f}ms")
            else:
                logger.info(f"  {key}: {value}")
        all_results['database'] = db_results

        # Profile each test case
        for i, (query, entities) in enumerate(test_cases, 1):
            logger.info(f"\n{i+1}. RETRIEVAL PROFILE: '{query}'")
            logger.info("-" * 40)

            results = self.profile_retrieval_components(query, entities)

            # Display timing breakdown
            logger.info("  Component Timings:")
            total_component_time = 0
            for key, value in results.items():
                if key.endswith('_ms'):
                    logger.info(f"    {key}: {value:.2f}ms")
                    total_component_time += value

            logger.info(f"\n  Total component time: {results.get('total_retrieval_ms', 0):.2f}ms")

            # Display counts
            logger.info("\n  Operation Counts:")
            for key, value in results.items():
                if not key.endswith('_ms'):
                    logger.info(f"    {key}: {value}")

            all_results[f'query_{i}'] = results

        # Run cProfile for detailed function-level profiling
        logger.info("\n" + "=" * 40)
        logger.info("DETAILED FUNCTION PROFILING")
        logger.info("=" * 40)

        config = {
            'use_leann': False,
            'retrieval_fusion': True,
            'use_leann_summaries': False,
        }
        retriever = MemoryRetriever(self.store, self.entity_index, config)

        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        # Run retrieval 10 times to get better statistics
        for _ in range(10):
            retriever.retrieve_context("Where do I live and work?", ["you"], turn_id=1)

        profiler.disable()

        # Get profile statistics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions

        profile_output = s.getvalue()
        logger.info(profile_output)

        # Identify bottlenecks
        logger.info("\n" + "=" * 40)
        logger.info("BOTTLENECK ANALYSIS")
        logger.info("=" * 40)

        bottlenecks = self.analyze_bottlenecks(all_results)
        for bottleneck in bottlenecks:
            logger.warning(f"⚠️  {bottleneck}")

        return all_results

    def analyze_bottlenecks(self, results: Dict[str, Any]) -> List[str]:
        """Analyze results to identify bottlenecks"""
        bottlenecks = []

        # Check database operations
        db = results.get('database', {})
        if db.get('entity_lookup_ms', 0) > 100:
            bottlenecks.append(f"Entity lookup is slow: {db['entity_lookup_ms']:.2f}ms - needs index optimization")
        if db.get('fts_query_ms', 0) > 200:
            bottlenecks.append(f"FTS search is slow: {db['fts_query_ms']:.2f}ms - consider FTS index optimization")

        # Check retrieval components
        for query_key in [k for k in results.keys() if k.startswith('query_')]:
            query_results = results[query_key]

            if query_results.get('entity_expansion_ms', 0) > 50:
                bottlenecks.append(f"Entity expansion taking {query_results['entity_expansion_ms']:.2f}ms")

            if query_results.get('multi_hop_ms', 0) > 100:
                bottlenecks.append(f"Multi-hop expansion is slow: {query_results['multi_hop_ms']:.2f}ms")

            if query_results.get('entity_scoring_total_ms', 0) > 500:
                bottlenecks.append(f"Entity scoring is the main bottleneck: {query_results['entity_scoring_total_ms']:.2f}ms")

            if query_results.get('fts_search_ms', 0) > 300:
                bottlenecks.append(f"FTS search in retriever: {query_results['fts_search_ms']:.2f}ms")

            if query_results.get('mmr_selection_ms', 0) > 200:
                bottlenecks.append(f"MMR selection algorithm: {query_results['mmr_selection_ms']:.2f}ms")

        # Check if total time doesn't match component sum
        for query_key in [k for k in results.keys() if k.startswith('query_')]:
            query_results = results[query_key]
            total = query_results.get('total_retrieval_ms', 0)
            components = sum(v for k, v in query_results.items() if k.endswith('_ms') and k != 'total_retrieval_ms')
            if abs(total - components) > 100:
                bottlenecks.append(f"Hidden overhead: {abs(total - components):.2f}ms unaccounted for")

        return bottlenecks


def main():
    """Main entry point"""
    logger.info("Starting deep retrieval profiling...")

    profiler = RetrievalProfiler()
    results = profiler.run_full_profile()

    # Save results to JSON for analysis
    output_file = "/tmp/retrieval_profile_results.json"
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = json.dumps(results, indent=2, default=str)
        f.write(json_results)

    logger.info(f"\n✅ Profile results saved to: {output_file}")
    logger.info("\n🔍 Key Findings:")
    logger.info("  - Check component timings to identify the slowest operation")
    logger.info("  - Look for database query inefficiencies")
    logger.info("  - Review function-level profiling for hot spots")
    logger.info("  - Examine bottleneck analysis for optimization targets")


if __name__ == "__main__":
    main()