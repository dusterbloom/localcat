#!/usr/bin/env python3
"""
Direct test of MemoryRetriever timing with mock data
"""

import os
import time
from collections import defaultdict
from typing import Dict, Set
from loguru import logger

# Mock a simple store for testing
class MockStore:
    def search_fts_detailed(self, query, limit=10):
        # Simulate FTS search with some delay
        time.sleep(0.05)  # 50ms simulated database query
        return [
            ("You live in San Francisco", "summary:1", int(time.time() * 1000)),
            ("You work at Anthropic", "summary:2", int(time.time() * 1000) - 5000),
            ("You have a dog named Max", "summary:3", int(time.time() * 1000) - 10000),
        ]

def test_retriever_performance():
    """Test retriever with mock data to identify bottlenecks"""
    from components.retrieval.memory_retriever import MemoryRetriever

    # Create mock entity index with realistic data
    entity_index = defaultdict(set)

    # Add test triples
    base_triples = [
        ("you", "lives_in", "San Francisco"),
        ("you", "works_at", "Anthropic"),
        ("you", "name", "Alex"),
        ("you", "has", "dog"),
        ("dog", "name", "Max"),
        ("San Francisco", "is", "city"),
        ("Anthropic", "is", "AI company"),
    ]

    # Add many more triples to simulate real-world complexity
    for i in range(100):
        entity = f"entity_{i}"
        for j in range(10):
            s = entity if j % 2 == 0 else "you"
            r = ["knows", "likes", "visited", "owns", "created"][j % 5]
            d = f"object_{i}_{j}"
            triple = (s, r, d)

            entity_index[s].add(triple)
            entity_index[d].add(triple)

            # Add base triples
            if i == 0:
                for base_triple in base_triples:
                    s, r, d = base_triple
                    entity_index[s].add(base_triple)
                    entity_index[d].add(base_triple)

    logger.info(f"Created entity index with {len(entity_index)} entities")
    logger.info(f"Total triples: {sum(len(v) for v in entity_index.values())}")

    # Create retriever
    store = MockStore()
    config = {
        'use_leann': False,  # Disable LEANN to isolate other components
        'retrieval_fusion': True,
        'use_leann_summaries': False,
    }

    retriever = MemoryRetriever(store, entity_index, config)

    # Set edge metadata for scoring
    now_ts = int(time.time() * 1000)
    for entity_set in entity_index.values():
        for triple in entity_set:
            if isinstance(triple, tuple) and len(triple) >= 3:
                s, r, d = triple[:3]
                retriever.edge_meta[(s, r, d)] = {
                    'ts': now_ts - (hash(triple) % 100000),  # Random recency
                    'weight': 0.5 + (hash(triple) % 50) / 100  # Random weight
                }

    # Test queries
    test_cases = [
        ("Where do I live?", ["you"]),
        ("Tell me about my work", ["you", "Anthropic"]),
        ("What is my dog's name?", ["you", "dog"]),
    ]

    logger.info("=" * 80)
    logger.info("RETRIEVER PERFORMANCE TEST")
    logger.info("=" * 80)

    for query, entities in test_cases:
        logger.info(f"\n📝 Query: '{query}'")
        logger.info(f"   Entities: {entities}")

        # Run retrieval 3 times to get average
        times = []
        for run in range(3):
            start = time.perf_counter()
            result = retriever.retrieve_context(query, entities, turn_id=1)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if run == 0:  # Log details only for first run
                logger.info(f"   Bullets: {len(result.bullets)}")
                logger.info(f"   Candidates: {result.retrieval_stats.get('candidates', 0)}")
                logger.info(f"   Expanded entities: {result.retrieval_stats.get('expanded_entities', 0)}")

        avg_time = sum(times) / len(times)
        logger.info(f"   ⏱️  Average time: {avg_time:.0f}ms (runs: {[f'{t:.0f}ms' for t in times]})")

    logger.info("\n" + "=" * 80)
    logger.info("Check logs above for timing breakdowns")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_retriever_performance()