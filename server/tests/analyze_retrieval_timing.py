#!/usr/bin/env python3
"""
Analyze real retrieval timings by adding detailed instrumentation to MemoryRetriever.
This will patch the retriever to log detailed timing breakdowns.
"""

import os
import time
from typing import Dict, List, Any
from loguru import logger

# Monkey-patch the MemoryRetriever to add detailed timing instrumentation
def instrument_retriever():
    """Patch MemoryRetriever.retrieve_context to add detailed timing"""

    from components.retrieval.memory_retriever import MemoryRetriever

    # Store original method
    original_retrieve = MemoryRetriever.retrieve_context

    def instrumented_retrieve_context(self, query: str, entities: List[str], turn_id: int, intent=None):
        """Instrumented version with detailed timing breakdown"""

        timings = {}
        total_start = time.perf_counter()

        logger.warning("=" * 80)
        logger.warning(f"RETRIEVAL TIMING ANALYSIS START")
        logger.warning(f"Query: '{query[:50]}...' | Entities: {entities}")
        logger.warning("=" * 80)

        # Step 1: Entity expansion
        expand_start = time.perf_counter()
        expanded_entities = self._expand_query_entities(entities, query)
        timings['entity_expansion_ms'] = (time.perf_counter() - expand_start) * 1000
        logger.warning(f"1. Entity Expansion: {timings['entity_expansion_ms']:.2f}ms | {len(entities)} → {len(expanded_entities)} entities")

        # Break down entity expansion further
        if expanded_entities:
            # Check multi-hop separately
            hop_start = time.perf_counter()
            base_set = set(entities)
            multi_hop = self._multi_hop_expansion(base_set, query)
            timings['multi_hop_ms'] = (time.perf_counter() - hop_start) * 1000
            logger.warning(f"   - Multi-hop expansion: {timings['multi_hop_ms']:.2f}ms | Found {len(multi_hop)} entities")

        # Step 2: Gather candidates
        gather_start = time.perf_counter()

        # 2a: Entity-based retrieval
        entity_start = time.perf_counter()
        entity_candidates = []
        now_ms = int(time.time() * 1000)
        recency_T_ms = 7 * 24 * 60 * 60 * 1000

        if self.graph_enabled:
            for i, entity in enumerate(expanded_entities):
                single_start = time.perf_counter()
                if entity in self.entity_index:
                    cands = self._score_entity_triples(entity, query, now_ms, recency_T_ms)
                    entity_candidates.extend(cands)
                single_time = (time.perf_counter() - single_start) * 1000
                if single_time > 50:  # Log slow entity lookups
                    logger.warning(f"   - Entity '{entity}': {single_time:.2f}ms for {len(self.entity_index.get(entity, []))} triples")

        timings['entity_scoring_ms'] = (time.perf_counter() - entity_start) * 1000
        logger.warning(f"2a. Entity Scoring: {timings['entity_scoring_ms']:.2f}ms | {len(entity_candidates)} candidates")

        # 2b: LEANN enhancement
        leann_start = time.perf_counter()
        leann_candidates = []
        if self.use_leann and self.retrieval_fusion:
            leann_candidates = self._retrieve_with_leann_enhancement(query, expanded_entities)
        timings['leann_ms'] = (time.perf_counter() - leann_start) * 1000
        logger.warning(f"2b. LEANN Enhancement: {timings['leann_ms']:.2f}ms | {len(leann_candidates)} candidates")

        # 2c: FTS search
        fts_start = time.perf_counter()
        fts_candidates = []
        if self.retrieval_fusion and query:
            fts_candidates = self._search_fts_summaries(query)
        timings['fts_ms'] = (time.perf_counter() - fts_start) * 1000
        logger.warning(f"2c. FTS Search: {timings['fts_ms']:.2f}ms | {len(fts_candidates)} candidates")

        all_candidates = entity_candidates + leann_candidates + fts_candidates
        timings['gather_total_ms'] = (time.perf_counter() - gather_start) * 1000
        logger.warning(f"2. Total Gathering: {timings['gather_total_ms']:.2f}ms | {len(all_candidates)} total candidates")

        # Step 3: MMR selection
        mmr_start = time.perf_counter()
        bullets = self._apply_mmr_selection(query, all_candidates, turn_id)
        timings['mmr_ms'] = (time.perf_counter() - mmr_start) * 1000
        logger.warning(f"3. MMR Selection: {timings['mmr_ms']:.2f}ms | {len(bullets)} bullets selected")

        # Total time
        timings['total_ms'] = (time.perf_counter() - total_start) * 1000

        # Calculate overhead
        component_sum = sum(v for k, v in timings.items() if k.endswith('_ms') and k != 'total_ms' and k != 'gather_total_ms')
        overhead = timings['total_ms'] - component_sum

        logger.warning("=" * 40)
        logger.warning(f"TOTAL: {timings['total_ms']:.2f}ms")
        logger.warning(f"Components: {component_sum:.2f}ms")
        logger.warning(f"Overhead: {overhead:.2f}ms ({overhead/timings['total_ms']*100:.1f}%)")

        # Identify bottleneck
        bottleneck = max(timings.items(), key=lambda x: x[1] if x[0].endswith('_ms') else 0)
        logger.warning(f"🔴 BOTTLENECK: {bottleneck[0]} = {bottleneck[1]:.2f}ms")
        logger.warning("=" * 80)

        # Call original method
        return original_retrieve(self, query, entities, turn_id, intent)

    # Replace method
    MemoryRetriever.retrieve_context = instrumented_retrieve_context
    logger.info("✅ MemoryRetriever instrumented with detailed timing analysis")

def main():
    """Run with instrumentation enabled"""
    import sys
    import subprocess

    # Enable instrumentation
    instrument_retriever()

    # Now run the bot with instrumentation
    logger.info("Starting bot with retrieval timing instrumentation...")
    logger.info("Connect to the client and speak to trigger retrieval...")
    logger.info("Watch for '🔴 BOTTLENECK' messages in the output")

    # Run bot.py directly
    subprocess.run([sys.executable, "bot.py"])

if __name__ == "__main__":
    main()