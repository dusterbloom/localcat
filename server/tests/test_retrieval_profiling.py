#!/usr/bin/env python3
"""
Deep profiling of memory retrieval operations to identify bottlenecks
"""

import os
import sys
import time
import tempfile
from contextlib import contextmanager

# Add server path and activate environment
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')


@contextmanager
def time_block(name: str):
    """Time a block of code with precise measurement"""
    start = time.perf_counter()
    print(f"⏱️ START {name}")
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * 1000
        print(f"✅ END {name}: {duration:.1f}ms")


def test_retrieval_bottlenecks():
    """Profile each step of memory retrieval to identify bottlenecks"""
    print("🔍 DEEP MEMORY RETRIEVAL PROFILING")
    print("=" * 60)

    from components.memory.hotmemory_facade import HotMemoryFacade
    from components.memory.memory_store import MemoryStore, Paths
    from components.session.session_store import SessionStore
    import tempfile

    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        with time_block("Initialize components"):
            paths = Paths(
                sqlite_path=os.path.join(temp_dir, "test_memory.db"),
                lmdb_dir=os.path.join(temp_dir, "test_graph.lmdb")
            )
            store = MemoryStore(paths)
            facade = HotMemoryFacade(store)

        # Pre-populate with some data to make retrieval meaningful
        with time_block("Pre-populate test data"):
            test_data = [
                "My favorite color is blue",
                "I work at TechCorp as a software engineer",
                "I live in San Francisco on Market Street",
                "My dog's name is Rex and he loves tennis balls",
                "I went to Stanford University for computer science"
            ]

            for i, text in enumerate(test_data, 1):
                facade.process_turn(text, f"setup_session_{i}", 1)

        print(f"\n🧪 Testing retrieval on populated database:")

        # Test retrieval with detailed timing
        test_query = "What is the weather like today?"

        with time_block("Full retrieval process"):
            # Get the retriever directly
            retriever = facade.retriever

            with time_block("Entity extraction"):
                # For this test, simulate entity extraction
                entities = ["What"]  # Simplified

            with time_block("Entity expansion"):
                expanded_entities = retriever._expand_query_entities(entities, test_query)
                print(f"    Entities: {entities} -> {expanded_entities}")

            with time_block("Multi-hop expansion"):
                base_entities = set(entities)
                expanded_set = retriever._multi_hop_expansion(base_entities, test_query)
                print(f"    Multi-hop: {base_entities} -> {expanded_set}")

            with time_block("Gather candidate triples"):
                candidates = retriever._gather_candidate_triples(test_query, expanded_entities)
                print(f"    Candidates found: {len(candidates)}")

            with time_block("Entity-based retrieval"):
                entity_candidates = []
                now_ms = int(time.time() * 1000)
                recency_T_ms = 7 * 24 * 60 * 60 * 1000  # 7 days

                for entity in expanded_entities:
                    if entity in retriever.entity_index:
                        with time_block(f"    Score entity '{entity}'"):
                            entity_triples = retriever._score_entity_triples(entity, test_query, now_ms, recency_T_ms)
                            entity_candidates.extend(entity_triples)
                            print(f"      Found {len(entity_triples)} triples for '{entity}'")

            with time_block("FTS summary search"):
                fts_results = retriever._search_fts_summaries(test_query)
                print(f"    FTS results: {len(fts_results)}")

            with time_block("MMR selection"):
                bullets = retriever._apply_mmr_selection(test_query, candidates, 1)
                print(f"    Selected bullets: {len(bullets)}")

        print(f"\n📊 ENTITY INDEX SIZE:")
        total_entities = len(retriever.entity_index)
        total_triples = sum(len(triples) for triples in retriever.entity_index.values())
        print(f"    Total entities: {total_entities}")
        print(f"    Total triples: {total_triples}")

        if total_entities > 0:
            print(f"    Average triples per entity: {total_triples/total_entities:.1f}")

        # Test with larger entity index if needed
        if total_entities == 0:
            print(f"\n⚠️ Entity index is empty - no graph data to retrieve from!")
            print(f"    This explains why retrieval is slow - system is doing expensive empty searches")


def test_memory_store_performance():
    """Test the underlying MemoryStore performance"""
    print(f"\n🗄️ MEMORY STORE PERFORMANCE TEST")
    print("=" * 40)

    from components.memory.memory_store import MemoryStore, Paths

    with tempfile.TemporaryDirectory() as temp_dir:
        with time_block("MemoryStore initialization"):
            paths = Paths(
                sqlite_path=os.path.join(temp_dir, "perf_test.db"),
                lmdb_dir=os.path.join(temp_dir, "perf_graph.lmdb")
            )
            store = MemoryStore(paths)

        with time_block("FTS search on empty store"):
            try:
                results = store.search_fts_detailed("weather", limit=10)
                print(f"    Empty FTS results: {len(results)}")
            except Exception as e:
                print(f"    FTS search failed: {e}")

        with time_block("Add test triples"):
            test_triples = [
                ("user", "likes", "blue"),
                ("user", "works_at", "TechCorp"),
                ("user", "lives_in", "San Francisco"),
                ("Rex", "is_pet_of", "user"),
                ("user", "studied_at", "Stanford")
            ]

            for s, r, d in test_triples:
                store.add_triple(s, r, d)

        with time_block("FTS search with data"):
            try:
                results = store.search_fts_detailed("user", limit=10)
                print(f"    FTS results with data: {len(results)}")
            except Exception as e:
                print(f"    FTS search failed: {e}")


if __name__ == "__main__":
    test_retrieval_bottlenecks()
    test_memory_store_performance()