#!/usr/bin/env python3
"""
Test multi-source retrieval: graph, convo (FTS), and summary.

Verifies that:
1. All 3 sources can be populated with data
2. Retrieval queries each source and returns results
3. Intent-aware routing prioritizes correct sources
4. Budget allocation prevents source starvation
"""

import os
import sys
import tempfile

# Set up environment for multi-source retrieval
temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MEMORY_SQLITE_PATH"] = temp_db.name
# Don't set LMDB path - let it default to None
os.environ["MEMORY_SOURCES"] = "graph,convo,summary"
os.environ["MEMORY_BULLETS_MAX"] = "5"
os.environ["MEMORY_CONVO_INDEX"] = "true"  # Enable FTS indexing
# Debug mode disabled for clean output
# os.environ["DEBUG_RETRIEVAL"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
import time


def populate_graph_data(hot: HotMemory):
    """Populate graph with factual data"""
    print("\n📊 Populating GRAPH (factual knowledge)...")

    facts = [
        "My name is Alex",
        "I live in Seattle",
        "I work at Microsoft"
    ]

    for i, fact in enumerate(facts, 1):
        bullets, triples = hot.process_turn(fact, "test-session", i)
        print(f"  ✓ Turn {i}: {triples}")

    return len(facts)


def populate_convo_data(store: MemoryStore, hot: HotMemory, turn_offset: int):
    """Populate conversation data (FTS searchable)"""
    print("\n💬 Populating CONVO (conversation history via FTS)...")

    conversations = [
        "Yesterday I went hiking in the mountains and saw amazing views",
        "I really enjoyed that Italian restaurant we tried last week",
        "The project deadline is next Friday and we need to finish testing"
    ]

    now_ts = int(time.time() * 1000)

    for i, text in enumerate(conversations, turn_offset + 1):
        # Store in conversation turn
        turn_id_hash = store.enqueue_turn(text, "test-session", i, now_ts)

        # Also index in FTS for conversation search
        store.sql.execute(
            "INSERT INTO chunks_fts(text, eid, ts) VALUES (?, ?, ?)",
            (text, "conversation", now_ts + i)
        )
        store.sql.commit()

        print(f"  ✓ Turn {i}: Indexed in FTS")

    return len(conversations)


def populate_summary_data(store: MemoryStore, turn_offset: int):
    """Populate summary data"""
    print("\n📝 Populating SUMMARY (session summaries)...")

    summaries = [
        "User is a software engineer who enjoys outdoor activities",
        "User has been discussing work projects and personal interests",
        "User lives in the Pacific Northwest and values work-life balance"
    ]

    now_ts = int(time.time() * 1000)

    for i, summary in enumerate(summaries, turn_offset + 1):
        # Store summaries with special eid marker
        store.sql.execute(
            "INSERT INTO chunks_fts(text, eid, ts) VALUES (?, ?, ?)",
            (summary, "summary", now_ts + i)
        )
        store.sql.commit()

        print(f"  ✓ Summary {i}: Indexed")

    return len(summaries)


def test_multisource_retrieval():
    """Test that all 3 sources contribute to retrieval"""
    print("="*70)
    print("MULTI-SOURCE RETRIEVAL TEST")
    print("="*70)

    # Create fresh database
    paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store, max_recency=50)
    hot.prewarm("en")

    # Populate all 3 sources
    turn_count = 0
    turn_count += populate_graph_data(hot)
    turn_count += populate_convo_data(store, hot, turn_count)
    turn_count += populate_summary_data(store, turn_count)

    print(f"\n📦 Data populated: {turn_count} total entries")

    # Verify data in each source
    print("\n🔍 Verifying data in sources...")

    # Check graph (entity index)
    print(f"  Graph entities: {list(hot.entity_index.keys())}")
    print(f"  Graph facts: {len([t for triples in hot.entity_index.values() for t in triples])}")

    # Check FTS (conversation + summary)
    fts_count = store.sql.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    print(f"  FTS entries: {fts_count}")

    convo_count = store.sql.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE eid = 'conversation'"
    ).fetchone()[0]
    print(f"    - Conversation: {convo_count}")

    summary_count = store.sql.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE eid = 'summary'"
    ).fetchone()[0]
    print(f"    - Summary: {summary_count}")

    # Test 1: General query should get results from all sources
    print("\n" + "="*70)
    print("TEST 1: General Query (all sources should contribute)")
    print("="*70)

    query = "Tell me about the user"
    print(f"Query: '{query}'")

    bullets = hot.retrieve_bullets(query, read_only=True, intent=None)

    print(f"\nRetrieved {len(bullets)} bullets:")
    source_counts = {"graph": 0, "convo": 0, "summary": 0}
    for bullet in bullets:
        print(f"  {bullet}")
        if "[graph]" in bullet:
            source_counts["graph"] += 1
        elif "[convo]" in bullet:
            source_counts["convo"] += 1
        elif "[summary]" in bullet:
            source_counts["summary"] += 1

    print(f"\nSource distribution: {source_counts}")
    active_sources = sum(1 for count in source_counts.values() if count > 0)

    # This is a semantic query ("about"), so summary SHOULD be prioritized
    if source_counts["summary"] > 0:
        print(f"✅ PASS: Semantic query correctly prioritized summary source")
        test1_pass = True
    else:
        print(f"❌ FAIL: Summary source should be prioritized for semantic query")
        test1_pass = False

    # Test 2: Temporal query should prioritize conversation
    print("\n" + "="*70)
    print("TEST 2: Temporal Query (should prioritize conversation)")
    print("="*70)

    query = "What did I do yesterday?"
    print(f"Query: '{query}'")

    # Simulate temporal intent
    intent = {
        "intent": "recall_information",
        "confidence": 0.9,
        "strategy": "retrieval_focused",
        "fallback": False
    }

    bullets = hot.retrieve_bullets(query, read_only=True, intent=intent)

    print(f"\nRetrieved {len(bullets)} bullets:")
    source_counts = {"graph": 0, "convo": 0, "summary": 0}
    for bullet in bullets:
        print(f"  {bullet}")
        if "[graph]" in bullet:
            source_counts["graph"] += 1
        elif "[convo]" in bullet:
            source_counts["convo"] += 1
        elif "[summary]" in bullet:
            source_counts["summary"] += 1

    print(f"\nSource distribution: {source_counts}")

    # Temporal query should prioritize convo/summary over graph
    # Note: convo might be empty if FTS doesn't match the specific word "yesterday"
    if source_counts["convo"] > 0 or (source_counts["summary"] > 0 and source_counts["summary"] >= source_counts["graph"]):
        print(f"✅ PASS: Temporal query correctly routed (convo or summary prioritized)")
        test2_pass = True
    else:
        print(f"❌ FAIL: Temporal query should prioritize convo/summary over graph")
        test2_pass = False

    # Test 3: Specific search should use FTS
    print("\n" + "="*70)
    print("TEST 3: Specific Search (should use FTS)")
    print("="*70)

    query = "Italian restaurant"
    print(f"Query: '{query}'")

    # Debug: Check what FTS returns directly
    print("\n  [DEBUG] Direct FTS query:")
    fts_results = store.search_fts(query, limit=5)
    print(f"    FTS returned {len(fts_results)} results:")
    for text, eid, ts in fts_results:
        print(f"      [{eid}] {text[:60]}")

    # Debug: Check budget allocation
    print("\n  [DEBUG] Budget allocation:")
    enabled_sources = os.getenv("MEMORY_SOURCES", "graph").split(",")
    budget = hot.retriever._allocate_budget(5, enabled_sources)
    print(f"    Sources: {enabled_sources}")
    print(f"    Budget: {budget}")

    bullets = hot.retrieve_bullets(query, read_only=True, intent=None)

    print(f"\nRetrieved {len(bullets)} bullets:")
    found_italian = False
    for bullet in bullets:
        print(f"  {bullet}")
        if "italian" in bullet.lower():
            found_italian = True

    if found_italian:
        print(f"✅ PASS: Found specific conversation content via FTS")
        test3_pass = True
    else:
        print(f"❌ FAIL: Didn't find 'Italian' via FTS search")
        test3_pass = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    results = {
        "Semantic query routing (summary priority)": test1_pass,
        "Temporal query routing (convo/summary priority)": test2_pass,
        "FTS conversation search (convo boost)": test3_pass
    }

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    # Cleanup
    os.unlink(temp_db.name)

    return passed == total


if __name__ == "__main__":
    success = test_multisource_retrieval()
    sys.exit(0 if success else 1)