#!/usr/bin/env python3
"""Test that retrieval works correctly WITHOUT intent (backward compatibility).

Converted to proper pytest style (no sys.exit at import/collection time).
"""

import os
import tempfile
import time

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory


def test_no_intent_retrieval_behavior():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    os.environ["MEMORY_SQLITE_PATH"] = temp_db.name
    os.environ["MEMORY_SOURCES"] = "graph,convo,summary"
    os.environ["MEMORY_BULLETS_MAX"] = "5"
    os.environ["MEMORY_CONVO_INDEX"] = "true"

    paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store, max_recency=50)
    hot.prewarm("en")

    # Populate data
    facts = ["My name is Alex", "I live in Seattle", "I work at Microsoft"]
    for i, fact in enumerate(facts, 1):
        hot.process_turn(fact, "test", i)

    # Add FTS conversation data
    now_ts = int(time.time() * 1000)
    convos = [
        "Yesterday I went hiking in the mountains",
        "I really enjoyed that Italian restaurant",
    ]
    for i, text in enumerate(convos, 4):
        store.sql.execute(
            "INSERT INTO chunks_fts(text, eid, ts) VALUES (?, ?, ?)",
            (text, "conversation", now_ts + i),
        )
    store.sql.commit()

    # Add summary data
    summaries = [
        "User is a software engineer who enjoys outdoor activities",
    ]
    for i, text in enumerate(summaries, 6):
        store.sql.execute(
            "INSERT INTO chunks_fts(text, eid, ts) VALUES (?, ?, ?)",
            (text, "summary", now_ts + i),
        )
    store.sql.commit()

    # Test 1: default graph-first without intent
    bullets = hot.retrieve_bullets("What do you know about me?", read_only=True, intent=None)
    sources = {"graph": 0, "convo": 0, "summary": 0}
    for b in bullets:
        if "[graph]" in b:
            sources["graph"] += 1
        elif "[convo]" in b:
            sources["convo"] += 1
        elif "[summary]" in b:
            sources["summary"] += 1
    assert sources["graph"] > 0, "Graph should be present by default without intent"

    # Test 2: FTS boost without intent
    bullets = hot.retrieve_bullets("Italian restaurant", read_only=True, intent=None)
    sources = {"graph": 0, "convo": 0, "summary": 0}
    for b in bullets:
        if "[graph]" in b:
            sources["graph"] += 1
        elif "[convo]" in b:
            sources["convo"] += 1
        elif "[summary]" in b:
            sources["summary"] += 1
    assert sources["convo"] > 0, "FTS matches should be boosted even without intent"

    # Test 3: Sanity no crashes
    for q in ("What's my name?", "Tell me something", "Random query here"):
        bullets = hot.retrieve_bullets(q, read_only=True, intent=None)
        assert isinstance(bullets, list)

    # Cleanup temp DB
    try:
        os.unlink(temp_db.name)
    except Exception:
        pass
