#!/usr/bin/env python3
"""Test that retrieval works correctly WITHOUT intent (backward compatibility)"""

import os
import sys
import tempfile

temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MEMORY_SQLITE_PATH"] = temp_db.name
os.environ["MEMORY_SOURCES"] = "graph,convo,summary"
os.environ["MEMORY_BULLETS_MAX"] = "5"
os.environ["MEMORY_CONVO_INDEX"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
import time

print("="*70)
print("TEST: Retrieval WITHOUT Intent (Backward Compatibility)")
print("="*70)

paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
store = MemoryStore(paths)
hot = HotMemory(store, max_recency=50)
hot.prewarm("en")

# Populate data
print("\n📊 Populating data...")
facts = ["My name is Alex", "I live in Seattle", "I work at Microsoft"]
for i, fact in enumerate(facts, 1):
    hot.process_turn(fact, "test", i)

# Add FTS conversation data
now_ts = int(time.time() * 1000)
convos = [
    "Yesterday I went hiking in the mountains",
    "I really enjoyed that Italian restaurant"
]
for i, text in enumerate(convos, 4):
    store.sql.execute("INSERT INTO chunks_fts(text, eid, ts) VALUES (?, ?, ?)",
                      (text, "conversation", now_ts + i))
store.sql.commit()

# Add summary data
summaries = ["User is a software engineer who enjoys outdoor activities"]
for i, text in enumerate(summaries, 6):
    store.sql.execute("INSERT INTO chunks_fts(text, eid, ts) VALUES (?, ?, ?)",
                      (text, "summary", now_ts + i))
store.sql.commit()

print(f"  ✓ Graph: 3 facts")
print(f"  ✓ Convo: 2 entries")
print(f"  ✓ Summary: 1 entry")

# Test 1: General query with NO intent
print("\n" + "="*70)
print("TEST 1: General Query (no intent = default graph-first)")
print("="*70)

query = "What do you know about me?"
print(f"Query: '{query}'")
print(f"Intent: None (backward compatibility mode)")

bullets = hot.retrieve_bullets(query, read_only=True, intent=None)

print(f"\nRetrieved {len(bullets)} bullets:")
sources = {"graph": 0, "convo": 0, "summary": 0}
for bullet in bullets:
    print(f"  {bullet}")
    if "[graph]" in bullet:
        sources["graph"] += 1
    elif "[convo]" in bullet:
        sources["convo"] += 1
    elif "[summary]" in bullet:
        sources["summary"] += 1

print(f"\nSource distribution: {sources}")

# Without intent, should default to graph-first but still allow other sources
if sources["graph"] > 0:
    print("✅ PASS: Graph results present (default behavior)")
    test1 = True
else:
    print("❌ FAIL: Graph should be present by default")
    test1 = False

# Test 2: FTS query with NO intent
print("\n" + "="*70)
print("TEST 2: FTS Query (no intent = should still boost convo matches)")
print("="*70)

query = "Italian restaurant"
print(f"Query: '{query}'")
print(f"Intent: None")

bullets = hot.retrieve_bullets(query, read_only=True, intent=None)

print(f"\nRetrieved {len(bullets)} bullets:")
sources = {"graph": 0, "convo": 0, "summary": 0}
for bullet in bullets:
    print(f"  {bullet}")
    if "[graph]" in bullet:
        sources["graph"] += 1
    elif "[convo]" in bullet:
        sources["convo"] += 1  
    elif "[summary]" in bullet:
        sources["summary"] += 1

print(f"\nSource distribution: {sources}")

# FTS match should still get 1.1x boost even without intent
if sources["convo"] > 0:
    print("✅ PASS: FTS match boosted even without intent")
    test2 = True
else:
    print("❌ FAIL: FTS matches should still be boosted")
    test2 = False

# Test 3: Check that it doesn't break
print("\n" + "="*70)
print("TEST 3: No Crashes (backward compatibility)")
print("="*70)

try:
    queries = [
        "What's my name?",
        "Tell me something",
        "Random query here"
    ]
    
    for q in queries:
        bullets = hot.retrieve_bullets(q, read_only=True, intent=None)
        print(f"  ✓ '{q}' → {len(bullets)} bullets")
    
    print("\n✅ PASS: No crashes, all queries work")
    test3 = True
except Exception as e:
    print(f"\n❌ FAIL: Crashed with {e}")
    test3 = False

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

results = {
    "Default graph-first behavior": test1,
    "FTS boost without intent": test2,
    "No crashes (backward compat)": test3
}

for name, passed in results.items():
    status = "✅" if passed else "❌"
    print(f"{status} {name}")

passed = sum(results.values())
total = len(results)
print(f"\nTotal: {passed}/{total} tests passed")

os.unlink(temp_db.name)
sys.exit(0 if passed == total else 1)
