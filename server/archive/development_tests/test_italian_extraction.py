#!/usr/bin/env python3
"""Test what happens with 'I enjoyed the Italian restaurant last night'"""

import os
import sys
import tempfile

temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MEMORY_SQLITE_PATH"] = temp_db.name

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory

print("="*70)
print("EXTRACTION TEST: 'I enjoyed the Italian restaurant last night'")
print("="*70)

paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
store = MemoryStore(paths)
hot = HotMemory(store, max_recency=50)
hot.prewarm("en")

# Test the exact phrase
text = "I enjoyed the Italian restaurant last night"
print(f"\nInput: '{text}'")

# Process and see what gets extracted
bullets, triples = hot.process_turn(text, "test-session", 1)

print(f"\n📊 Extraction Results:")
print(f"  Triples extracted: {len(triples)}")
for s, r, d in triples:
    print(f"    ({s}, {r}, {d})")

print(f"\n  Bullets retrieved: {len(bullets)}")
for bullet in bullets:
    print(f"    {bullet}")

print(f"\n🔍 Entity Index Contents:")
for entity, entity_triples in hot.entity_index.items():
    print(f"  '{entity}' → {entity_triples}")

print(f"\n🔍 Recency Buffer Contents:")
for item in hot.recency_buffer:
    print(f"  ({item.s}, {item.r}, {item.d})")

os.unlink(temp_db.name)
