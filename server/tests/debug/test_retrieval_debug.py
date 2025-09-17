#!/usr/bin/env python3
"""Debug retrieval system to see why dog facts aren't being retrieved"""

import os
import sys
import sqlite3
from pathlib import Path

# Add server to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Set correct database paths
os.environ["HOTMEM_SQLITE"] = "../data/memory.db"
os.environ["HOTMEM_LMDB_DIR"] = "../data/graph.lmdb"
os.environ["USER_ID"] = "peppi"

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade

# Check database directly
db_path = Path(server_dir).parent / "data" / "memory.db"
print(f"Database path: {db_path}")
print(f"Database exists: {db_path.exists()}")

# Check edges
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
edge_count = cur.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
print(f"Total edges in DB: {edge_count}")

# Check dog facts
dog_edges = cur.execute("""
    SELECT src, rel, dst FROM edge
    WHERE src='you' AND dst LIKE '%dog%'
    OR src='you' AND rel='has'
    OR dst LIKE '%potola%'
    OR dst LIKE '%milo%'
    LIMIT 10
""").fetchall()
print(f"\nDog-related edges: {dog_edges}")
conn.close()

# Now test with HotMemoryFacade
print("\n=== Testing HotMemoryFacade ===")
paths = Paths(
    sqlite_path="../data/memory.db",
    lmdb_dir="../data/graph.lmdb"
)
store = MemoryStore(paths)
facade = HotMemoryFacade(store)

# Rebuild indices from store
print("Rebuilding indices from store...")
facade.rebuild_from_store()

# Check entity index
print(f"Entity index size: {len(facade.entity_index)}")
print(f"'you' facts count: {len(facade.entity_index.get('you', []))}")
print(f"'dog' facts count: {len(facade.entity_index.get('dog', []))}")
print(f"'potola' facts count: {len(facade.entity_index.get('potola', []))}")

# Sample facts
if 'you' in facade.entity_index:
    you_facts = list(facade.entity_index['you'])[:5]
    print(f"\nSample 'you' facts: {you_facts}")

if 'dog' in facade.entity_index:
    dog_facts = list(facade.entity_index['dog'])[:5]
    print(f"Sample 'dog' facts: {dog_facts}")

# Test retrieval
print("\n=== Testing Retrieval ===")
query = "What do you know about my dog?"
entities = facade._extract_entities_light(query)
print(f"Extracted entities: {entities}")

# Test expanded entities
expanded = facade.retriever._expand_query_entities_optimized(entities, query)
print(f"Expanded entities: {expanded}")

# Test retrieval
result = facade.retriever.retrieve_context(query, entities, turn_id=1)
print(f"\nRetrieved bullets: {result.bullets}")
print(f"Retrieval stats: {result.retrieval_stats}")

# Also test with explicit entities
print("\n=== Testing with explicit entities ===")
explicit_entities = ["you", "dog", "potola"]
result2 = facade.retriever.retrieve_context(query, explicit_entities, turn_id=1)
print(f"Retrieved bullets: {result2.bullets}")