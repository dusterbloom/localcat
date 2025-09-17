#!/usr/bin/env python3
"""Test that age facts are stored and retrieved correctly"""

import os
import sys
from pathlib import Path

# Setup
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))
os.environ["HOTMEM_SQLITE"] = "../data/memory.db"
os.environ["HOTMEM_LMDB_DIR"] = "../data/graph.lmdb"
os.environ["USER_ID"] = "peppi"

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade

# Initialize
paths = Paths(sqlite_path="../data/memory.db", lmdb_dir="../data/graph.lmdb")
store = MemoryStore(paths)
facade = HotMemoryFacade(store)

print("Testing age storage and retrieval...")

# 1. Store a fact about age
print("\n1. Storing fact: 'My dog Potola is 5 years old'")
result = facade.process_turn(
    text="My dog Potola is 5 years old",
    session_id="age_test",
    turn_id=1,
    user_id="peppi"
)
print(f"   Stored triples: {result.triples}")

# 2. Ask about the age
print("\n2. Asking: 'How old is my dog?'")
result = facade.process_turn(
    text="How old is my dog?",
    session_id="age_test",
    turn_id=2,
    user_id="peppi"
)
print(f"   Retrieved bullets: {result.bullets[:3] if result.bullets else 'None'}")

# 3. Ask about Potola specifically
print("\n3. Asking: 'How old is Potola?'")
result = facade.process_turn(
    text="How old is Potola?",
    session_id="age_test",
    turn_id=3,
    user_id="peppi"
)
print(f"   Retrieved bullets: {result.bullets[:3] if result.bullets else 'None'}")

print("\n✅ Test complete!")