#!/usr/bin/env python3
"""Test that questions now work properly without entity extraction"""

import os
import sys
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

print("Testing question retrieval without entity extraction...\n")

# Initialize system
paths = Paths(
    sqlite_path="../data/memory.db",
    lmdb_dir="../data/graph.lmdb"
)
store = MemoryStore(paths)
facade = HotMemoryFacade(store)

# Rebuild indices
facade.rebuild_from_store()
print(f"Loaded {len(facade.entity_index)} entities from database\n")

# Test questions
test_questions = [
    "What do you know about my dog?",
    "How old is Potola?",
    "What's my favorite number?",
    "Where do I live?",
    "Do you remember my dog's name?",
]

for question in test_questions:
    print(f"Q: {question}")

    # Process as a pure question
    result = facade.process_turn(
        text=question,
        session_id="test_session",
        turn_id=1,
        user_id="peppi"
    )

    print(f"   Intent: {result.intent.intent.value}")
    print(f"   Needs retrieval: {result.needs_retrieval}")
    print(f"   Needs storage: {result.needs_storage}")
    print(f"   Retrieved bullets: {len(result.bullets)}")

    if result.bullets:
        # Show first 3 bullets
        for bullet in result.bullets[:3]:
            if isinstance(bullet, tuple) and len(bullet) >= 3:
                s, r, d = bullet[:3]
                print(f"     - {s} {r} {d}")
            else:
                print(f"     - {bullet}")

    print()