#!/usr/bin/env python3
"""Test temporal extraction for age-related facts"""

import os
import sys
from pathlib import Path

# Add server to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Configure environment
os.environ["HOTMEM_SQLITE"] = "../data/memory.db"
os.environ["HOTMEM_LMDB_DIR"] = "../data/graph.lmdb"
os.environ["USER_ID"] = "peppi"

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade

print("Testing temporal extraction...\n")

# Initialize system
paths = Paths(
    sqlite_path="../data/memory.db",
    lmdb_dir="../data/graph.lmdb"
)
store = MemoryStore(paths)
facade = HotMemoryFacade(store)

# Test temporal extraction
test_phrases = [
    "My dog Potola is 5 years old",
    "Potola turned 5 last month",
    "I got Potola 5 years ago",
    "My dog is five years old"
]

for phrase in test_phrases:
    print(f"Input: '{phrase}'")

    # Process as fact statement
    result = facade.process_turn(
        text=phrase,
        session_id="temporal_test",
        turn_id=1,
        user_id="peppi"
    )

    print(f"  Intent: {result.intent.intent.value}")
    print(f"  Triples extracted: {len(result.triples)}")
    if result.triples:
        for s, r, d in result.triples[:3]:
            print(f"    - {s} | {r} | {d}")

    # Check if temporal extractor is enabled
    if hasattr(facade, 'temporal_extractor'):
        print(f"  Temporal extractor enabled: {facade.temporal_extractor.enabled}")

    print()

# Check feature flags
from components.memory.config import HotMemoryConfig
config = HotMemoryConfig()
print(f"\nFeature Flags:")
print(f"  use_temporal_extraction: {config.features.use_temporal_extraction}")
print(f"  use_semantic_filter: {config.features.use_semantic_filter}")
print(f"  use_coref: {config.features.use_coref}")
