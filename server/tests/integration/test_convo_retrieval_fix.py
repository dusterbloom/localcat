#!/usr/bin/env python3
"""
Test that conversation retrieval is now working after the fix.

Before fix: _convo_retrieve() skipped all conversations because
  if eid == "summary" never matched "session:peppi" format

After fix: Properly filters out summary: prefixed entries
"""

import sys
sys.path.insert(0, "/Users/peppi/Dev/localcat/server")

from core.memory.memory_hotpath import HotMemory
from core.memory.memory_store import MemoryStore

def test_convo_retrieval():
    """Test that conversations are now retrieved"""

    # Initialize memory system using config
    import os
    from core.memory.config import MemoryConfig
    from core.memory.memory_store import MemoryStore

    from core.memory.memory_store import Paths

    user_id = os.getenv("USER_ID", "peppi")

    # Create store with correct paths
    paths = Paths(
        sqlite_path="../data/memory.db",
        lmdb_dir="../data/graph.lmdb"
    )
    store = MemoryStore(paths=paths)

    hot = HotMemory(store=store)
    hot.user_eid = user_id

    # Test query that should match conversations
    query = "good evening"

    print("="*80)
    print("CONVERSATION RETRIEVAL TEST")
    print("="*80)
    print(f"\nQuery: '{query}'")
    print(f"Expected: Should return conversation bullets with [convo] tag")
    print(f"Database has: 26 entries in session:peppi, 85 in peppi")
    print()

    # Retrieve bullets
    bullets = hot.retrieve_bullets(query, read_only=True)

    print(f"Retrieved {len(bullets)} bullets:")
    for bullet in bullets:
        print(f"  {bullet}")

    # Check if we got conversation bullets
    convo_bullets = [b for b in bullets if "[convo]" in b]
    summary_bullets = [b for b in bullets if "[summary]" in b]
    graph_bullets = [b for b in bullets if "[graph]" in b]

    print()
    print(f"Breakdown:")
    print(f"  - [graph] bullets: {len(graph_bullets)}")
    print(f"  - [convo] bullets: {len(convo_bullets)}")
    print(f"  - [summary] bullets: {len(summary_bullets)}")

    # Verify fix
    if len(convo_bullets) > 0:
        print()
        print("✅ SUCCESS: Conversation retrieval is now working!")
        print(f"   Found {len(convo_bullets)} conversation bullets")
        return True
    else:
        print()
        print("❌ FAILURE: No conversation bullets retrieved")
        print("   This suggests the fix didn't work or there are no matching conversations")
        return False

if __name__ == "__main__":
    success = test_convo_retrieval()
    sys.exit(0 if success else 1)
