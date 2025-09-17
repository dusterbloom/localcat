#!/usr/bin/env python3
"""
Test if summaries are retrieved when asking about previous sessions
"""
import os
import sys
import tempfile
from loguru import logger

sys.path.insert(0, '.')

def test_summary_retrieval():
    """Test that summaries from previous sessions are retrieved"""

    # Suppress logs
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    with tempfile.TemporaryDirectory() as tmpdir:
        from components.memory.memory_store import MemoryStore, Paths
        from components.memory.hotmemory_facade import HotMemoryFacade

        paths = Paths(
            sqlite_path=os.path.join(tmpdir, 'test.db'),
            lmdb_dir=os.path.join(tmpdir, 'test.lmdb')
        )

        store = MemoryStore(paths)
        hot = HotMemoryFacade(store)

        print("=== Summary Retrieval Test ===\n")

        # Simulate a previous session with a conversation about video games
        print("1. Simulating previous session...")

        # Store some facts from the "previous session"
        prev_session_id = "session_1758138598_peppi"

        # Store facts about the video game discussion
        result1 = hot.process_turn('I love Mario Kart on Nintendo Switch', prev_session_id, 1, 'peppi')
        print(f"   Stored: {result1.triples}")

        result2 = hot.process_turn('Low Cuts Land would be my city-state in Civilization VI', prev_session_id, 2, 'peppi')
        print(f"   Stored: {result2.triples}")

        # Now store a session summary (like what would happen at session end)
        import time
        ts = int(time.time() * 1000)
        summary_text = """Previous conversation with peppi included:
- Discussion about enjoying Mario Kart on Nintendo Switch
- Conversation about Civilization VI strategy
- Creating a city-state called "Low Cuts Land" with maritime trade focus
- Implementing "Balanced Prosperity" social policy
- Trade guilds instead of traditional banks
- Low interest rates with stability fees"""

        # Store as session summary
        eid = f"session:{prev_session_id}"
        store.enqueue_mention(eid=eid, text=summary_text, ts=ts, sid=prev_session_id, tid=10)
        store.flush()
        print(f"   Stored session summary for {eid}")

        # Also store a periodic summary (like the 30-second summaries)
        eid2 = f"summary:{prev_session_id}"
        periodic_summary = "User discussed video games, particularly Mario Kart and Civilization VI strategy"
        store.enqueue_mention(eid=eid2, text=periodic_summary, ts=ts+1, sid=prev_session_id, tid=11)
        store.flush()
        print(f"   Stored periodic summary for {eid2}\n")

        # Rebuild indices
        hot.rebuild_from_store()

        # Now start a "new session" and ask about continuing
        print("2. New session: 'Can we continue from where we were?'")
        new_session_id = "session_1758140197_peppi"

        # This should retrieve the summaries
        result3 = hot.process_turn('Can we continue from where we were?', new_session_id, 1, 'peppi')
        print(f"   Intent: {result3.intent.intent.value}")
        print(f"   Needs retrieval: {result3.needs_retrieval}")
        print(f"   Retrieved bullets: {len(result3.bullets or [])} items")

        if result3.bullets:
            print("\n   Bullets retrieved:")
            for i, bullet in enumerate(result3.bullets):
                print(f"     [{i+1}] {bullet[:100]}...")
        else:
            print("   ❌ NO BULLETS RETRIEVED!")

        # Try more specific queries
        print("\n3. Testing specific queries:")

        queries = [
            "What did we discuss about video games?",
            "Tell me about Low Cuts Land",
            "What was our conversation about?",
            "Mario Kart"
        ]

        for query in queries:
            result = hot.process_turn(query, new_session_id, 2, 'peppi')
            print(f"\n   Query: '{query}'")
            print(f"   Retrieved: {len(result.bullets or [])} bullets")
            if result.bullets:
                print(f"   First bullet: {result.bullets[0][:80]}...")

        print("\n=== Summary ===")

        # Check if FTS index has our summaries
        fts_results = store.search_fts_detailed("video games", limit=10)
        print(f"FTS search for 'video games': {len(fts_results)} results")
        for text, eid, ts in fts_results[:3]:
            print(f"  - {eid}: {text[:60]}...")

        # Final verdict
        if result3.bullets and any('video game' in str(b).lower() or 'mario' in str(b).lower() for b in result3.bullets):
            print("\n✅ SUCCESS: Summaries are being retrieved!")
        else:
            print("\n❌ FAILURE: Summaries are NOT being retrieved properly")

if __name__ == "__main__":
    test_summary_retrieval()