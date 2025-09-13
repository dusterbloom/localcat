#!/usr/bin/env python3
"""
Integration test to reproduce the hanging issue during conversation corrections.

Test scenario:
1. User: "Where do I live?"
2. Bot: (responds with some location)
3. User: "No, that is wrong. I live in Sardinia."
4. System should process the correction and continue, but it hangs.
"""

import sys
import os
import time
sys.path.insert(0, '.')

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade
from components.session.session_store import SessionStore, get_session_store


def test_conversation_correction():
    """Test the conversation correction scenario that hangs"""
    print("🧪 Testing conversation correction scenario...")
    print("=" * 60)
    
    # Clean up any existing test files
    for path in ["test_sessions.db", "test_memory.db", "test_graph.lmdb"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    try:
        # Initialize stores
        session_store = SessionStore("test_sessions.db")
        paths = Paths(sqlite_path="test_memory.db", lmdb_dir="test_graph.lmdb")
        memory_store = MemoryStore(paths)
        hot_memory = HotMemoryFacade(memory_store)
        
        print("✅ Initialized storage systems")
        
        # Create a session for user "peppi"
        session_id = session_store.create_session("peppi")
        print(f"🆕 Created session: {session_id}")
        
        # === TURN 1: User asks where they live ===
        print(f"\n👤 User: 'Where do I live?'")
        session_store.add_message(session_id, "user", "Where do I live?", 1)
        
        # Process with HotMemory
        bullets, triples = hot_memory.process_turn("Where do I live?", session_id, 1)
        print(f"🧠 Extracted {len(triples)} triples: {triples}")
        print(f"📝 Retrieved {len(bullets)} bullets: {bullets}")
        
        # Simulate bot response (this would come from LLM in real scenario)
        bot_response = "I think you live in Berlin."
        session_store.add_message(session_id, "assistant", bot_response, 1)
        hot_memory.store_assistant_response(session_id, bot_response, 1)
        print(f"🤖 Bot: '{bot_response}'")
        
        # === TURN 2: User corrects the bot ===
        print(f"\n👤 User: 'No, that is wrong. I live in Sardinia.'")
        session_store.add_message(session_id, "user", "No, that is wrong. I live in Sardinia.", 2)
        
        # This is where it hangs - let's see what happens
        print("⏳ Processing correction...")
        start_time = time.time()
        
        # Process the correction
        bullets2, triples2 = hot_memory.process_turn("No, that is wrong. I live in Sardinia.", session_id, 2)
        
        end_time = time.time()
        print(f"✅ Correction processed in {end_time - start_time:.2f} seconds")
        print(f"🧠 Extracted {len(triples2)} triples: {triples2}")
        print(f"📝 Retrieved {len(bullets2)} bullets: {bullets2}")
        
        # Store assistant response for correction
        bot_response2 = "I understand, you live in Sardinia. I've updated my memory."
        session_store.add_message(session_id, "assistant", bot_response2, 2)
        hot_memory.store_assistant_response(session_id, bot_response2, 2)
        print(f"🤖 Bot: '{bot_response2}'")
        
        # === VERIFY PERSISTENCE ===
        print(f"\n🔍 Verifying persistence...")
        
        # Check if the fact is stored in memory
        all_edges = memory_store.get_all_edges()
        sardinia_edges = [edge for edge in all_edges if 'sardinia' in str(edge).lower()]
        print(f"📊 Found {len(sardinia_edges)} edges containing 'sardinia':")
        for edge in sardinia_edges:
            print(f"  - {edge}")
        
        # Check session knowledge links
        knowledge_links = session_store.get_session_knowledge(session_id)
        print(f"🔗 Session has {len(knowledge_links)} knowledge links")
        
        # Test retrieval
        print(f"\n🧪 Testing retrieval...")
        test_bullets, test_triples = hot_memory.process_turn("Where do I live?", session_id, 3)
        print(f"📝 Retrieved {len(test_bullets)} bullets on second ask:")
        for bullet in test_bullets:
            print(f"  - {bullet}")
        
        print(f"\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        for path in ["test_sessions.db", "test_memory.db", "test_graph.lmdb"]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                else:
                    os.remove(path)


if __name__ == "__main__":
    test_conversation_correction()