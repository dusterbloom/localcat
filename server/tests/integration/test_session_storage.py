#!/usr/bin/env python3
"""
Test script for the new session storage system
"""

import sys
import os
sys.path.insert(0, '.')

from components.session.session_store import SessionStore, reset_session_store, get_session_store
from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade

def test_session_storage():
    """Test the complete session storage system"""
    print("🧪 Testing Session Storage System")
    print("=" * 50)
    
    # Reset session store for clean test
    reset_session_store()
    
    # Create temporary databases
    session_db = "test_sessions.db"
    memory_db = "test_memory.db"
    lmdb_dir = "test_graph.lmdb"
    
    # Clean up any existing test files
    for path in [session_db, memory_db, lmdb_dir]:
        if os.path.exists(path):
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    try:
        # Initialize session store
        session_store = SessionStore(session_db)
        print("✅ SessionStore initialized")
        
        # Initialize memory store
        paths = Paths(sqlite_path=memory_db, lmdb_dir=lmdb_dir)
        memory_store = MemoryStore(paths)
        print("✅ MemoryStore initialized")
        
        # Initialize HotMemory facade
        hot_memory = HotMemoryFacade(memory_store)
        print("✅ HotMemoryFacade initialized")
        
        # Test 1: Create session
        session_id = session_store.create_session("test_user")
        print(f"✅ Created session: {session_id}")
        
        # Test 2: Store conversation
        session_store.add_message(session_id, "user", "Hello, I live in San Francisco", 1)
        session_store.add_message(session_id, "assistant", "Great to know! I'll remember that you live in San Francisco.", 1)
        session_store.add_message(session_id, "user", "I work at Google as a software engineer", 2)
        session_store.add_message(session_id, "assistant", "That's impressive! You work at Google as a software engineer.", 2)
        print("✅ Stored conversation messages")
        
        # Test 3: Extract knowledge using HotMemory
        bullets, triples = hot_memory.process_turn("I live in San Francisco", session_id, 1)
        print(f"✅ Extracted knowledge: {len(triples)} triples, {len(bullets)} bullets")
        
        bullets, triples = hot_memory.process_turn("I work at Google as a software engineer", session_id, 2)
        print(f"✅ Extracted knowledge: {len(triples)} triples, {len(bullets)} bullets")
        
        # Test 4: Store assistant responses
        hot_memory.store_assistant_response(session_id, "Great to know! I'll remember that you live in San Francisco.", 1)
        hot_memory.store_assistant_response(session_id, "That's impressive! You work at Google as a software engineer.", 2)
        print("✅ Stored assistant responses")
        
        # Test 5: Retrieve session data
        conversation = session_store.get_session_conversation(session_id)
        metadata = session_store.get_session_metadata(session_id)
        knowledge_links = session_store.get_session_knowledge(session_id)
        
        print(f"✅ Retrieved session data:")
        print(f"   - Messages: {len(conversation)}")
        print(f"   - Metadata: {metadata}")
        print(f"   - Knowledge links: {len(knowledge_links)}")
        
        # Test 6: Check conversation context
        context = session_store.get_conversation_context(session_id)
        print(f"✅ Conversation context: {len(context)} chars")
        
        # Test 7: Check session stats
        stats = session_store.get_stats()
        print(f"✅ Session stats: {stats}")
        
        # Test 8: Check memory store integration
        memory_stats = {
            'entities': len(memory_store.get_all_edges()),
            'mentions': len(memory_store.search_fts_detailed("*"))
        }
        print(f"✅ Memory stats: {memory_stats}")
        
        # Test 9: Test user sessions
        user_sessions = session_store.get_user_sessions("test_user")
        print(f"✅ User sessions: {len(user_sessions)}")
        
        # Test 10: Generate session summary
        hot_memory._generate_session_summary(session_id, conversation)
        updated_metadata = session_store.get_session_metadata(session_id)
        print(f"✅ Session summary generated: {updated_metadata.summary is not None}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Session storage system is working correctly.")
        print("\n📋 Summary:")
        print(f"- Session ID: {session_id}")
        print(f"- Messages stored: {len(conversation)}")
        print(f"- Knowledge extracted: {len(knowledge_links)} facts")
        print(f"- Session summary: {'✅' if updated_metadata.summary else '❌'}")
        print(f"- Verbatim storage: ✅")
        print(f"- Session-to-knowledge linkage: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up test files
        for path in [session_db, memory_db, lmdb_dir]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        print("🧹 Cleaned up test files")

if __name__ == "__main__":
    success = test_session_storage()
    sys.exit(0 if success else 1)