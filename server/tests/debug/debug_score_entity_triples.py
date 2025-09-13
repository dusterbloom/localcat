#!/usr/bin/env python3
"""
Debug script to identify the exact hanging location in _score_entity_triples
"""

import sys
import os
import time
import signal
sys.path.insert(0, '.')

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade
from components.session.session_store import SessionStore, get_session_store


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def debug_score_entity_triples():
    """Debug the _score_entity_triples method specifically"""
    print("🔍 Debugging _score_entity_triples method...")
    print("=" * 60)
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout
    
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
        
        # Create a session and add initial data
        session_id = session_store.create_session("peppi")
        session_store.add_message(session_id, "user", "I live in Berlin.", 1)
        bullets, triples = hot_memory.process_turn("I live in Berlin.", session_id, 1)
        bot_response = "I understand, you live in Berlin."
        session_store.add_message(session_id, "assistant", bot_response, 1)
        hot_memory.store_assistant_response(session_id, bot_response, 1)
        
        print(f"📝 Added initial data: {len(triples)} triples")
        print(f"🔍 Entity index state: {dict(hot_memory.entity_index)}")
        
        # Now test the specific problematic query
        query = "No, that is wrong. I live in Sardinia."
        entities = ['wrong', 'you', 'that', 'sardinia']
        
        print(f"\n🧪 Testing _score_entity_triples with query: {query}")
        print(f"Entities: {entities}")
        
        # Test each entity individually
        for entity in entities:
            print(f"\n📊 Testing entity: '{entity}'")
            if entity in hot_memory.entity_index:
                print(f"  Found {len(hot_memory.entity_index[entity])} triples")
                for i, triple in enumerate(hot_memory.entity_index[entity]):
                    print(f"  Triple {i}: {triple}")
                    
                    # Test scoring this specific triple
                    try:
                        signal.alarm(3)  # 3 second timeout per triple
                        result = hot_memory.retriever._score_entity_triples(entity, query, int(time.time() * 1000), 7 * 24 * 60 * 60 * 1000)
                        print(f"  ✅ Scored successfully: {len(result)} candidates")
                    except TimeoutError:
                        print(f"  ❌ TIMEOUT scoring entity '{entity}' with triple {triple}")
                        return
                    except Exception as e:
                        print(f"  ❌ Error scoring entity '{entity}': {e}")
                        return
            else:
                print(f"  Entity not in index")
        
        print(f"\n🎉 All entities tested successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        signal.alarm(0)  # Cancel the alarm
        # Clean up
        for path in ["test_sessions.db", "test_memory.db", "test_graph.lmdb"]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                else:
                    os.remove(path)


if __name__ == "__main__":
    debug_score_entity_triples()