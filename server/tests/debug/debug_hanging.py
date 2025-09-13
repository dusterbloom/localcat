#!/usr/bin/env python3
"""
Debug version to identify where exactly the hanging occurs
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


def debug_hanging_issue():
    """Debug the hanging issue step by step"""
    print("🔍 Debugging the hanging issue...")
    print("=" * 60)
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)  # 15 second timeout
    
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
        
        # === First, let's populate some data ===
        print(f"\n📝 Step 1: Adding initial data...")
        session_store.add_message(session_id, "user", "I live in Berlin.", 1)
        
        # Process with HotMemory - this should work
        print("Processing initial statement...")
        bullets, triples = hot_memory.process_turn("I live in Berlin.", session_id, 1)
        print(f"🧠 Initial extraction: {len(triples)} triples")
        
        # Simulate bot response
        bot_response = "I understand, you live in Berlin."
        session_store.add_message(session_id, "assistant", bot_response, 1)
        hot_memory.store_assistant_response(session_id, bot_response, 1)
        
        # === Now let's examine the entity_index state ===
        print(f"\n🔍 Step 2: Examining entity_index state...")
        print(f"Entity index keys: {list(hot_memory.entity_index.keys())}")
        
        total_edges = 0
        for entity, triples_set in hot_memory.entity_index.items():
            print(f"  Entity '{entity}': {len(triples_set)} triples")
            for triple in triples_set:
                print(f"    - {triple}")
                total_edges += 1
        
        print(f"Total edges calculated manually: {total_edges}")
        
        # === Test the problematic query step by step ===
        print(f"\n🧪 Step 3: Testing the problematic query...")
        query = "No, that is wrong. I live in Sardinia."
        entities = ['wrong', 'you', 'that', 'sardinia']
        
        print(f"Query: {query}")
        print(f"Initial entities: {entities}")
        
        # Test entity expansion
        print("Testing entity expansion...")
        try:
            signal.alarm(5)  # 5 second timeout for this step
            expanded = hot_memory.retriever._expand_query_entities(entities, query)
            print(f"✅ Entity expansion successful: {expanded}")
        except TimeoutError:
            print("❌ Entity expansion timed out!")
            return
        except Exception as e:
            print(f"❌ Entity expansion failed: {e}")
            return
        
        signal.alarm(15)  # Reset timeout
        
        # Test candidate gathering
        print("Testing candidate gathering...")
        try:
            signal.alarm(5)  # 5 second timeout for this step
            candidates = hot_memory.retriever._gather_candidate_triples(query, expanded, None)
            print(f"✅ Candidate gathering successful: {len(candidates)} candidates")
        except TimeoutError:
            print("❌ Candidate gathering timed out!")
            return
        except Exception as e:
            print(f"❌ Candidate gathering failed: {e}")
            return
        
        signal.alarm(15)  # Reset timeout
        
        # Test MMR selection
        print("Testing MMR selection...")
        try:
            signal.alarm(5)  # 5 second timeout for this step
            bullets = hot_memory.retriever._apply_mmr_selection(query, candidates, 2)
            print(f"✅ MMR selection successful: {len(bullets)} bullets")
        except TimeoutError:
            print("❌ MMR selection timed out!")
            return
        except Exception as e:
            print(f"❌ MMR selection failed: {e}")
            return
        
        print(f"\n🎉 All steps completed successfully!")
        
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
    debug_hanging_issue()