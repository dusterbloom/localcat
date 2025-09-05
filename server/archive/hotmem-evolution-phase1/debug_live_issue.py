#!/usr/bin/env python3
"""
Debug the live issue from user logs: memory bullets being truncated and wrong extractions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory
import time

def test_live_issues():
    """Test the specific issues from the live logs"""
    
    print("=== Debugging Live Memory Issues ===")
    
    # Initialize memory system
    paths = Paths(sqlite_path="test_live.db", lmdb_dir="test_live.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Clean start
    try:
        store.clear_all()
    except:
        pass
    
    # Simulate the facts that should be in memory based on logs
    print("\n1. Setting up existing memory state:")
    
    # The user seems to have these facts
    existing_facts = [
        ("you", "name", "pepe"),
        ("you", "has", "favorite number"),  # This is weird - should be you favorite_number 7
        ("you", "favorite_number", "7"),  # What it should be
    ]
    
    for s, r, d in existing_facts:
        store.observe_edge(s, r, d, 0.8, int(time.time() * 1000))
        print(f"   Stored: ({s}, {r}, {d})")
    
    store.flush()
    hot.rebuild_from_store()
    
    print(f"\n2. Testing question that failed: 'Do you know my name?'")
    
    # Test the exact question from logs
    query = " Do you know my name? "
    turn_id = 1
    
    # Test the process_turn method
    bullets, triples = hot.process_turn(query, "default-user", turn_id)
    
    print(f"   Query: '{query.strip()}'")
    print(f"   Bullets returned: {bullets}")
    print(f"   Triples extracted: {triples}")
    
    print(f"\n3. Testing new statement: 'My favorite number is 77'")
    
    # Test the statement that caused weird extractions
    statement = " My favorite number is 77. "
    turn_id = 2
    
    bullets2, triples2 = hot.process_turn(statement, "default-user", turn_id)
    
    print(f"   Statement: '{statement.strip()}'")
    print(f"   Bullets returned: {bullets2}")
    print(f"   Triples extracted: {triples2}")
    
    print(f"\n4. Testing memory context injection format:")
    
    # Test what the actual context message would look like
    if bullets2:
        memory_content = "\n".join(bullets2[:3])
        memory_message = {
            "role": "user",
            "content": f"[Memory context]\n{memory_content}"
        }
        
        print(f"   Full memory message:")
        print(f"   Role: {memory_message['role']}")
        print(f"   Content: {repr(memory_message['content'])}")
        print(f"   Content length: {len(memory_message['content'])} chars")
        
        # Check if it would be truncated
        if len(memory_message['content']) > 200:
            print(f"   ⚠️  Content may be truncated (>{200} chars)")
        
    print("\n=== Live Issue Debug Complete ===")


if __name__ == "__main__":
    test_live_issues()