#!/usr/bin/env python3
"""
Debug script to test memory retrieval issue with "dog name potola"
"""
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory

def test_retrieval_issue():
    """Test the specific retrieval issue with 'dog name potola'"""
    
    print("=== Testing Memory Retrieval Issue ===")
    
    # Initialize memory system
    paths = Paths(sqlite_path="test_debug.db", lmdb_dir="test_debug.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Clean start
    try:
        store.clear_all()
    except:
        pass
    
    # Step 1: Store the fact "dog name potola"
    print("\n1. Storing fact: dog name potola")
    triples = [("dog", "name", "potola")]
    
    for s, r, d in triples:
        hot.store.observe_edge(s, r, d, 0.8, int(time.time() * 1000))
        print(f"   Stored: ({s}, {r}, {d})")
    
    # Force flush to disk
    hot.store.flush()
    print("   Flushed to disk")
    
    # Rebuild indices 
    hot.rebuild_from_store()
    print(f"   Entity index keys: {list(hot.entity_index.keys())}")
    print(f"   Facts for 'dog': {hot.entity_index.get('dog', 'NOT FOUND')}")
    print(f"   Facts for 'potola': {hot.entity_index.get('potola', 'NOT FOUND')}")
    
    # Step 2: Test light entity extraction for questions about dog
    print("\n2. Testing light entity extraction:")
    
    test_queries = [
        "what is my dog's name",
        "what's my dog's name", 
        "what is the dog's name",
        "tell me about my dog",
        "what is my Dog's name"  # Capitalized
    ]
    
    for query in test_queries:
        entities = hot._extract_entities_light(query)
        print(f"   Query: '{query}' -> Entities: {entities}")
        
        # Test retrieval
        bullets = hot._retrieve_context(query, entities, 1)
        print(f"   -> Bullets: {bullets}")
        print()
    
    # Step 3: Test full entity extraction using NLP
    print("\n3. Testing full NLP entity extraction:")
    
    for query in test_queries:
        try:
            entities, triples_found, neg_count, doc = hot._extract(query, "en")
            print(f"   Query: '{query}' -> Full entities: {entities}")
            
            # Test retrieval with full entities
            bullets = hot._retrieve_context(query, entities, 1)
            print(f"   -> Bullets: {bullets}")
            print()
        except Exception as e:
            print(f"   Error with NLP extraction: {e}")
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    import time
    test_retrieval_issue()