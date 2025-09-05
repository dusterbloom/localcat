#!/usr/bin/env python3
"""
Test comprehensive bullet formatting for different relation types
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory
import time

def test_bullet_formatting():
    """Test bullet formatting for various relation types"""
    
    print("=== Testing Memory Bullet Formatting ===")
    
    # Initialize memory system
    paths = Paths(sqlite_path="test_format.db", lmdb_dir="test_format.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Clean start
    try:
        store.clear_all()
    except:
        pass
    
    # Test facts with different relation types
    test_facts = [
        ("dog", "name", "potola"),
        ("you", "name", "alex"),
        ("dog", "age", "5 years old"),
        ("you", "age", "25"),
        ("cat", "has", "fluffy tail"),
        ("you", "has", "blue eyes"),
        ("sarah", "is", "developer"),
        ("you", "is", "happy"),
        ("john", "lives_in", "paris"),
        ("you", "lives_in", "san francisco"),
        ("jane", "works_at", "google"),
        ("you", "works_at", "anthropic"),
        ("mike", "v:likes", "pizza"),
        ("you", "v:enjoys", "coding"),
        ("car", "favorite_color", "red"),
        ("you", "friend_of", "alice"),
    ]
    
    print(f"\n1. Storing {len(test_facts)} test facts...")
    for s, r, d in test_facts:
        store.observe_edge(s, r, d, 0.8, int(time.time() * 1000))
        print(f"   Stored: ({s}, {r}, {d})")
    
    # Force flush and rebuild
    store.flush()
    hot.rebuild_from_store()
    
    print(f"\n2. Testing bullet formatting:")
    
    # Test each fact's formatting
    for s, r, d in test_facts:
        formatted = hot._format_memory_bullet(s, r, d)
        print(f"   ({s}, {r}, {d}) -> {formatted}")
    
    print("\n3. Testing queries that should retrieve formatted bullets:")
    
    test_queries = [
        ("what is my name", ["you"]),
        ("tell me about my dog", ["dog", "you"]),
        ("what does sarah do", ["sarah"]),
        ("where do you live", ["you"]),
        ("what does mike like", ["mike"]),
    ]
    
    for query, expected_entities in test_queries:
        print(f"\n   Query: '{query}'")
        
        # Test light extraction
        light_entities = hot._extract_entities_light(query)
        light_bullets = hot._retrieve_context(query, light_entities, 1)
        print(f"   Light entities: {light_entities}")
        print(f"   Light bullets: {light_bullets}")
        
    print("\n=== Formatting Test Complete ===")

if __name__ == "__main__":
    test_bullet_formatting()