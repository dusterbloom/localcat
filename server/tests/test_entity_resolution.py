#!/usr/bin/env python3
"""
Test script to validate intelligent entity resolution with graph traversal.
This tests the "Michael Chen" -> "your husband" scenario.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_entity_resolution():
    """Test that entity resolution can find related entities"""
    
    print("🧠 Testing HotMem V4 Intelligent Entity Resolution")
    print("=" * 60)
    
    # Import inside function to avoid circular imports
    from components.memory.memory_hotpath import HotMemory
    
    # Initialize memory system
    memory = HotMemory(
        user_id="peppi",
        sqlite_path=":memory:",
        lmdb_path=None,
        use_relik=False,
        use_leann=False,
        use_onnx_srl=False,
        use_srl=False,
        use_coref=False
    )
    
    # Store test data - simulate user facts
    test_facts = [
        "My name is Sarah Williams",
        "I work at Google as a senior software engineer", 
        "I live in San Francisco",
        "I have two children",
        "My husband is Dr. Michael Chen",
        "I have a Tesla Model 3"
    ]
    
    print("📝 Storing test facts...")
    for i, fact in enumerate(test_facts):
        bullets, triples = memory.process_turn(fact, session_id="test", turn_id=i)
        print(f"  Fact {i+1}: {fact}")
        print(f"    Extracted {len(triples)} triples")
        if triples:
            for triple in triples:
                print(f"      {triple}")
    
    # Test entity resolution scenarios
    test_queries = [
        ("Who is Michael Chen?", ["michael", "chen", "husband"]),
        ("Tell me about my husband", ["husband"]),
        ("Where do I work?", ["work", "google"]),
        ("What car do I have?", ["car", "tesla"]),
        ("Where do I live?", ["live", "san francisco"])
    ]
    
    print("\n🔍 Testing entity resolution and retrieval...")
    for query, expected_keywords in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Extract entities from query
        entities = memory._extract_entities_light(query)
        print(f"  Extracted entities: {entities}")
        
        # Test retrieval 
        bullets = memory._retrieve_context(query, entities, turn_id=100)
        print(f"  Retrieved {len(bullets)} bullets:")
        for bullet in bullets:
            print(f"    {bullet}")
        
        # Check if expected keywords are found
        all_text = " ".join(bullets).lower()
        found_keywords = [kw for kw in expected_keywords if kw in all_text]
        print(f"  Found keywords: {found_keywords}")
        
        if len(found_keywords) >= len(expected_keywords) // 2:
            print("  ✅ SUCCESS: Found relevant information")
        else:
            print("  ❌ FAILURE: Missing expected information")
    
    print("\n" + "=" * 60)
    print("🎯 Entity Resolution Test Complete")

if __name__ == "__main__":
    test_entity_resolution()