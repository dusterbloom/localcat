#!/usr/bin/env python3
"""
Simple test to verify entity resolution is working by testing a realistic scenario.
This simulates the "Who is Michael Chen?" -> "your husband" use case.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_realistic_entity_resolution():
    """Test realistic entity resolution scenario"""
    
    print("🧠 Testing HotMem V4 Realistic Entity Resolution")
    print("=" * 60)
    
    # Import inside function to avoid circular imports
    from components.memory.memory_hotpath import HotMemory
    from components.memory.memory_store import MemoryStore, Paths
    import tempfile
    
    # Initialize memory system
    with tempfile.TemporaryDirectory() as temp_dir:
        store = MemoryStore(Paths(
            sqlite_path=f"{temp_dir}/test.db",
            lmdb_dir=f"{temp_dir}/lmdb"
        ))
        memory = HotMemory(store)
        
        print("📝 Building knowledge graph with realistic user data...")
        
        # Simulate a realistic conversation about personal relationships
        conversation = [
            "My name is Sarah Williams and I'm 32 years old",
            "I work at Google as a senior software engineer", 
            "I live in San Francisco with my family",
            "I'm married to Dr. Michael Chen, he's a cardiologist",
            "We have two children, Emma and Liam",
            "I drive a Tesla Model 3 to work every day"
        ]
        
        stored_facts = []
        for i, utterance in enumerate(conversation):
            try:
                bullets, triples = memory.process_turn(utterance, session_id="test", turn_id=i)
                stored_facts.extend(triples)
                print(f"  Turn {i+1}: {utterance}")
                print(f"    Extracted {len(triples)} triples")
                for triple in triples:
                    print(f"      {triple}")
            except Exception as e:
                print(f"  Error processing turn {i+1}: {e}")
        
        print(f"\n📊 Total facts stored: {len(stored_facts)}")
        
        # Test queries that should benefit from entity resolution
        test_queries = [
            ("Who is Michael Chen?", ["husband", "michael", "chen", "cardiologist"]),
            ("Tell me about my husband", ["husband", "michael", "chen"]),
            ("Where do I work?", ["google", "software", "engineer"]),
            ("What car do I drive?", ["tesla", "model"]),
            ("Where do I live?", ["san", "francisco"]),
            ("Do I have children?", ["emma", "liam", "children"])
        ]
        
        print("\n🔍 Testing entity resolution queries...")
        
        success_count = 0
        total_count = len(test_queries)
        
        for query, expected_keywords in test_queries:
            print(f"\n❓ Query: '{query}'")
            
            try:
                # Extract entities and retrieve context
                entities = memory._extract_entities_light(query)
                bullets = memory._retrieve_context(query, entities, turn_id=100)
                
                print(f"  Entities found: {entities}")
                print(f"  Bullets retrieved: {len(bullets)}")
                
                # Check for relevant information
                all_text = " ".join(bullets).lower()
                found_keywords = [kw for kw in expected_keywords if kw in all_text]
                
                print(f"  Expected keywords: {expected_keywords}")
                print(f"  Found keywords: {found_keywords}")
                
                if found_keywords:
                    print("  ✅ SUCCESS: Found relevant information")
                    for bullet in bullets:
                        print(f"    {bullet}")
                    success_count += 1
                else:
                    print("  ❌ FAILURE: No relevant information found")
                    
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
        
        print("\n" + "=" * 60)
        print(f"📈 Entity Resolution Results: {success_count}/{total_count} queries successful")
        
        if success_count >= total_count * 0.7:
            print("🎉 GOOD: Entity resolution is working reasonably well")
        elif success_count >= total_count * 0.5:
            print("⚠️  FAIR: Entity resolution needs improvement")
        else:
            print("❌ POOR: Entity resolution needs significant work")
        
        return success_count >= total_count * 0.7

if __name__ == "__main__":
    success = test_realistic_entity_resolution()
    sys.exit(0 if success else 1)