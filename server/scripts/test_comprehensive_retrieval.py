#!/usr/bin/env python3
"""
Comprehensive test script for HotMem retrieval validation.
Adds diverse test data and validates retrieval across different session IDs.
"""

import os
import sys
import time
import random
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.memory.memory_store import MemoryStore, Paths
from components.memory.memory_hotpath import HotMemory, _load_nlp

def generate_test_data() -> List[Dict[str, Any]]:
    """Generate comprehensive test data for validation"""
    return [
        # Personal information
        {"triples": [("you", "has", "sister"), ("sister", "works_at", "Google"), ("sister", "lives_in", "San Francisco")],
         "description": "Family information - sister at Google"},
        
        {"triples": [("you", "studied_at", "Stanford University"), ("you", "graduated_in", "2020"), ("you", "major_in", "Computer Science")],
         "description": "Education background"},
        
        # Work related
        {"triples": [("you", "works_at", "TechCorp"), ("you", "position", "Senior Engineer"), ("TechCorp", "located_in", "Silicon Valley")],
         "description": "Work information"},
        
        {"triples": [("Sarah", "is", "manager"), ("Sarah", "works_at", "TechCorp"), ("Sarah", "manages", "AI team")],
         "description": "Manager information"},
        
        # Interests and hobbies
        {"triples": [("you", "enjoys", "hiking"), ("you", "hikes_in", "Rocky Mountains"), ("you", "goes_hiking", "every weekend")],
         "description": "Hobby - hiking"},
        
        {"triples": [("you", "plays", "guitar"), ("you", "plays_in", "band"), ("band", "genre", "rock")],
         "description": "Music hobby"},
        
        # Recent events
        {"triples": [("you", "attended", "Tech Conference 2024"), ("conference", "location", "Seattle"), ("you", "met", "Elon Musk")],
         "description": "Recent conference"},
        
        {"triples": [("you", "visited", "Japan"), ("you", "visited", "Tokyo"), ("you", "tried", "sushi")],
         "description": "Travel experience"},
        
        # Scientific/Technical
        {"triples": [("Albert Einstein", "developed", "Theory of Relativity"), ("Einstein", "born_in", "Germany"), ("Einstein", "won", "Nobel Prize")],
         "description": "Scientific facts"},
        
        {"triples": [("Tesla", "manufactures", "electric cars"), ("Tesla", "founded_by", "Elon Musk"), ("Tesla", "headquarters", "Austin")],
         "description": "Company information"},
        
        # Projects
        {"triples": [("you", "working_on", "AI project"), ("project", "deadline", "December"), ("project", "uses", "machine learning")],
         "description": "Current project"},
        
        {"triples": [("you", "collaborates_with", "research team"), ("team", "focus_on", "NLP"), ("team", "published", "paper")],
         "description": "Research collaboration"},
        
        # Relationships
        {"triples": [("John", "is", "colleague"), ("John", "specializes_in", "backend"), ("you", "work_with", "John")],
         "description": "Work relationships"},
        
        {"triples": [("Emma", "is", "friend"), ("Emma", "works_at", "Apple"), ("you", "meet_with", "Emma", "weekly")],
         "description": "Personal relationships"},
        
        # Skills
        {"triples": [("you", "skilled_in", "Python"), ("you", "skilled_in", "machine learning"), ("you", "skilled_in", "data analysis")],
         "description": "Technical skills"},
        
        # Health/Wellness
        {"triples": [("you", "exercises", "regularly"), ("you", "runs", "5K daily"), ("you", "prefers", "morning runs")],
         "description": "Exercise routine"},
        
        {"triples": [("you", "meditates", "daily"), ("meditation", "duration", "30 minutes"), ("meditation", "helps", "focus")],
         "description": "Meditation practice"},
        
        # Goals
        {"triples": [("you", "wants_to", "learn Japanese"), ("you", "practices", "Japanese"), ("you", "plans_to", "visit Japan again")],
         "description": "Language learning goals"},
        
        {"triples": [("you", "plans_to", "start startup"), ("startup", "focus", "AI"), ("you", "researching", "market")],
         "description": "Career goals"},
    ]

def add_test_data_to_memory(hot_memory: HotMemory, memory_store: MemoryStore, session_id: str = "test_session"):
    """Add comprehensive test data to memory"""
    test_data = generate_test_data()
    
    print(f"Adding {len(test_data)} test scenarios to memory...")
    
    for i, data in enumerate(test_data):
        print(f"  {i+1}. {data['description']}")
        
        for triple in data["triples"]:
            if len(triple) == 3:
                s, r, d = triple
                conf = random.uniform(0.7, 0.95)  # High confidence for test data
                
                # Add to memory store
                memory_store.observe_edge(s, r, d, conf=conf, now_ts=int(time.time() * 1000))
                
                # Add to hot memory entity index
                hot_memory.entity_index[s].add((s, r, d))
                hot_memory.entity_index[d].add((s, r, d))
                
                # Add metadata
                hot_memory.edge_meta[(s, r, d)] = {
                    'ts': int(time.time() * 1000),
                    'weight': conf,
                    'session_id': session_id,
                    'description': data['description']
                }
    
    # Print summary
    total_edges = sum(len(triples) for triples in hot_memory.entity_index.values())
    print(f"\n✅ Added {total_edges} edges across {len(hot_memory.entity_index)} entities")

def test_retrieval_scenarios(hot_memory: HotMemory, session_id: str = "test_session"):
    """Test retrieval with various scenarios"""
    
    test_scenarios = [
        "I need to discuss the AI project with Sarah from my team",
        "My sister who works at Google called me yesterday",
        "I'm planning my trip to Japan and want to practice Japanese",
        "The Tech Conference in Seattle was great this year",
        "I should call John about the backend development work",
        "Emma from Apple wants to meet for lunch next week",
        "I'm working on my machine learning skills for the project",
        "My morning run and meditation routine helps me focus",
        "Tesla's electric cars are really impressive technology",
        "Stanford University was a great place to study computer science",
        "I need to prepare for the AI project deadline in December",
        "My research team published a paper on NLP recently",
        "I want to start my own AI startup someday",
        "Elon Musk has some interesting ideas about the future",
        "The band I play guitar with is practicing new rock songs"
    ]
    
    print(f"\n🧪 Testing retrieval with {len(test_scenarios)} scenarios...")
    print(f"Session ID: {session_id}")
    print("=" * 60)
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario}")
        
        try:
            # Extract entities using spaCy
            nlp = _load_nlp('en')
            doc = nlp(scenario) if nlp else None
            
            entities_set = set()
            entity_map = hot_memory._build_entity_map(doc, entities_set) if doc else {}
            extracted_entities = sorted(list(entities_set))
            
            print(f"   🔍 Extracted entities: {extracted_entities}")
            
            # Test retrieval with extracted entities
            bullets = hot_memory._retrieve_context(scenario, extracted_entities, turn_id=i, intent=None)
            
            result = {
                'scenario': scenario,
                'bullet_count': len(bullets),
                'bullets': bullets,
                'entities': extracted_entities
            }
            results.append(result)
            
            print(f"   📊 Retrieved {len(bullets)} bullets:")
            for j, bullet in enumerate(bullets, 1):
                print(f"      {j}. {bullet}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'scenario': scenario,
                'bullet_count': 0,
                'bullets': [],
                'error': str(e)
            })
    
    # Print summary statistics
    total_bullets = sum(r['bullet_count'] for r in results)
    avg_bullets = total_bullets / len(results) if results else 0
    successful_scenarios = len([r for r in results if r['bullet_count'] > 0])
    
    print(f"\n📈 Retrieval Summary:")
    print(f"   Total scenarios tested: {len(test_scenarios)}")
    print(f"   Successful retrievals: {successful_scenarios}")
    print(f"   Average bullets per scenario: {avg_bullets:.1f}")
    print(f"   Total bullets retrieved: {total_bullets}")
    
    return results

def main():
    """Main test function"""
    print("🧪 Comprehensive HotMem Retrieval Test")
    print("=" * 50)
    
    # Initialize memory components
    paths = Paths()
    memory_store = MemoryStore(paths=paths)
    hot_memory = HotMemory(memory_store)
    
    # Prewarm models
    print("🔥 Prewarming models...")
    hot_memory.prewarm('en')
    
    # Test with different session IDs
    session_ids = ["test_session_1", "test_session_2", "test_session_3"]
    
    for session_id in session_ids:
        print(f"\n🚀 Testing with session: {session_id}")
        print("-" * 40)
        
        # Add test data
        add_test_data_to_memory(hot_memory, memory_store, session_id)
        
        # Test retrieval
        results = test_retrieval_scenarios(hot_memory, session_id)
        
        # Print session summary
        avg_bullets = sum(r['bullet_count'] for r in results) / len(results) if results else 0
        print(f"\n✅ Session {session_id} complete - Average: {avg_bullets:.1f} bullets/scenario")
        
        # Brief pause between sessions
        if session_id != session_ids[-1]:
            time.sleep(1)
    
    print(f"\n🎉 All tests completed successfully!")
    
    # Final memory stats
    total_edges = sum(len(triples) for triples in hot_memory.entity_index.values())
    print(f"📊 Final memory state: {total_edges} edges across {len(hot_memory.entity_index)} entities")

if __name__ == "__main__":
    main()