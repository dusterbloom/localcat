#!/usr/bin/env python3
"""
Failure analysis script to understand retrieval limitations and identify optimization opportunities.
"""

import os
import sys
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.memory.memory_store import MemoryStore, Paths
from components.memory.memory_hotpath import HotMemory, _load_nlp

def analyze_failures():
    """Analyze failed retrieval scenarios to identify patterns"""
    
    # Initialize memory components
    paths = Paths()
    memory_store = MemoryStore(paths=paths)
    hot_memory = HotMemory(memory_store)
    
    # Prewarm models
    print("🔥 Prewarming models...")
    hot_memory.prewarm('en')
    
    # Add comprehensive test data
    from test_comprehensive_retrieval import generate_test_data, add_test_data_to_memory
    add_test_data_to_memory(hot_memory, memory_store, "analysis_session")
    
    # Test scenarios that previously failed
    failed_scenarios = [
        "Emma from Apple wants to meet for lunch next week",
        "Tesla's electric cars are really impressive technology", 
        "Stanford University was a great place to study computer science",
        "I need to prepare for the AI project deadline in December",
        "My research team published a paper on NLP recently"
    ]
    
    print("\n🔍 Analyzing Failed Scenarios")
    print("=" * 60)
    
    for i, scenario in enumerate(failed_scenarios, 1):
        print(f"\n{i}. {scenario}")
        
        try:
            # Extract entities
            nlp = _load_nlp('en')
            doc = nlp(scenario) if nlp else None
            
            entities_set = set()
            entity_map = hot_memory._build_entity_map(doc, entities_set) if doc else {}
            extracted_entities = sorted(list(entities_set))
            
            print(f"   🔍 Extracted entities: {extracted_entities}")
            
            # Find matching entities in memory
            matching_entities = []
            for entity in extracted_entities:
                if entity in hot_memory.entity_index:
                    matching_entities.append(entity)
                    print(f"   ✅ Found matches for '{entity}': {len(hot_memory.entity_index[entity])} edges")
                else:
                    print(f"   ❌ No matches for '{entity}'")
            
            # Check for related entities that might help
            print(f"   🔗 Related entity analysis:")
            for entity in extracted_entities:
                related = []
                for mem_entity in hot_memory.entity_index.keys():
                    if entity.lower() in mem_entity.lower() or mem_entity.lower() in entity.lower():
                        if mem_entity != entity:
                            related.append(mem_entity)
                if related:
                    print(f"      '{entity}' → related: {related}")
            
            # Test retrieval
            bullets = hot_memory._retrieve_context(scenario, extracted_entities, turn_id=i, intent=None)
            
            print(f"   📊 Retrieved {len(bullets)} bullets")
            if bullets:
                print("   🎯 Success - bullets found:")
                for j, bullet in enumerate(bullets[:3], 1):  # Show first 3
                    print(f"      {j}. {bullet}")
            else:
                print("   ❌ Still no bullets - need optimization")
                
        except Exception as e:
            print(f"   💥 Error: {e}")

def analyze_success_patterns():
    """Analyze successful scenarios to understand patterns"""
    
    print("\n🎯 Analyzing Success Patterns")
    print("=" * 60)
    
    # Initialize memory components
    paths = Paths()
    memory_store = MemoryStore(paths=paths)
    hot_memory = HotMemory(memory_store)
    hot_memory.prewarm('en')
    
    # Add test data
    from test_comprehensive_retrieval import generate_test_data, add_test_data_to_memory
    add_test_data_to_memory(hot_memory, memory_store, "pattern_analysis")
    
    # Successful scenarios
    successful_scenarios = [
        "I need to discuss the AI project with Sarah from my team",
        "My sister who works at Google called me yesterday",
        "I'm planning my trip to Japan and want to practice Japanese",
        "The Tech Conference in Seattle was great this year",
        "I should call John about the backend development work"
    ]
    
    for i, scenario in enumerate(successful_scenarios, 1):
        print(f"\n{i}. {scenario}")
        
        try:
            # Extract entities
            nlp = _load_nlp('en')
            doc = nlp(scenario) if nlp else None
            
            entities_set = set()
            entity_map = hot_memory._build_entity_map(doc, entities_set) if doc else {}
            extracted_entities = sorted(list(entities_set))
            
            print(f"   🔍 Extracted entities: {extracted_entities}")
            
            # Find matching entities in memory
            matching_entities = []
            for entity in extracted_entities:
                if entity in hot_memory.entity_index:
                    edge_count = len(hot_memory.entity_index[entity])
                    matching_entities.append((entity, edge_count))
            
            print(f"   ✅ Direct matches: {[(e, c) for e, c in matching_entities]}")
            
            # Test retrieval
            bullets = hot_memory._retrieve_context(scenario, extracted_entities, turn_id=i, intent=None)
            
            print(f"   📊 Retrieved {len(bullets)} bullets")
            
            # Analyze scoring patterns
            if bullets:
                print(f"   🏆 Success factors:")
                print(f"      - Entity coverage: {len(matching_entities)}/{len(extracted_entities)} = {len(matching_entities)/len(extracted_entities)*100:.1f}%")
                print(f"      - Average edges per entity: {sum(c for _, c in matching_entities)/len(matching_entities) if matching_entities else 0:.1f}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")

def identify_optimization_opportunities():
    """Identify specific optimization opportunities"""
    
    print("\n🚀 Optimization Opportunities Analysis")
    print("=" * 60)
    
    # Initialize memory
    paths = Paths()
    memory_store = MemoryStore(paths=paths)
    hot_memory = HotMemory(memory_store)
    hot_memory.prewarm('en')
    
    # Add test data
    from test_comprehensive_retrieval import generate_test_data, add_test_data_to_memory
    add_test_data_to_memory(hot_memory, memory_store, "optimization_analysis")
    
    # Analyze current retrieval parameters
    print("📊 Current Retrieval Parameters:")
    print(f"   - K_max: {getattr(hot_memory, 'K_max', 'N/A')}")
    print(f"   - Scoring weights: α={getattr(hot_memory, 'alpha', 'N/A')}, β={getattr(hot_memory, 'beta', 'N/A')}, γ={getattr(hot_memory, 'gamma', 'N/A')}, δ={getattr(hot_memory, 'delta', 'N/A')}")
    print(f"   - MMR lambda: {getattr(hot_memory, 'lambda_rel', 'N/A')}")
    print(f"   - Threshold percentile: {getattr(hot_memory, 'threshold_percentile', 'N/A')}")
    
    # Entity overlap analysis
    print(f"\n🔗 Entity Relationship Analysis:")
    
    # Find potential entity synonyms or related terms
    all_entities = list(hot_memory.entity_index.keys())
    print(f"   Total entities in memory: {len(all_entities)}")
    
    # Look for potential entity variations
    entity_groups = {}
    for entity in all_entities:
        base = entity.lower().replace(' ', '_').replace('-', '_')
        if base not in entity_groups:
            entity_groups[base] = []
        entity_groups[base].append(entity)
    
    # Show potential entity groupings
    multi_entity_groups = {k: v for k, v in entity_groups.items() if len(v) > 1}
    print(f"   Potential entity groupings: {len(multi_entity_groups)}")
    for base, variants in list(multi_entity_groups.items())[:5]:  # Show first 5
        print(f"      {base}: {variants}")
    
    # Analyze connectivity
    print(f"\n🌐 Graph Connectivity Analysis:")
    total_edges = sum(len(triples) for triples in hot_memory.entity_index.values())
    avg_edges_per_entity = total_edges / len(all_entities) if all_entities else 0
    print(f"   - Total edges: {total_edges}")
    print(f"   - Average edges per entity: {avg_edges_per_entity:.1f}")
    
    # Find most connected entities
    most_connected = sorted([(entity, len(triples)) for entity, triples in hot_memory.entity_index.items()], 
                           key=lambda x: x[1], reverse=True)[:10]
    print(f"   - Top connected entities: {[(e, c) for e, c in most_connected]}")

def main():
    """Main analysis function"""
    print("🔍 HotMem Retrieval Optimization Analysis")
    print("=" * 50)
    
    analyze_failures()
    analyze_success_patterns()
    identify_optimization_opportunities()
    
    print(f"\n🎯 Next Steps for 90% Success Rate:")
    print("   1. Implement fuzzy entity matching for partial matches")
    print("   2. Add semantic similarity for non-exact entity matches") 
    print("   3. Implement multi-hop graph traversal (2-hop, 3-hop)")
    print("   4. Optimize threshold and MMR parameters")
    print("   5. Add entity synonym expansion")

if __name__ == "__main__":
    main()