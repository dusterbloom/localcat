#!/usr/bin/env python3
"""Test script to verify LM Studio relation extractor integration"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add server directory to Python path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

from components.memory.memory_hotpath import HotMemory
from components.memory.memory_store import MemoryStore, Paths

async def test_relation_extractor():
    """Test if relation extractor is being called"""
    
    print("🧪 Testing HotMem with LM Studio relation extractor...")
    
    # Initialize storage
    paths = Paths(
        sqlite_path="/tmp/test_memory.db",
        lmdb_dir="/tmp/test_graph.lmdb"
    )
    
    # Clean up any existing test data
    if os.path.exists(paths.sqlite_path):
        os.remove(paths.sqlite_path)
    
    # Initialize HotMemory
    store = MemoryStore(paths)
    hot_memory = HotMemory(store)
    
    # Test text that should trigger relation extraction
    test_text = "Tim Cook is the CEO of Apple and lives in California."
    
    print(f"📝 Test text: '{test_text}'")
    print(f"🔧 Assisted extraction enabled: {hot_memory.assisted_enabled}")
    print(f"🤖 Assisted model: {hot_memory.assisted_model}")
    print(f"🌐 Assisted base URL: {hot_memory.assisted_base_url}")
    
    # Process the text
    try:
        bullets, triples = hot_memory.process_turn(
            text=test_text,
            session_id="test_session",
            turn_id=1
        )
        
        print(f"\n🎯 Results:")
        print(f"   - Memory bullets: {len(bullets)}")
        print(f"   - Extracted triples: {len(triples)}")
        
        if bullets:
            print(f"   - Bullets: {bullets[:3]}")  # Show first 3
        
        if triples:
            print(f"   - Triples: {triples[:5]}")  # Show first 5
            
            # Check if we got the expected relations
            expected_relations = ["CEO_of", "lives_in"]
            found_relations = [r for s, r, d in triples]
            
            print(f"\n📊 Relation Analysis:")
            for rel in expected_relations:
                if rel in found_relations:
                    print(f"   ✅ Found: {rel}")
                else:
                    print(f"   ❌ Missing: {rel}")
        else:
            print("   ⚠️  No triples extracted")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    try:
        os.remove(paths.sqlite_path)
    except:
        pass

if __name__ == "__main__":
    asyncio.run(test_relation_extractor())