#!/usr/bin/env python3
"""
Test HotMem extraction with complex literary dialogue and long sentences
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths
import tempfile
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)

# Test cases: Complex literary dialogue and speeches
TEST_CASES = [
    {
        "name": "Casablanca - Rick's monologue", 
        "text": "I came to Casablanca for the waters, even though I was misinformed because Casablanca is in the desert, and I stayed because Ilsa walked into my gin joint out of all the gin joints in all the towns in all the world."
    },
    {
        "name": "Hamlet - To be or not to be",
        "text": "Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and, by opposing, end them, is the question that haunts me."
    },
    {
        "name": "Pride and Prejudice - Complex introduction", 
        "text": "Mr. Darcy owns Pemberley estate which has ten thousand acres, he rides horses every morning, and his sister Georgiana plays piano beautifully while their aunt Lady Catherine lives in Kent."
    },
    {
        "name": "Great Gatsby - Backstory",
        "text": "Jay Gatsby, whose real name is James Gatz, was born in North Dakota, worked on Dan Cody's yacht for five years, and now lives in West Egg where he throws lavish parties hoping Daisy will attend."
    },
    {
        "name": "Conversation - Multiple facts",
        "text": "Yesterday I met Sarah who works at Microsoft in Seattle, she graduated from Stanford in 2019, drives a Tesla, and mentioned that her brother Tom lives in Portland and teaches at Reed College."
    }
]

def test_complex_extraction():
    """Test HotMem with complex sentences"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = Paths(
            sqlite_path=os.path.join(temp_dir, "memory.db"),
            lmdb_dir=os.path.join(temp_dir, "graph.lmdb")
        )
        store = MemoryStore(paths)
        hotmem = HotMemory(store)
        
        print("🧪 Testing HotMem with Complex Literary Dialogue")
        print("=" * 60)
        
        for i, case in enumerate(TEST_CASES, 1):
            print(f"\n{i}. {case['name']}")
            print(f"Input: {case['text']}")
            print("-" * 40)
            
            try:
                # Extract facts
                bullets, triples = hotmem.process_turn(case['text'], session_id="test", turn_id=i)
                
                print(f"📝 Extracted {len(triples)} triples:")
                for j, (s, r, d) in enumerate(triples, 1):
                    print(f"   {j}. ({s}) --[{r}]--> ({d})")
                
                print(f"💡 Generated {len(bullets)} bullets:")
                for j, bullet in enumerate(bullets, 1):
                    print(f"   {j}. {bullet}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
            print("-" * 40)
        
        # Show final metrics
        print(f"\n📊 Final Metrics:")
        metrics = hotmem.get_metrics()
        for key, values in metrics.items():
            if isinstance(values, dict):
                print(f"   {key}: mean={values.get('mean', 0):.1f}ms, p95={values.get('p95', 0):.1f}ms")
            else:
                print(f"   {key}: {values}")

if __name__ == "__main__":
    test_complex_extraction()