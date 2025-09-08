#!/usr/bin/env python3
"""
Test the enhanced ReLiK replacement with various complex sentences.
"""

import os
import sys
import time
import tempfile
from components.memory.memory_hotpath import HotMemory
from components.memory.memory_store import MemoryStore, Paths

# Enable enhanced extractor
os.environ['HOTMEM_USE_RELIK'] = 'true'

def test_sentences():
    """Test various complex sentences with the enhanced extractor."""
    
    test_cases = [
        # Passive voice with conjunctions
        "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino in 1976.",
        
        # Compound sentence with multiple relations
        "My brother Tom lives in Portland and teaches at Reed College.",
        
        # Discovery with complex object
        "Marie Curie discovered radium and polonium with her husband Pierre.",
        
        # CEO relationship
        "Elon Musk is the CEO of Tesla and SpaceX.",
        
        # Complex nested sentence
        "The professor who taught the class that I took last semester recently published a book about AI.",
        
        # Location and capital
        "Barcelona, which is the capital of Catalonia, is the second largest city in Spain.",
        
        # Family relations with attributes
        "My sister Sarah, who is 28 years old, works at Microsoft as a software engineer.",
        
        # Educational background
        "John studied computer science at MIT and later got his PhD from Stanford.",
    ]
    
    # Create temporary store
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = Paths(
            sqlite_path=os.path.join(tmpdir, 'test.db'),
            lmdb_dir=os.path.join(tmpdir, 'test.lmdb')
        )
        store = MemoryStore(paths)
        hm = HotMemory(store)
        
        print("=" * 80)
        print("Enhanced ReLiK Replacement - Complex Sentence Testing")
        print("=" * 80)
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {text[:60]}...")
            
            # Process the sentence
            start = time.perf_counter()
            bullets, stored_facts = hm.process_turn(text, turn_id=str(i), session_id=f"test_session_{i}")
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"⏱️  Processing time: {elapsed:.1f}ms")
            
            # Show extracted facts
            if stored_facts:
                print(f"✅ Extracted {len(stored_facts)} facts:")
                for s, r, o in stored_facts[:5]:  # Show first 5
                    print(f"   ({s}) -[{r}]-> ({o})")
                if len(stored_facts) > 5:
                    print(f"   ... and {len(stored_facts) - 5} more")
            else:
                print("❌ No facts extracted")
            
            # Show generated bullets
            if bullets:
                print(f"📋 Generated {len(bullets)} bullets:")
                for bullet in bullets[:5]:  # Show first 5
                    print(f"   {bullet}")
                if len(bullets) > 3:
                    print(f"   ... and {len(bullets) - 3} more")
        
        # Test retrieval
        print("\n" + "=" * 80)
        print("Testing Retrieval")
        print("=" * 80)
        
        queries = [
            "Who founded Apple?",
            "Where does Tom live?",
            "What did Marie Curie discover?",
            "What companies does Elon Musk run?",
            "Where did John study?",
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            start = time.perf_counter()
            results = hm.retrieve(query)
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"⏱️  Retrieval time: {elapsed:.1f}ms")
            if results:
                print(f"📊 Found {len(results)} results:")
                for result in results[:2]:
                    print(f"   • {result}")
            else:
                print("❌ No results found")
        
        print("\n" + "=" * 80)
        print("Test Complete!")
        print("=" * 80)


if __name__ == "__main__":
    test_sentences()