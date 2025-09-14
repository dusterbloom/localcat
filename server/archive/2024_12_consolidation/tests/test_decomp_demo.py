#!/usr/bin/env python3
"""
Demonstrate DECOMP benefits on complex sentences
"""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from components.memory.memory_hotpath import HotMemory
from components.memory.memory_store import MemoryStore, Paths

def test_decomposition_benefits():
    """Show clear benefits of sentence decomposition"""
    
    # Complex sentences that should benefit from decomposition
    complex_sentences = [
        "Sarah, who works at Google in Mountain View, recently moved to San Francisco and bought a Tesla Model 3.",
        "When John founded OpenAI in 2015, he wanted to ensure that artificial intelligence benefits all of humanity, so he made it a non-profit organization.",
        "The company that Elon Musk started, which focuses on electric vehicles and sustainable energy, has revolutionized the automotive industry since its IPO in 2010.",
        "If Sarah continues working at Google while living in San Francisco, she will need to commute daily, which might be why she bought the Tesla."
    ]
    
    print("🔧 Testing DECOMP Feature Benefits")
    print("=" * 60)
    
    for sentence in complex_sentences:
        print(f"\n📝 Testing: {sentence[:60]}...")
        
        # Test WITHOUT decomposition
        print("\n❌ WITHOUT DECOMP:")
        os.environ["HOTMEM_DECOMPOSE_CLAUSES"] = "false"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Paths(
                sqlite_path=f"{temp_dir}/test.db",
                lmdb_dir=f"{temp_dir}/lmdb"
            ))
            memory = HotMemory(store=store)
            
            try:
                bullets, triples = memory.process_turn(sentence, session_id="test", turn_id=1)
                print(f"   Extracted {len(triples)} triples:")
                for triple in triples[:5]:  # Show first 5
                    print(f"   - {triple}")
                no_decomp_count = len(triples)
            except Exception as e:
                print(f"   Error: {e}")
                no_decomp_count = 0
        
        # Test WITH decomposition  
        print("\n✅ WITH DECOMP:")
        os.environ["HOTMEM_DECOMPOSE_CLAUSES"] = "true"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Paths(
                sqlite_path=f"{temp_dir}/test.db",
                lmdb_dir=f"{temp_dir}/lmdb"
            ))
            memory = HotMemory(store=store)
            
            try:
                bullets, triples = memory.process_turn(sentence, session_id="test", turn_id=1)
                print(f"   Extracted {len(triples)} triples:")
                for triple in triples[:8]:  # Show first 8
                    print(f"   - {triple}")
                with_decomp_count = len(triples)
            except Exception as e:
                print(f"   Error: {e}")
                with_decomp_count = 0
        
        # Show improvement
        improvement = with_decomp_count - no_decomp_count
        if improvement > 0:
            print(f"\n🎯 DECOMP BENEFIT: +{improvement} additional facts extracted!")
        elif improvement == 0:
            print(f"\n🤔 No improvement detected")
        else:
            print(f"\n⚠️  Unexpected: -{abs(improvement)} fewer facts")
        
        print("-" * 60)

if __name__ == "__main__":
    test_decomposition_benefits()