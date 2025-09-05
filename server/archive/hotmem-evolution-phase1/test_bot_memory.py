#!/usr/bin/env python3
"""
Quick test of bot's HotMem system to verify optimizations are working
"""
import os
import sys
import time

# Set environment for in-memory testing
os.environ["HOTMEM_SQLITE"] = ":memory:"
os.environ["HOTMEM_LMDB_DIR"] = ""

sys.path.insert(0, os.path.dirname(__file__))

from hotpath_processor import HotPathMemoryProcessor

def test_bot_memory():
    """Test the bot's memory system directly"""
    print("🚀 Testing Bot's HotMem System")
    print("="*50)
    
    # Initialize processor (same as bot.py)
    print("Initializing HotPathMemoryProcessor...")
    start_time = time.perf_counter()
    
    processor = HotPathMemoryProcessor(
        sqlite_path=":memory:",
        lmdb_dir="test_bot.lmdb",
        user_id="test-user",
        enable_metrics=True
    )
    
    init_time = (time.perf_counter() - start_time) * 1000
    print(f"✅ Initialization complete: {init_time:.1f}ms")
    
    # Test a few memory operations
    test_sentences = [
        "My name is Alex",
        "I live in Seattle", 
        "My dog's name is Buddy"
    ]
    
    print(f"\n📝 Testing {len(test_sentences)} sentences...")
    total_time = 0
    
    for i, sentence in enumerate(test_sentences, 1):
        start = time.perf_counter()
        
        # Process via HotMem (simulating what bot does)
        bullets, triples = processor.hot.process_turn(sentence, "test", i)
        
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        
        print(f"{i}. '{sentence}': {elapsed:.1f}ms")
        if triples:
            print(f"   Extracted: {triples}")
    
    avg_time = total_time / len(test_sentences)
    print(f"\n📊 Performance Summary:")
    print(f"   Average: {avg_time:.1f}ms per sentence")
    print(f"   Total: {total_time:.1f}ms")
    print(f"   Budget: {'✅ PASS' if avg_time < 30 else '❌ FAIL'} (<30ms target)")
    
    # Check metrics
    metrics = processor.hot.get_metrics()
    print(f"\n⚙️  System Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict) and 'mean' in value:
            print(f"   {key}: {value['mean']:.1f}ms avg")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    test_bot_memory()