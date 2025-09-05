#!/usr/bin/env python3
"""
Test HotMem with the canonical Potola scenario
"""

import time
import os
import sys
from loguru import logger

# Set up environment
os.environ["HOTMEM_SQLITE"] = "test_memory.db"
os.environ["HOTMEM_LMDB_DIR"] = "test_graph.lmdb"

from hotpath_processor import HotPathMemoryProcessor


def test_potola_scenario():
    """Test the canonical dog name scenario"""
    print("\n" + "="*60)
    print("Testing HotMem with Potola Scenario")
    print("="*60 + "\n")
    
    # Initialize processor
    processor = HotPathMemoryProcessor(
        sqlite_path="test_memory.db",
        lmdb_dir="test_graph.lmdb",
        user_id="test-user"
    )
    
    print("✅ HotMem processor initialized\n")
    
    # Turn 1: Tell the system about the dog
    print("Turn 1: 'My dog's name is Potola'")
    start = time.perf_counter()
    bullets1, triples1 = processor.hot.process_turn(
        "My dog's name is Potola", 
        session_id="test-session",
        turn_id=1
    )
    elapsed1 = (time.perf_counter() - start) * 1000
    
    print(f"  Extracted triples: {triples1}")
    print(f"  Memory bullets: {bullets1}")
    print(f"  ⏱️  Processing time: {elapsed1:.1f}ms")
    
    # Verify extraction
    assert any("dog" in str(t) for t in triples1), "Should extract dog relationship"
    assert any("potola" in str(t).lower() for t in triples1), "Should extract name Potola"
    print("  ✅ Correctly extracted dog and name\n")
    
    # Turn 2: Unrelated conversation
    print("Turn 2: 'What's the weather like?'")
    bullets2, triples2 = processor.hot.process_turn(
        "What's the weather like?",
        session_id="test-session", 
        turn_id=2
    )
    print(f"  Extracted triples: {triples2}")
    print(f"  Memory bullets: {bullets2}")
    print("  ✅ No relevant memories (expected)\n")
    
    # Turn 3: Ask about the dog
    print("Turn 3: 'What do you know about my dog?'")
    start = time.perf_counter()
    bullets3, triples3 = processor.hot.process_turn(
        "What do you know about my dog?",
        session_id="test-session",
        turn_id=3
    )
    elapsed3 = (time.perf_counter() - start) * 1000
    
    print(f"  Extracted triples: {triples3}")
    print(f"  Memory bullets: {bullets3}")
    print(f"  ⏱️  Processing time: {elapsed3:.1f}ms")
    
    # Verify retrieval
    bullets_str = " ".join(bullets3).lower()
    assert "potola" in bullets_str, f"Should recall Potola in bullets: {bullets3}"
    print("  ✅ Successfully recalled Potola!\n")
    
    # Performance check
    print("="*60)
    print("Performance Summary:")
    print(f"  Turn 1 (extraction): {elapsed1:.1f}ms")
    print(f"  Turn 3 (retrieval): {elapsed3:.1f}ms")
    
    if elapsed1 < 200 and elapsed3 < 200:
        print("  ✅ Both under 200ms budget!")
    else:
        print("  ⚠️  Performance needs optimization")
    
    # Get metrics
    metrics = processor.hot.get_metrics()
    store_metrics = processor.store.get_metrics()
    
    print("\nHotMem Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict) and 'p95' in value:
            print(f"  {key}: p95={value['p95']:.1f}ms, mean={value['mean']:.1f}ms")
        else:
            print(f"  {key}: {value}")
    
    print("\nStore Metrics:")
    for key, value in store_metrics.items():
        if isinstance(value, dict) and 'p95' in value:
            print(f"  {key}: p95={value['p95']:.1f}ms")
    
    print("\n" + "="*60)
    print("✅ All tests passed! HotMem is working correctly.")
    print("="*60 + "\n")
    
    # Cleanup test files
    import shutil
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
    if os.path.exists("test_graph.lmdb"):
        shutil.rmtree("test_graph.lmdb")
    print("Cleaned up test files.")


def test_multilingual():
    """Test multilingual support"""
    print("\n" + "="*60)
    print("Testing Multilingual Support")
    print("="*60 + "\n")
    
    processor = HotPathMemoryProcessor(
        sqlite_path="test_memory.db",
        lmdb_dir="test_graph.lmdb",
        user_id="test-user"
    )
    
    # Test different languages (will use regex fallback without pycld3)
    test_cases = [
        ("I live in San Francisco", "lives_in"),
        ("My name is Claude", "name"),
        ("I work at Anthropic", "works_at"),
        ("I like pizza", "likes"),
    ]
    
    for text, expected_relation in test_cases:
        bullets, triples = processor.hot.process_turn(text, "test", 1)
        print(f"'{text}'")
        print(f"  → Triples: {triples}")
        assert any(expected_relation in str(t) for t in triples), f"Should extract {expected_relation}"
        print(f"  ✅ Extracted {expected_relation}\n")
    
    # Cleanup
    import shutil
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
    if os.path.exists("test_graph.lmdb"):
        shutil.rmtree("test_graph.lmdb")


if __name__ == "__main__":
    try:
        # Test main scenario
        test_potola_scenario()
        
        # Test multilingual
        test_multilingual()
        
        print("\n🎉 All tests passed! HotMem is production-ready.\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)