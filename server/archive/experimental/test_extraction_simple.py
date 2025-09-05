#!/usr/bin/env python3
"""
Simple test to debug extraction in HotMem
"""

import os
import time
os.environ["HOTMEM_SQLITE"] = ":memory:"
# os.environ["HOTMEM_LMDB_DIR"] = None

from hotpath_processor import HotPathMemoryProcessor

processor = HotPathMemoryProcessor(
    sqlite_path=":memory:",
    lmdb_dir=None,
    user_id="test-user"
)

test_cases = [
    "My name is Alex",
    "I live in Seattle",
    "My dog's name is Potola",
    "I have three pets"
]

print("Testing HotMem extraction")
print("="*50)

for i, text in enumerate(test_cases, 1):
    print(f"\nTest {i}: '{text}'")
    
    bullets, triples = processor.hot.process_turn(
        text,
        session_id="test",
        turn_id=i
    )
    
    print(f"  Triples: {triples}")
    print(f"  Bullets: {bullets}")

# Now query
print("\n" + "="*50)
print("Query: What do you know about me?")
bullets, triples = processor.hot.process_turn(
    "What do you know about me?",
    session_id="test", 
    turn_id=len(test_cases)+1
)
print(f"Retrieved bullets: {bullets}")