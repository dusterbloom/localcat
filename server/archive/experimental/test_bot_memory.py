#!/usr/bin/env python3
"""
Test the bot's memory with the USGS 27-pattern implementation
This simulates what would happen in a conversation
"""

import os
os.environ["HOTMEM_SQLITE"] = "bot_test.db"
os.environ["HOTMEM_LMDB_DIR"] = "bot_test.lmdb"

from hotpath_processor import HotPathMemoryProcessor

# Initialize the processor like bot.py does
processor = HotPathMemoryProcessor(
    sqlite_path="bot_test.db",
    lmdb_dir="bot_test.lmdb", 
    user_id="test-user",
    enable_metrics=True
)

# Simulate a conversation
conversation = [
    "Hi, my name is Alex Thompson",
    "I live in Seattle and work at Microsoft",
    "My dog's name is Potola",
    "I have three cats named Whiskers, Shadow, and Luna",
    "My favorite programming language is Python",
    "I was born in Chicago in 1995",
    "Caroline and I are colleagues",
    "I moved from Sweden 4 years ago",
    "I enjoy hiking and photography"
]

print("Simulating conversation with bot memory")
print("="*60)

for i, text in enumerate(conversation, 1):
    print(f"\n[User {i}]: {text}")
    
    bullets, triples = processor.hot.process_turn(text, "session-1", i)
    
    if triples:
        print(f"  Extracted: {len(triples)} facts")
        for t in triples[:3]:  # Show first 3
            print(f"    {t}")
    
    if bullets:
        print(f"  Memory context ({len(bullets)} bullets):")
        for b in bullets:
            print(f"    {b}")

# Now test some queries
print("\n" + "="*60)
print("TESTING MEMORY RECALL")
print("="*60)

queries = [
    "What is my name?",
    "Where do I live?",
    "What are my pets' names?",
    "Tell me about myself"
]

for query in queries:
    print(f"\n[Query]: {query}")
    bullets, _ = processor.hot.process_turn(query, "session-1", len(conversation) + queries.index(query) + 1)
    
    if bullets:
        print("  Retrieved memories:")
        for b in bullets:
            print(f"    {b}")
    else:
        print("  No memories retrieved")

# Show performance metrics
print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

stats = processor.get_memory_stats()
for key, value in stats['hot_metrics'].items():
    if isinstance(value, dict) and 'p95' in value:
        print(f"{key}: p95={value['p95']:.1f}ms, mean={value['mean']:.1f}ms")
    else:
        print(f"{key}: {value}")

# Cleanup
import shutil
if os.path.exists("bot_test.db"):
    os.remove("bot_test.db")
if os.path.exists("bot_test.lmdb"):
    shutil.rmtree("bot_test.lmdb")
print("\n✅ Test complete, cleaned up test files")