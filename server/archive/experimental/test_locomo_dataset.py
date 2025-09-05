#!/usr/bin/env python3
"""
Test HotMem with locomo10.json dataset
Convert Q&A pairs to statements and test extraction/retrieval
"""

import json
import os
import time

# Set up test environment
os.environ["HOTMEM_SQLITE"] = "test_locomo.db"
os.environ["HOTMEM_LMDB_DIR"] = "test_locomo.lmdb"

from hotpath_processor import HotPathMemoryProcessor

# Load dataset
with open("/Users/peppi/Dev/localcat/docs/locomo10.json") as f:
    data = json.load(f)

# Extract some Q&A pairs and convert to statements
test_statements = [
    # From Q&A about Caroline
    "Caroline went to the LGBTQ support group on 7 May 2023",
    "Caroline's identity is transgender woman",
    "Caroline researched adoption agencies", 
    "Caroline is single",
    "Caroline gave a speech at a school the week before 9 June 2023",
    "Caroline has had her current group of friends for 4 years",
    "Caroline moved from Sweden 4 years ago",
    "Caroline went to the LGBTQ conference on 10 July 2023",
    "Caroline likes reading and wants to be a counselor",
    "Caroline participated in a pride parade, school speech, and support group",
    
    # From Q&A about Melanie
    "Melanie painted a sunrise in 2022",
    "Melanie ran a charity race the sunday before 25 May 2023",
    "Melanie is planning on going camping in June 2023",
    "Melanie has read Nothing is Impossible and Charlotte's Web",
    "Melanie does running and pottery to destress",
    "Melanie went to the pottery workshop the Friday before 15 July 2023",
    "Melanie went camping the week before 27 June 2023",
    
    # Mixed person statements
    "Caroline and Melanie are friends",
    "Both Caroline and Melanie care about social causes"
]

print("Testing HotMem with Locomo10 Dataset")
print("="*60)

# Initialize processor
processor = HotPathMemoryProcessor(
    sqlite_path="test_locomo.db",
    lmdb_dir="test_locomo.lmdb",
    user_id="test-user"
)

print("\n✅ HotMem processor initialized")
print(f"Processing {len(test_statements)} statements from locomo10 dataset\n")

# Process statements
for i, statement in enumerate(test_statements, 1):
    print(f"\nStatement {i}: '{statement}'")
    
    bullets, triples = processor.hot.process_turn(
        statement,
        session_id="locomo-test",
        turn_id=i
    )
    
    if triples:
        print(f"  Extracted: {triples}")
    else:
        print("  No triples extracted")
    
    time.sleep(0.05)  # Small pause

print("\n" + "="*60)
print("QUERIES")
print("="*60)

# Test queries based on the Q&A
test_queries = [
    "What is Caroline's identity?",
    "When did Caroline go to the LGBTQ support group?",
    "What did Caroline research?",
    "Where did Caroline move from?",
    "What books has Melanie read?",
    "What does Melanie do to destress?",
    "When did Melanie paint something?",
    "Are Caroline and Melanie friends?"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    
    bullets, _ = processor.hot.process_turn(
        query,
        session_id="locomo-test",
        turn_id=len(test_statements) + test_queries.index(query) + 1
    )
    
    if bullets:
        print("  Retrieved:")
        for bullet in bullets[:3]:  # Top 3
            print(f"    {bullet}")
    else:
        print("  No relevant memories")

# Get metrics
print("\n" + "="*60)
print("METRICS")
print("="*60)

metrics = processor.hot.get_metrics()
for key, value in metrics.items():
    if isinstance(value, dict) and 'p95' in value:
        print(f"  {key}: p95={value['p95']:.1f}ms, mean={value['mean']:.1f}ms")
    else:
        print(f"  {key}: {value}")

# Cleanup
import shutil
if os.path.exists("test_locomo.db"):
    os.remove("test_locomo.db")
if os.path.exists("test_locomo.lmdb"):
    shutil.rmtree("test_locomo.lmdb")
print("\nCleaned up test files.")