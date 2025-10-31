#!/usr/bin/env python3
"""
Debug test: What happens when we ask "What's my favorite food?"
Let's see step by step what retrieval returns.
"""
import sys
sys.path.insert(0, '.')

from core.memory.hotpath_processor import HotPathMemoryProcessor
from loguru import logger

# Initialize with production database
hotpath = HotPathMemoryProcessor(
    user_id="fantastic",
    config=None
)

print("=" * 80)
print("DATABASE STATE")
print("=" * 80)

# Check what's in entity_index
print(f"\nEntity index keys: {list(hotpath.hot.entity_index.keys())[:20]}")

if "you" in hotpath.hot.entity_index:
    print(f"\n'you' entity edges:")
    for triple in list(hotpath.hot.entity_index["you"])[:10]:
        print(f"  {triple}")

if "favorite food" in hotpath.hot.entity_index:
    print(f"\n'favorite food' entity edges:")
    for triple in hotpath.hot.entity_index["favorite food"]:
        print(f"  {triple}")

print("\n" + "=" * 80)
print("TEST 1: Current behavior (question = no entities)")
print("=" * 80)

query = "What's my favorite food?"
print(f"\nQuery: {query}")

# Simulate current behavior: question skips extraction
entities = []
print(f"Entities extracted: {entities}")

# Call retrieve with empty entities
bullets = hotpath.hot.retrieve_bullets(query, read_only=True)
print(f"\nRetrieved {len(bullets)} bullets:")
for i, bullet in enumerate(bullets, 1):
    print(f"  {i}. {bullet}")

print("\n" + "=" * 80)
print("TEST 2: If we extract entities from question")
print("=" * 80)

# Extract entities from the question
from core.memory.memory_hotpath import MemoryHotPath
entities, _, _, _, _ = hotpath.hot._cached_extract(query, "en")
entities = hotpath.hot.extractor.refine_entities(query, entities)
print(f"\nExtracted entities: {entities}")

# Now retrieve WITH entities
# Note: retrieve_bullets doesn't take entities parameter, it extracts internally
# So we need to check what retrieval.retrieve() would get

print(f"\nWhat graph traversal would find from these entities:")
for entity in entities[:5]:
    if entity in hotpath.hot.entity_index:
        print(f"\n  From '{entity}':")
        for triple in list(hotpath.hot.entity_index[entity])[:3]:
            s, r, d = triple
            humanized = f"{s} {r.replace('_', ' ')} {d}"
            print(f"    - {humanized}")
            
            # Check if we can follow to second hop
            if d in hotpath.hot.entity_index:
                print(f"      Second hop from '{d}':")
                for triple2 in list(hotpath.hot.entity_index[d])[:2]:
                    s2, r2, d2 = triple2
                    humanized2 = f"{s2} {r2.replace('_', ' ')} {d2}"
                    print(f"        - {humanized2}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
The issue: Even WITH entity extraction, we only get the FIRST hop:
  "you have favorite food"

But we NEED the SECOND hop to answer the question:
  "favorite food is steak"

The system needs to:
1. Extract entities from question
2. Do multi-hop graph traversal
3. Return BOTH hops to answer "What's MY favorite food?" → "steak"
""")

