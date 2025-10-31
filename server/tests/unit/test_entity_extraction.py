#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '.')
os.environ['MEMORY_DB_PATH'] = '/Users/peppi/Library/Application Support/LocalCat/data/memory.db'
os.environ['MEMORY_LMDB_PATH'] = '/Users/peppi/Library/Application Support/LocalCat/data/memory.lmdb'

from core.memory.hotpath_processor import HotPathMemoryProcessor

hotpath = HotPathMemoryProcessor(user_id="fantastic", config=None)

query = "What's my favorite food?"
print(f"Query: {query}\n")

# Extract entities like the system would
entities, _, _, _, _ = hotpath.hot._cached_extract(query, "en")
entities = hotpath.hot.extractor.refine_entities(query, entities)

print(f"Extracted entities: {entities}\n")

# Check what each entity would retrieve
for entity in entities[:10]:
    if entity in hotpath.hot.entity_index:
        edges = list(hotpath.hot.entity_index[entity])
        print(f"Entity '{entity}' → {len(edges)} edges:")
        for s, r, d in edges[:5]:
            print(f"  ({s}, {r}, {d})")
    else:
        print(f"Entity '{entity}' → NOT in index")
    print()

# Now simulate full retrieval with these entities
print("=" * 60)
print("SIMULATED RETRIEVAL (with entity extraction)")
print("=" * 60)

bullets_from_entities = []
for entity in entities[:5]:
    if entity in hotpath.hot.entity_index:
        for s, r, d in list(hotpath.hot.entity_index[entity])[:3]:
            if r == "has":
                bullet = f"{s} has {d}"
            elif r == "is":
                bullet = f"{s} is {d}"
            else:
                bullet = f"{s} {r.replace('_', ' ')} {d}"
            bullets_from_entities.append(bullet)

print(f"\nWould return {len(bullets_from_entities)} bullets:")
for i, bullet in enumerate(bullets_from_entities, 1):
    print(f"  {i}. {bullet}")

print("\n" + "=" * 60)
print("ANSWER TO 'What's my favorite food?'")
print("=" * 60)
if any("steak" in b.lower() for b in bullets_from_entities):
    print("✅ WOULD FIND: steak")
else:
    print("❌ WOULD NOT find the answer")

