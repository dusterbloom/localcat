#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '.')

# Force production database path
os.environ['MEMORY_DB_PATH'] = '/Users/peppi/Library/Application Support/LocalCat/data/memory.db'
os.environ['MEMORY_LMDB_PATH'] = '/Users/peppi/Library/Application Support/LocalCat/data/memory.lmdb'

from core.memory.hotpath_processor import HotPathMemoryProcessor

hotpath = HotPathMemoryProcessor(user_id="fantastic", config=None)

print("=" * 80)
print("PRODUCTION DATABASE - Entity Index State")
print("=" * 80)
print(f"\nTotal entities in index: {len(hotpath.hot.entity_index)}")
print(f"\nEntity index keys (first 30): {list(hotpath.hot.entity_index.keys())[:30]}")

if "you" in hotpath.hot.entity_index:
    edges = list(hotpath.hot.entity_index["you"])
    print(f"\n'you' entity has {len(edges)} edges (showing first 15):")
    for triple in edges[:15]:
        print(f"  {triple}")

if "favorite food" in hotpath.hot.entity_index:
    edges = list(hotpath.hot.entity_index["favorite food"])
    print(f"\n'favorite food' entity has {len(edges)} edges:")
    for triple in edges:
        print(f"  {triple}")
else:
    print("\n❌ 'favorite food' NOT in entity_index!")

print("\n" + "=" * 80)
print("Database check")
print("=" * 80)
import sqlite3
conn = sqlite3.connect('/Users/peppi/Library/Application Support/LocalCat/data/memory.db')
cursor = conn.cursor()
cursor.execute("SELECT src, rel, dst, status, pos FROM edge WHERE (src LIKE '%food%' OR dst LIKE '%food%') AND status = 1")
results = cursor.fetchall()
print(f"\nFood edges in DATABASE:")
for row in results:
    print(f"  {row}")

