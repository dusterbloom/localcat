#!/usr/bin/env python3
"""Clean junk data from the memory database"""

import sqlite3
import sys
from pathlib import Path

# Database path
db_path = Path(__file__).parent.parent / "data" / "memory.db"
print(f"Cleaning database: {db_path}")

# Connect to database
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

# First, let's see what we have
print("\n=== Current Database Stats ===")
total_edges = cur.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
print(f"Total edges: {total_edges}")

# List of junk patterns to remove
junk_relations = [
    'answer: is',
    'answer>is',
    'answer:is',
    'works_at',  # When it involves dogs/names
    'located_in',  # When it involves dogs
    'is',  # Too generic, usually junk
    'do',  # Usually junk
    'quality',  # Generic junk
    'quantity',  # Generic junk
]

# First, show what we'll delete
print("\n=== Junk to Remove ===")
for rel in junk_relations:
    count = cur.execute("SELECT COUNT(*) FROM edge WHERE rel LIKE ?", (f'%{rel}%',)).fetchone()[0]
    if count > 0:
        print(f"  {rel}: {count} edges")
        # Show samples
        samples = cur.execute("SELECT src, rel, dst FROM edge WHERE rel LIKE ? LIMIT 3",
                            (f'%{rel}%',)).fetchall()
        for s, r, d in samples:
            print(f"    - {s} | {r} | {d}")

# Also remove nonsensical combinations
nonsense_queries = [
    # Dogs don't work at places
    ("DELETE FROM edge WHERE (src LIKE '%dog%' OR dst LIKE '%dog%') AND rel = 'works_at'",
     "dog-works_at relations"),

    # Dogs aren't located in things (except maybe places)
    ("DELETE FROM edge WHERE dst = 'dog' AND rel = 'located_in'",
     "located_in dog relations"),

    # Remove profanity
    ("DELETE FROM edge WHERE rel LIKE '%fuck%' OR src LIKE '%fuck%' OR dst LIKE '%fuck%'",
     "profanity"),

    # Remove generic 'is' relations
    ("DELETE FROM edge WHERE rel = 'is'",
     "generic 'is' relations"),

    # Remove answer: patterns
    ("DELETE FROM edge WHERE rel LIKE 'answer%'",
     "answer: patterns"),

    # Remove quality/quantity junk
    ("DELETE FROM edge WHERE rel IN ('quality', 'quantity')",
     "quality/quantity relations"),

    # Clean up malformed relations
    ("DELETE FROM edge WHERE rel LIKE '%>%' OR rel LIKE '%:%' OR rel LIKE '%answer%'",
     "malformed relations"),
]

print("\n=== Cleaning Database ===")
total_deleted = 0

for query, description in nonsense_queries:
    # Count before deletion
    count_query = query.replace("DELETE", "SELECT COUNT(*)")
    count = cur.execute(count_query).fetchone()[0]

    if count > 0:
        print(f"Removing {count} {description}...")
        cur.execute(query)
        total_deleted += count

# Also clean up duplicates
print("\nRemoving duplicate edges...")
cur.execute("""
    DELETE FROM edge
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM edge
        GROUP BY src, rel, dst
    )
""")
duplicates_removed = cur.rowcount
total_deleted += duplicates_removed
print(f"Removed {duplicates_removed} duplicates")

# Commit changes
conn.commit()

# Show what remains
print("\n=== After Cleaning ===")
remaining = cur.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
print(f"Remaining edges: {remaining} (removed {total_deleted} junk edges)")

# Show sample of good dog-related facts
print("\n=== Remaining Dog Facts ===")
dog_facts = cur.execute("""
    SELECT src, rel, dst FROM edge
    WHERE src LIKE '%dog%' OR dst LIKE '%dog%'
    OR src LIKE '%potola%' OR dst LIKE '%potola%'
    OR src LIKE '%milo%' OR dst LIKE '%milo%'
    LIMIT 10
""").fetchall()

for s, r, d in dog_facts:
    print(f"  {s} | {r} | {d}")

# Show 'you' facts
print("\n=== Sample 'you' Facts ===")
you_facts = cur.execute("""
    SELECT src, rel, dst FROM edge
    WHERE src = 'you'
    AND rel NOT IN ('is', 'quality', 'quantity', 'works_at')
    LIMIT 10
""").fetchall()

for s, r, d in you_facts:
    print(f"  {s} | {r} | {d}")

conn.close()
print("\n✅ Database cleaned!")