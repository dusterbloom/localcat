#!/usr/bin/env python3
"""
Generate synthetic test data for evaluation

Creates realistic conversation data with known ground truth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory


def generate_test_data(db_path: str):
    """Generate synthetic test conversations"""

    # Create in-memory store or use provided path
    store = MemoryStore(Paths(sqlite_path=db_path, lmdb_dir=None))
    hot = HotMemory(store)

    print(f"Generating test data in: {db_path}")

    # Session 1: User introduces themselves multiple times (high confidence)
    session1 = "session-1"
    conversations1 = [
        "My name is Alice",
        "I'm Alice",
        "Call me Alice",
        "I work at Google",
        "I work at Google in Mountain View",
        "I love Python programming",
        "I really enjoy Python",
    ]

    for i, text in enumerate(conversations1):
        hot.process_turn(text, session1, i)

    # Session 2: User changes their mind (conflicting info)
    session2 = "session-2"
    conversations2 = [
        "I live in San Francisco",
        "Actually, I live in Oakland now",  # Conflict
        "I have a cat named Whiskers",
    ]

    for i, text in enumerate(conversations2):
        hot.process_turn(text, session2, i)

    # Session 3: Uncertain statements (hedging)
    session3 = "session-3"
    conversations3 = [
        "I think I like coffee",
        "Maybe I enjoy hiking",
        "I'm not sure but I might play guitar",
    ]

    for i, text in enumerate(conversations3):
        hot.process_turn(text, session3, i)

    # Session 4: Confident statements
    session4 = "session-4"
    conversations4 = [
        "My favorite color is blue",
        "My favorite color is definitely blue",
        "I absolutely love the color blue",
    ]

    for i, text in enumerate(conversations4):
        hot.process_turn(text, session4, i)

    # Session 5: Questions (should not create edges)
    session5 = "session-5"
    conversations5 = [
        "What is your name?",
        "Where do you live?",
        "Do you like programming?",
    ]

    for i, text in enumerate(conversations5):
        hot.process_turn(text, session5, i)

    # Flush all data
    store.flush_if_needed(max_ops=1)

    # Print statistics
    cur = store.sql.cursor()

    edge_count = cur.execute("SELECT COUNT(*) FROM edge WHERE status=1").fetchone()[0]
    turn_count = cur.execute("SELECT COUNT(*) FROM conversation_turn").fetchone()[0]
    source_count = cur.execute("SELECT COUNT(*) FROM edge_source").fetchone()[0]

    print(f"\n✅ Test data generated:")
    print(f"   Edges: {edge_count}")
    print(f"   Turns: {turn_count}")
    print(f"   Edge-Source links: {source_count}")

    # Show sample edges
    print(f"\nSample edges:")
    edges = cur.execute("""
        SELECT src, rel, dst, weight, pos, neg
        FROM edge
        WHERE status=1
        LIMIT 5
    """).fetchall()

    for src, rel, dst, weight, pos, neg in edges:
        print(f"   ({src}, {rel}, {dst}) - weight={weight:.2f}, pos={pos}, neg={neg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate test data")
    parser.add_argument("--db", default="data/test_memory.db", help="Database path")
    args = parser.parse_args()

    db_path = Path(__file__).parent.parent / args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    generate_test_data(str(db_path))