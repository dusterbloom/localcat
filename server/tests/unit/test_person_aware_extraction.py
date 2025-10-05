#!/usr/bin/env python3
"""
Test Person-aware extraction using Universal Dependencies Person feature.

Tests that the system correctly distinguishes:
- Person=1 (I, me, my) → User talking about themselves → Store as user facts
- Person=2 (you, your) → User talking to/about AI → Skip/Don't store
- Person=3 (he, she, they) → User talking about others → Store as third-party facts
"""

import os
import sys
import tempfile

# Set up test environment
temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MEMORY_SQLITE_PATH"] = temp_db.name

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory


def test_person_first_user_facts():
    """Test Person=1: User talking about themselves should be stored"""
    print("\n=== TEST: Person=1 (First Person) ===")

    paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store, max_recency=50)
    hot.prewarm("en")

    test_cases = [
        ("I live in Sardinia", [("you", "lives_in", "sardinia")]),
        ("My name is Alice", [("you", "name", "alice")]),
        ("I work at Microsoft", [("you", "works_at", "microsoft")]),
    ]

    all_passed = True
    for text, expected_triples in test_cases:
        bullets, triples = hot.process_turn(text, "test-session", 1)

        # Normalize for comparison
        normalized_triples = [(s.lower(), r.lower(), d.lower()) for s, r, d in triples]
        expected_normalized = [(s.lower(), r.lower(), d.lower()) for s, r, d in expected_triples]

        if normalized_triples == expected_normalized:
            print(f"  ✅ '{text}' → {triples}")
        else:
            print(f"  ❌ '{text}'")
            print(f"     Expected: {expected_triples}")
            print(f"     Got: {triples}")
            all_passed = False

    return all_passed


def test_person_second_ai_context():
    """Test Person=2: User talking to/about AI should NOT be stored as user facts"""
    print("\n=== TEST: Person=2 (Second Person - AI Context) ===")

    paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store, max_recency=50)
    hot.prewarm("en")

    test_cases = [
        "You live in a computer",
        "Your name is Locat",
        "You are an AI assistant",
    ]

    all_passed = True
    for text in test_cases:
        bullets, triples = hot.process_turn(text, "test-session", 1)

        # Should NOT extract triples with subject="you" (user)
        user_facts = [(s, r, d) for s, r, d in triples if s.lower() == "you"]

        if len(user_facts) == 0:
            print(f"  ✅ '{text}' → Correctly skipped (no user facts)")
        else:
            print(f"  ❌ '{text}' → Incorrectly stored as user fact: {user_facts}")
            all_passed = False

    return all_passed


def test_person_third_other_people():
    """Test Person=3: User talking about others should be stored"""
    print("\n=== TEST: Person=3 (Third Person) ===")

    paths = Paths(sqlite_path=temp_db.name, lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store, max_recency=50)
    hot.prewarm("en")

    test_cases = [
        ("He lives in Rome", [("he", "lives_in", "rome")]),
        ("She works at Google", [("she", "works_at", "google")]),
        ("Alice is a teacher", [("alice", "is", "teacher")]),
    ]

    all_passed = True
    for text, expected_pattern in test_cases:
        bullets, triples = hot.process_turn(text, "test-session", 1)

        # Check that third-person facts are extracted
        normalized_triples = [(s.lower(), r.lower(), d.lower()) for s, r, d in triples]

        # Check if any triple matches the expected pattern
        matched = any(
            s == expected_pattern[0][0] and
            r == expected_pattern[0][1] and
            d == expected_pattern[0][2]
            for s, r, d in normalized_triples
        )

        if matched:
            print(f"  ✅ '{text}' → {triples}")
        else:
            print(f"  ❌ '{text}'")
            print(f"     Expected pattern: {expected_pattern}")
            print(f"     Got: {triples}")
            all_passed = False

    return all_passed


def run_all_tests():
    """Run all Person-aware extraction tests"""
    print("="*70)
    print("PERSON-AWARE EXTRACTION TESTS")
    print("="*70)

    results = {
        "Person=1 (User facts)": test_person_first_user_facts(),
        "Person=2 (AI context - skip)": test_person_second_ai_context(),
        "Person=3 (Third-party facts)": test_person_third_other_people(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")

    # Cleanup
    os.unlink(temp_db.name)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
