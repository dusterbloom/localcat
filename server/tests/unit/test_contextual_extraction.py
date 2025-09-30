"""
Test contextual extraction granularity implementation.

Tests for Phase 1-3: Prepositional phrases, adjectives, and compound nouns.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory


def test_prep_phrase_extraction():
    """Test extraction of prepositional phrases."""
    print("\n=== Test: Prepositional Phrases ===")

    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)

    test_cases = [
        {
            "text": "I love swimming in the sea",
            "expected_triple": ("you", "love", "swimming in the sea"),
            "expected_entity": "swimming"
        },
        {
            "text": "I don't like swimming in lakes",
            "expected_triple": ("you", "like", "swimming in lakes"),
            "expected_entity": "swimming",
            "negated": True
        },
        {
            "text": "I work on AI projects",
            "expected_triple": ("you", "work_on", "ai projects"),
            "expected_entity": "projects"
        }
    ]

    for case in test_cases:
        print(f"\nTest: {case['text']}")
        entities, triples, neg_count, doc, aliases = hot._extract(case['text'], "en")

        print(f"  Triples: {triples}")
        print(f"  Entities: {entities}")
        print(f"  Aliases: {aliases}")

        # Check if expected triple exists
        found = any(
            s == case['expected_triple'][0] and
            r == case['expected_triple'][1] and
            d == case['expected_triple'][2]
            for s, r, d in triples
        )

        if found:
            print(f"  ✅ Found expected triple: {case['expected_triple']}")
        else:
            print(f"  ❌ Expected triple not found: {case['expected_triple']}")

        # Check if base entity is in entities list
        if case['expected_entity'] in entities:
            print(f"  ✅ Base entity '{case['expected_entity']}' in entities")
        else:
            print(f"  ❌ Base entity '{case['expected_entity']}' not in entities")

        # Check negation if expected
        if case.get('negated'):
            if neg_count > 0:
                print(f"  ✅ Negation detected (count: {neg_count})")
            else:
                print(f"  ❌ Negation not detected")


def test_adjective_extraction():
    """Test extraction of adjectival modifiers."""
    print("\n=== Test: Adjectives ===")

    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)

    test_cases = [
        {
            "text": "I drive a red car",
            "expected_triple": ("you", "drive", "red car"),
            "expected_entity": "car"
        },
        {
            "text": "I saw a big blue house",
            "expected_triple": ("you", "saw", "big blue house"),
            "expected_entity": "house"
        }
    ]

    for case in test_cases:
        print(f"\nTest: {case['text']}")
        entities, triples, neg_count, doc, aliases = hot._extract(case['text'], "en")

        print(f"  Triples: {triples}")
        print(f"  Entities: {entities}")
        print(f"  Aliases: {aliases}")

        # Check if expected triple exists
        found = any(
            s == case['expected_triple'][0] and
            r == case['expected_triple'][1] and
            d == case['expected_triple'][2]
            for s, r, d in triples
        )

        if found:
            print(f"  ✅ Found expected triple: {case['expected_triple']}")
        else:
            print(f"  ❌ Expected triple not found: {case['expected_triple']}")

        # Check if base entity is in entities list
        if case['expected_entity'] in entities:
            print(f"  ✅ Base entity '{case['expected_entity']}' in entities")
        else:
            print(f"  ❌ Base entity '{case['expected_entity']}' not in entities")


def test_compound_extraction():
    """Test extraction of compound nouns."""
    print("\n=== Test: Compound Nouns ===")

    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)

    test_cases = [
        {
            "text": "I study machine learning",
            "expected_triple": ("you", "study", "machine learning"),
            "expected_entity": "learning"
        },
        {
            "text": "I'm a software engineer",
            "expected_triple": ("you", "is", "software engineer"),
            "expected_entity": "engineer"
        }
    ]

    for case in test_cases:
        print(f"\nTest: {case['text']}")
        entities, triples, neg_count, doc, aliases = hot._extract(case['text'], "en")

        print(f"  Triples: {triples}")
        print(f"  Entities: {entities}")
        print(f"  Aliases: {aliases}")

        # Check if expected triple exists
        found = any(
            s == case['expected_triple'][0] and
            r == case['expected_triple'][1] and
            d == case['expected_triple'][2]
            for s, r, d in triples
        )

        if found:
            print(f"  ✅ Found expected triple: {case['expected_triple']}")
        else:
            print(f"  ❌ Expected triple not found: {case['expected_triple']}")

        # Check if base entity is in entities list
        if case['expected_entity'] in entities:
            print(f"  ✅ Base entity '{case['expected_entity']}' in entities")
        else:
            print(f"  ❌ Base entity '{case['expected_entity']}' not in entities")


def test_combined_extraction():
    """Test extraction with all three types combined."""
    print("\n=== Test: Combined (Compounds + Adjectives + Preps) ===")

    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)

    text = "I work on complex machine learning projects in San Francisco"
    print(f"\nTest: {text}")

    entities, triples, neg_count, doc, aliases = hot._extract(text, "en")

    print(f"  Triples: {triples}")
    print(f"  Entities: {entities}")
    print(f"  Aliases: {aliases}")

    # Look for enriched extraction
    found_enriched = any(
        "machine learning" in d and "complex" in d and "san francisco" in d.lower()
        for s, r, d in triples
    )

    if found_enriched:
        print(f"  ✅ Found enriched triple with all modifiers")
    else:
        print(f"  ❌ Enriched triple not found")


def test_dual_registration():
    """Test that dual registration works for entity index."""
    print("\n=== Test: Dual Registration ===")

    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)

    # Process a turn with enriched extraction
    text = "I love swimming in the sea"
    print(f"\nProcessing: {text}")

    bullets, triples = hot.process_turn(text, "test_session", 1)

    print(f"  Extracted triples: {triples}")
    print(f"  Entity aliases: {hot._entity_aliases}")

    # Check entity_index for both forms
    if "swimming" in hot.entity_index:
        edges = hot.entity_index["swimming"]
        print(f"  ✅ Base 'swimming' indexed with {len(edges)} edges")
        print(f"     Edges: {list(edges)}")
    else:
        print(f"  ❌ Base 'swimming' not in entity_index")

    if "swimming in the sea" in hot.entity_index:
        edges = hot.entity_index["swimming in the sea"]
        print(f"  ✅ Enriched 'swimming in the sea' indexed with {len(edges)} edges")
    else:
        print(f"  ❌ Enriched 'swimming in the sea' not in entity_index")


def test_base_entity_extraction():
    """Test the _extract_base_entity helper."""
    print("\n=== Test: Base Entity Extraction Heuristic ===")

    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)

    test_cases = [
        ("swimming in the sea", "swimming"),
        ("red car", "car"),
        ("machine learning", "learning"),
        ("complex machine learning projects", "projects"),
        ("meeting on tuesday", "meeting"),
        ("car", "car"),  # Single word
    ]

    for enriched, expected_base in test_cases:
        result = hot._extract_base_entity(enriched)
        if result == expected_base:
            print(f"  ✅ '{enriched}' -> '{result}'")
        else:
            print(f"  ❌ '{enriched}' -> '{result}' (expected '{expected_base}')")


if __name__ == "__main__":
    print("Testing Contextual Extraction Granularity Implementation")
    print("=" * 60)

    test_prep_phrase_extraction()
    test_adjective_extraction()
    test_compound_extraction()
    test_combined_extraction()
    test_dual_registration()
    test_base_entity_extraction()

    print("\n" + "=" * 60)
    print("All tests completed!")