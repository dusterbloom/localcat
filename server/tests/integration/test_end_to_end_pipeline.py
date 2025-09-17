#!/usr/bin/env python3
"""
End-to-End Test: Intent → Extraction → Retrieval → Context Packing
Tests the complete flow from bot.py through to LLM context
"""

import os
import sys
from pathlib import Path

# Add server to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Configure environment
os.environ["HOTMEM_SQLITE"] = "../data/memory.db"
os.environ["HOTMEM_LMDB_DIR"] = "../data/graph.lmdb"
os.environ["USER_ID"] = "peppi"
os.environ["HOTMEM_USE_RULE_BASED_INTENT"] = "true"
os.environ["HOTMEM_ENABLE_METRICS"] = "true"
os.environ["CONTEXT_PROGRESSIVE_MODE"] = "true"

from components.processing.hotpath_processor import HotPathMemoryProcessor
from components.context.context_orchestrator import pack_context
from components.context.memory_config import get_global_config

print("=" * 80)
print("END-TO-END TEST: Intent → Extraction → Retrieval → Context Packing")
print("=" * 80)

# Initialize processor like bot.py does
memory = HotPathMemoryProcessor(
    sqlite_path="../data/memory.db",
    lmdb_dir="../data/graph.lmdb",
    user_id="peppi",
    enable_metrics=True,
    context_aggregator=None  # We'll test without for now
)

# Test cases: Questions (should retrieve, not store) and Facts (should store)
test_cases = [
    {
        "type": "QUESTION",
        "text": "What do you know about my dog?",
        "expected": {
            "intent": "pure_question",
            "extracts_facts": False,
            "retrieves": True,
            "entities_contain": ["dog", "you"]
        }
    },
    {
        "type": "QUESTION",
        "text": "How old is Potola?",
        "expected": {
            "intent": "pure_question",
            "extracts_facts": False,
            "retrieves": True,
            "entities_contain": ["potola"]
        }
    },
    {
        "type": "FACT",
        "text": "My dog Potola is 5 years old",
        "expected": {
            "intent": "fact_statement",
            "extracts_facts": True,
            "retrieves": False,
            "triples_contain": [("dog", "age")]  # Simplified check
        }
    },
    {
        "type": "QUESTION",
        "text": "Where do I live?",
        "expected": {
            "intent": "pure_question",
            "extracts_facts": False,
            "retrieves": True,
            "entities_contain": ["you"]
        }
    }
]

print("\n📊 Running end-to-end tests...\n")

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['type']} - \"{test['text']}\"")
    print("-" * 60)

    # Process the turn
    result = memory.hot.process_turn(
        text=test['text'],
        session_id=f"test_{i}",
        turn_id=1,
        user_id="peppi"
    )

    # Check intent classification
    intent = result.intent.intent.value if result.intent else "unknown"
    print(f"✓ Intent: {intent} {'✅' if intent == test['expected']['intent'] else '❌'}")

    # Check fact extraction
    facts_extracted = len(result.triples) > 0
    should_extract = test['expected']['extracts_facts']
    print(f"✓ Extracts facts: {facts_extracted} {'✅' if facts_extracted == should_extract else '❌'}")
    if facts_extracted:
        print(f"  Triples: {result.triples[:3]}")

    # Check retrieval
    retrieved = len(result.bullets) > 0
    should_retrieve = test['expected']['retrieves']
    print(f"✓ Retrieves: {retrieved} {'✅' if retrieved == should_retrieve else '❌'}")
    if retrieved:
        print(f"  Bullets ({len(result.bullets)}): {result.bullets[:3]}")

    # Check entity extraction for questions
    if test['type'] == "QUESTION" and 'entities_contain' in test['expected']:
        # Get entities from the expansion (stored in result)
        entities = result.expanded_entities if hasattr(result, 'expanded_entities') else []
        entities_ok = any(e in str(entities).lower() for e in test['expected']['entities_contain'])
        print(f"✓ Entity extraction: {'✅' if entities_ok else '❌'}")
        if entities:
            print(f"  Entities: {entities[:5]}")

    print()

# Test context packing
print("\n" + "=" * 80)
print("CONTEXT PACKING TEST")
print("=" * 80)

# Simulate what would be injected into LLM context
config = get_global_config()

# Get some bullets from a real query
test_result = memory.hot.process_turn(
    text="What do you know about my dog?",
    session_id="context_test",
    turn_id=1,
    user_id="peppi"
)

if test_result.bullets:
    # Format bullets for context
    memory_bullets = [f"• {b}" if isinstance(b, str) else f"• {b[0]} {b[1]} {b[2]}"
                     for b in test_result.bullets[:5]]

    # Pack context like HotPathMemoryProcessor does
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What do you know about my dog?"}
    ]

    packed_messages, stats = pack_context(
        messages=messages,
        memory_bullets=memory_bullets,
        summary_text=None,
        budget_tokens=config.budget_tokens,
        inject_role=config.inject_role,
        inject_header=config.inject_header,
        system_hint=None,
        progressive_mode=config.progressive_mode
    )

    print("\n📦 Context Packing Result:")
    print(f"  Original messages: {len(messages)}")
    print(f"  Packed messages: {len(packed_messages)}")
    print(f"  Memory bullets injected: {len(memory_bullets)}")
    if stats:
        print(f"  Stats: {stats}")

    # Show the injected context
    for msg in packed_messages:
        if "factual context" in msg.get("content", "").lower():
            print(f"\n📝 Injected Memory Context:")
            print(msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"])
            break

print("\n" + "=" * 80)
print("✅ END-TO-END TEST COMPLETE")
print("=" * 80)