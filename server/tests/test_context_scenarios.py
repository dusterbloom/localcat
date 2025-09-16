#!/usr/bin/env python3
"""
Test different conversation scenarios to verify progressive context
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.context.context_orchestrator import pack_context

def test_scenario(name, query, memory_bullets, summary, expected_memory):
    """Test a specific conversation scenario"""

    base_messages = [
        {"role": "system", "content": "You are Locat, a personal assistant."},
        {"role": "user", "content": query}
    ]

    messages, stats = pack_context(
        messages=base_messages,
        memory_bullets=memory_bullets,
        summary_text=summary,
        budget_tokens=4096,
        progressive_mode=True
    )

    has_memory = any("Memory Context:" in msg.get('content', '') for msg in messages)

    print(f"🔍 {name}")
    print(f"   Query: '{query}'")
    print(f"   Expected memory: {expected_memory}")
    print(f"   Got memory: {has_memory}")
    print(f"   Total messages: {len(messages)}")
    print(f"   Token breakdown: sys={stats['tokens_system']}, mem={stats['tokens_memory']}, dlg={stats['tokens_dialogue']}")

    if has_memory == expected_memory:
        print(f"   ✅ PASS")
    else:
        print(f"   ❌ FAIL")

    print()

def main():
    print("=== Progressive Context Scenario Testing ===\n")

    # Conversation memory (facts from previous discussions)
    conversation_memory = [
        "• User's name is Peppi",
        "• Lives in Sardinia, Italy",
        "• Discussed local AI concepts",
        "• Interested in Mediterranean culture"
    ]

    summary = "Previous conversation about local AI and Sardinia"

    # Test scenarios that SHOULD NOT trigger memory
    print("🚫 Scenarios that should NOT use memory:\n")

    test_scenario(
        "Pure greeting",
        "Hello!",
        conversation_memory,
        summary,
        expected_memory=False  # Memory available but query doesn't need it
    )

    test_scenario(
        "General knowledge question",
        "What is 2+2?",
        conversation_memory,
        summary,
        expected_memory=False  # Math doesn't need personal memory
    )

    test_scenario(
        "New topic - general request",
        "Can you help me write a Python function?",
        conversation_memory,
        summary,
        expected_memory=False  # Programming help doesn't need personal context
    )

    # Test scenarios that SHOULD trigger memory
    print("✅ Scenarios that should USE memory:\n")

    test_scenario(
        "Question about previous topic",
        "Tell me more about the local AI we discussed",
        conversation_memory,
        summary,
        expected_memory=True  # References previous conversation
    )

    test_scenario(
        "Personal question",
        "What's my name again?",
        conversation_memory,
        summary,
        expected_memory=True  # Asking about personal info
    )

    test_scenario(
        "Follow-up on personal fact",
        "Tell me about where I live",
        conversation_memory,
        summary,
        expected_memory=True  # Personal location info
    )

    test_scenario(
        "Cultural reference",
        "Any interesting facts about Mediterranean culture?",
        conversation_memory,
        summary,
        expected_memory=True  # Related to user's interests
    )

    # Test empty memory scenarios
    print("🆕 Scenarios with NO prior memory:\n")

    test_scenario(
        "First conversation - greeting",
        "Hi there!",
        [],  # No memory
        None,  # No summary
        expected_memory=False  # Should be clean, minimal
    )

    test_scenario(
        "First conversation - question",
        "Can you help me with something?",
        [],  # No memory
        None,  # No summary
        expected_memory=False  # No memory to inject
    )

if __name__ == "__main__":
    main()