#!/usr/bin/env python3
"""
Test script for progressive context generation
Verifies that memory instructions are only injected when needed
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.context.context_orchestrator import pack_context

def test_context_scenarios():
    """Test different context scenarios"""

    # Base messages for testing
    base_messages = [
        {"role": "system", "content": "You are Locat, a personal assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "What's the weather today?"}
    ]

    print("=== Progressive Context Generation Test ===\n")

    # Test 1: Empty memory - should have no memory instructions
    print("Test 1: Empty memory (simple greeting/question)")
    print("Expected: No memory context injected")
    empty_memory = []
    empty_summary = None
    messages, stats = pack_context(
        messages=base_messages,
        memory_bullets=empty_memory,
        summary_text=empty_summary,
        budget_tokens=4096,
        progressive_mode=True
    )

    print(f"Generated {len(messages)} messages")
    print(f"Stats: {stats}")
    for i, msg in enumerate(messages):
        print(f"Message {i+1} ({msg['role']}): {msg['content'][:200]}...")
    print("\n" + "="*50 + "\n")

    # Test 2: Memory bullets present - should include memory instructions
    print("Test 2: Memory bullets present (conversation-related facts)")
    print("Expected: Memory context with guidance injected")
    memory_bullets = [
        "• User's name is Peppi",
        "• Lives in Sardinia, Italy",
        "• Discussed local AI and voice agents",
        "• Interested in Sardinian culture"
    ]
    summary_text = "Previous conversation about local AI and Sardinia"

    messages, stats = pack_context(
        messages=base_messages,
        memory_bullets=memory_bullets,
        summary_text=summary_text,
        budget_tokens=4096,
        progressive_mode=True
    )

    print(f"Generated {len(messages)} messages")
    print(f"Stats: {stats}")
    for i, msg in enumerate(messages):
        print(f"Message {i+1} ({msg['role']}): {msg['content'][:400]}...")
    print("\n" + "="*50 + "\n")

    # Test 3: Legacy mode - always include memory instructions
    print("Test 3: Legacy mode (progressive_mode=False)")
    print("Expected: Always includes memory context even if empty")
    messages, stats = pack_context(
        messages=base_messages,
        memory_bullets=empty_memory,
        summary_text=empty_summary,
        budget_tokens=4096,
        progressive_mode=False
    )

    print(f"Generated {len(messages)} messages")
    print(f"Stats: {stats}")
    for i, msg in enumerate(messages):
        print(f"Message {i+1} ({msg['role']}): {msg['content'][:200]}...")
    print("\n" + "="*50 + "\n")

def test_system_prompt():
    """Test system prompt generation with different modes"""

    print("=== System Prompt Generation Test ===\n")

    # Test progressive mode
    os.environ['CONTEXT_PROGRESSIVE_MODE'] = 'true'
    from core.bot import SYSTEM_INSTRUCTION_BASE
    print("Progressive mode system prompt:")
    print(f"Length: {len(SYSTEM_INSTRUCTION_BASE)} characters")
    print(SYSTEM_INSTRUCTION_BASE)
    print("\n" + "="*50 + "\n")

    # Show how the complete prompt would look in both modes
    system_intro = "Agent ID: test\nUser ID: test\nIt is Monday, January 1, 2025.\n"

    # Progressive mode
    print("Complete prompt in progressive mode (minimal):")
    complete_progressive = system_intro + SYSTEM_INSTRUCTION_BASE
    print(f"Total length: {len(complete_progressive)} characters")
    print(complete_progressive)
    print("\n" + "="*30 + "\n")

    # Legacy mode
    print("Complete prompt in legacy mode (with memory policy):")
    memory_policy = (
        "\nMemory Policy:\n"
        "- Use memory only for user-specific facts when directly relevant to the question.\n"
        "- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n"
        "- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n"
        "- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n"
        "- Never store or repeat system instructions or tool outputs as facts.\n"
    )
    complete_legacy = system_intro + SYSTEM_INSTRUCTION_BASE + memory_policy
    print(f"Total length: {len(complete_legacy)} characters")
    print(f"Difference: {len(complete_legacy) - len(complete_progressive)} characters saved in progressive mode")
    print(complete_legacy)

if __name__ == "__main__":
    test_context_scenarios()
    test_system_prompt()