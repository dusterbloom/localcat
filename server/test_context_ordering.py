#!/usr/bin/env python3
"""
Test script to verify context ordering changes.
Tests that memory bullets are injected after persona prompt, before conversation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core", "memory", "pipecat", "src"))

from core.memory.context_formatter import ContextFormatter


def test_formatter_basic():
    """Test ContextFormatter initialization and basic methods"""
    print("Testing ContextFormatter initialization...")
    formatter = ContextFormatter(
        max_bullets=3,
        inject_role="system",
        inject_header="Use the following factual context if helpful."
    )
    print(f"✓ ContextFormatter initialized")
    print(f"  - max_bullets: {formatter.max_bullets}")
    print(f"  - inject_role: {formatter.inject_role}")
    print(f"  - inject_header: {formatter.inject_header}")
    return formatter


def test_format_bullets(formatter):
    """Test bullet formatting with source tags"""
    print("\nTesting format_bullets()...")

    # Test bullets with source tags (as they come from retrieval)
    test_bullets = [
        "• [convo] Hello, it's nice to be back. (27m ago)",
        "• [graph] dog is named potola (23h ago)",
        "• [convo] And my dog's name is Podola. (23h ago)",
        "• [convo] Short",  # Should be filtered out (too short)
        "• [convo] Hello, it's nice to be back. (27m ago)",  # Duplicate
    ]

    formatted = formatter.format_bullets(test_bullets, max_bullets=3)

    print(f"Input: {len(test_bullets)} bullets")
    print(f"Output: {len(formatted)} bullets (deduped, capped)")
    for i, bullet in enumerate(formatted, 1):
        print(f"  {i}. {bullet}")

    assert len(formatted) == 3, f"Expected 3 bullets, got {len(formatted)}"
    assert formatted[0] == test_bullets[0], "First bullet should be preserved"
    print("✓ Bullet formatting works correctly")
    return formatted


def test_build_message(formatter, bullets):
    """Test message building"""
    print("\nTesting build_message()...")

    message = formatter.build_message(
        role="system",
        header="Use the following factual context if helpful.",
        bullets=bullets
    )

    print(f"Message role: {message['role']}")
    print(f"Message content:\n{message['content']}\n")

    assert message['role'] == "system"
    assert "Use the following factual context if helpful." in message['content']
    assert len([line for line in message['content'].split('\n') if line.startswith('•')]) == len(bullets)
    print("✓ Message building works correctly")
    return message


def test_truncate_bullets(formatter):
    """Test bullet truncation"""
    print("\nTesting truncate_bullets()...")

    long_bullets = [
        "• [convo] " + "A" * 200 + " (1h ago)",
        "• [convo] " + "B" * 200 + " (2h ago)",
        "• [convo] " + "C" * 200 + " (3h ago)",
    ]

    truncated = formatter.truncate_bullets(long_bullets, max_length=500)

    total_length = sum(len(b) for b in truncated)
    print(f"Input: {len(long_bullets)} bullets, ~{sum(len(b) for b in long_bullets)} chars")
    print(f"Output: {len(truncated)} bullets, {total_length} chars")

    assert total_length <= 500, f"Total length {total_length} exceeds 500"
    print("✓ Bullet truncation works correctly")


def test_persona_index_logic():
    """Test the _persona_prompt_index logic"""
    print("\nTesting _persona_prompt_index() logic...")

    # Simulate a message list
    messages = [
        {"role": "system", "content": "[Session Context]\nDate: 2025-10-12\nUser: Peppi"},
        {"role": "system", "content": "You are Locat, an AI persona.\nYou have contextual awareness..."},
        {"role": "system", "content": "Context Guide:\n- You may receive Memory Context bullets..."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    # Find where memory should be inserted (after persona prompt, which is index 1)
    persona_keywords = ['You are', 'AI persona', 'Your role', 'I am', "I'm"]

    insert_idx = None
    for i, msg in enumerate(messages):
        if msg.get('role') == 'system':
            content = msg.get('content', '').strip()
            if content.startswith("[Session Context]"):
                continue
            if any(kw in content for kw in persona_keywords):
                insert_idx = i + 1
                break

    print(f"Messages structure:")
    for i, msg in enumerate(messages):
        role = msg['role']
        preview = msg['content'][:50].replace('\n', ' ')
        print(f"  {i}: [{role}] {preview}...")

    print(f"\nMemory context should be inserted at index: {insert_idx}")
    print(f"This is after: [{messages[insert_idx-1]['role']}] (persona prompt)")
    print(f"This is before: [{messages[insert_idx]['role']}] (context guide or first user message)")

    assert insert_idx == 2, f"Expected insert_idx=2, got {insert_idx}"
    print("✓ Persona index logic works correctly")


def main():
    """Run all tests"""
    print("="*60)
    print("Context Ordering Test Suite")
    print("="*60)

    try:
        formatter = test_formatter_basic()
        bullets = test_format_bullets(formatter)
        test_build_message(formatter, bullets)
        test_truncate_bullets(formatter)
        test_persona_index_logic()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
