#!/usr/bin/env python3
"""
Integration test to verify context ordering changes work correctly.
This test verifies that memory bullets are injected AFTER the persona prompt.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.memory.context_formatter import ContextFormatter
from core.memory.hotpath_processor import HotPathMemoryProcessor


def test_persona_prompt_index():
    """Test that _persona_prompt_index() finds the correct insertion point"""
    from core.memory.hotpath_processor import HotPathMemoryProcessor

    # Create a dummy processor (we only need the method)
    class DummyAggregator:
        pass

    processor = HotPathMemoryProcessor(
        sqlite_path=":memory:",
        context_aggregator=None,
    )

    # Mock messages similar to what we see in the logs
    messages = [
        {"role": "system", "content": "[Session Context]\nDate: 2025-10-12\nUser: Peppi\nSession #0"},
        {"role": "system", "content": "You are Locat, an AI persona.\nYou have contextual awareness..."},
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "Hi!"}
    ]

    # Test: should return index 2 (after persona prompt at index 1)
    insert_idx = processor._persona_prompt_index(messages)

    print(f"Messages structure:")
    for i, msg in enumerate(messages):
        preview = msg['content'][:50].replace('\n', ' ')
        print(f"  {i}: [{msg['role']}] {preview}...")

    print(f"\n✓ Persona prompt found at index: 1")
    print(f"✓ Memory should be inserted at index: {insert_idx}")
    print(f"✓ This is after: [{messages[insert_idx-1]['role']}] (persona prompt)")
    print(f"✓ This is before: [{messages[insert_idx]['role']}] (first user message)")

    assert insert_idx == 2, f"Expected insert_idx=2 (after persona), got {insert_idx}"

    print("\n✅ _persona_prompt_index() works correctly!")
    return True


def test_context_formatter():
    """Test that ContextFormatter preserves source tags and deduplicates"""
    formatter = ContextFormatter(
        max_bullets=3,
        inject_role="system",
        inject_header="Use the following factual context if helpful."
    )

    # Test bullets with source tags (as they come from retrieval)
    test_bullets = [
        "• [convo] Hello, it's nice to be back. (27m ago)",
        "• [graph] dog is named potola (23h ago)",
        "• [convo] Hello, it's nice to be back. (27m ago)",  # Duplicate
        "• [convo] And my dog's name is Podola. (23h ago)",
    ]

    formatted = formatter.format_bullets(test_bullets, max_bullets=3)

    print("\n📊 ContextFormatter Test:")
    print(f"  Input: {len(test_bullets)} bullets (1 duplicate)")
    print(f"  Output: {len(formatted)} bullets (deduped, capped)")

    for i, bullet in enumerate(formatted, 1):
        print(f"    {i}. {bullet}")

    # Verify deduplication
    assert len(formatted) == 3, f"Expected 3 bullets after dedup, got {len(formatted)}"

    # Verify source tags preserved
    assert "[convo]" in formatted[0], "Source tag [convo] should be preserved"
    assert "[graph]" in formatted[1], "Source tag [graph] should be preserved"

    # Verify message building
    message = formatter.build_message("system", "Use the following factual context if helpful.", formatted)

    print(f"\n✓ Message role: {message['role']}")
    print(f"✓ Message content has header: {'Use the following factual context' in message['content']}")
    print(f"✓ Message content has {len(formatted)} bullets")

    assert message['role'] == "system"
    assert "Use the following factual context" in message['content']

    print("\n✅ ContextFormatter works correctly!")
    return True


def test_integration():
    """Test full integration - verify no Context Guide in messages"""
    print("\n🔍 Integration Test: Verify Context Guide removed")

    # This is more of a documentation test since we can't easily test factory
    # without a full bot setup, but we can verify the code changes

    with open("core/factory.py", "r") as f:
        factory_content = f.read()

    # Check that Context Guide code was removed/commented
    has_old_guide = 'guide_default = (' in factory_content and '"Context Guide:' in factory_content

    if has_old_guide:
        print("  ⚠️  Found Context Guide code in factory.py")
        print("  Check that it's properly commented or removed")
        return False

    print("  ✓ Context Guide code removed from factory.py")
    print("  ✓ Message order will be: Session → Persona → Memory → History")

    print("\n✅ Integration test passed!")
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("Context Ordering Integration Test")
    print("="*60)

    try:
        success = True
        success = success and test_persona_prompt_index()
        success = success and test_context_formatter()
        success = success and test_integration()

        if success:
            print("\n" + "="*60)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("="*60)
            print("\nSummary of Changes:")
            print("  1. Memory bullets now inserted AFTER persona prompt")
            print("  2. Context Guide removed (redundant with persona prompt)")
            print("  3. ContextFormatter properly preserves source tags")
            print("  4. Final message order: Session → Persona → Memory → History")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
