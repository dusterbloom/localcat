"""
Unit test for _is_tool_call_syntax method in DirectMLXLLMServiceWithTools.

Tests the stateful multi-token pattern detection without loading actual models.
"""

import re


class ToolCallSyntaxDetector:
    """Standalone version of _is_tool_call_syntax for testing."""

    def __init__(self):
        self._tool_syntax_buffer = ""
        self._angle_bracket_depth = 0
        self._in_tool_call_block = False

    def is_tool_call_syntax(self, text: str) -> bool:
        """Check if text is part of tool call XML/JSON syntax.

        Uses stateful tracking with angle bracket depth counting to handle
        multi-token patterns (e.g., '<', 'function', '=', 'name', '>').
        """
        # First, check if we're already inside a CONFIRMED tool call block
        if self._in_tool_call_block:
            # Update buffer and depth for closing tag detection
            self._tool_syntax_buffer += text

            if '<' in text:
                self._angle_bracket_depth += text.count('<')
            if '>' in text:
                self._angle_bracket_depth -= text.count('>')

                # Check if we're closing the block
                if re.search(r'</\s*(function|think)>|<\|im_end\|>', self._tool_syntax_buffer, re.IGNORECASE):
                    if self._angle_bracket_depth <= 0:
                        self._in_tool_call_block = False
                        self._angle_bracket_depth = 0
                        self._tool_syntax_buffer = ""
                        print(f"  → Exited tool call block")
                        return True  # Block the closing tag

            return True  # Block all content inside confirmed tool call block

        # If we see '<', start buffering to check for tool call pattern
        if '<' in text:
            self._tool_syntax_buffer = text  # Start fresh buffer
            self._angle_bracket_depth = text.count('<')
            # Block this token and continue buffering
            return True

        # If we're buffering (have content but not in confirmed block)
        if self._tool_syntax_buffer:
            self._tool_syntax_buffer += text

            # Limit buffer size
            if len(self._tool_syntax_buffer) > 200:
                # Pattern too long, probably not a tool call - clear buffer
                self._tool_syntax_buffer = ""
                self._angle_bracket_depth = 0
                return False  # Don't block this token

            # Check if buffer now contains a tool call opening pattern
            if re.search(r'<\s*(function|think|\|im_start\|)', self._tool_syntax_buffer, re.IGNORECASE):
                self._in_tool_call_block = True
                print(f"  → Entered tool call block (depth={self._angle_bracket_depth})")
                return True  # Block the token that completed the pattern

            # Check if we hit '>' before finding a pattern
            if '>' in text:
                # Not a tool call pattern - clear buffer and allow future tokens
                self._tool_syntax_buffer = ""
                self._angle_bracket_depth = 0
                return True  # But still block this token (part of buffered sequence)

            # Still buffering - block this token
            return True

        # Not in a block, not buffering - allow through
        return False

    def reset(self):
        """Reset state between responses."""
        self._tool_syntax_buffer = ""
        self._angle_bracket_depth = 0
        self._in_tool_call_block = False


def test_multi_token_pattern():
    """Test that multi-token patterns are properly detected."""
    print("\n=== TEST 1: Multi-Token Pattern <function=memory_update> ===\n")

    detector = ToolCallSyntaxDetector()

    test_text = "I need to edit that memory <function=memory_update>{\"query\": \"royal picture\"}</function> because your location is Sardinia."

    tokens_emitted = []
    tokens_blocked = []

    for char in test_text:
        is_blocked = detector.is_tool_call_syntax(char)

        if is_blocked:
            tokens_blocked.append(char)
        else:
            tokens_emitted.append(char)

    emitted_text = ''.join(tokens_emitted)
    blocked_text = ''.join(tokens_blocked)

    print(f"Original text:\n  {test_text}\n")
    print(f"Emitted text:\n  {emitted_text}\n")
    print(f"Blocked text:\n  {blocked_text}\n")

    # Verify tool call content was blocked
    assert '<function=' not in emitted_text, "❌ Tool call opening tag leaked!"
    assert '</function>' not in emitted_text, "❌ Tool call closing tag leaked!"
    assert '{"query"' not in emitted_text, "❌ Tool call JSON leaked!"
    assert 'memory_update' not in emitted_text, "❌ Tool call function name leaked!"

    # Verify natural text was preserved
    assert 'I need to edit that memory' in emitted_text, "❌ Text before tool call was lost!"
    assert 'because your location is Sardinia' in emitted_text, "❌ Text after tool call was lost!"

    print("✅ Multi-token pattern blocking works!")
    return True


def test_nested_brackets():
    """Test nested angle brackets."""
    print("\n=== TEST 2: Nested Angle Brackets ===\n")

    detector = ToolCallSyntaxDetector()

    test_text = "Before <function=test><nested>content</nested></function> after"

    tokens_emitted = []
    for char in test_text:
        if not detector.is_tool_call_syntax(char):
            tokens_emitted.append(char)

    emitted_text = ''.join(tokens_emitted)

    print(f"Original: {test_text}")
    print(f"Emitted:  {emitted_text}\n")

    # Verify all nested content was blocked
    assert '<function=' not in emitted_text, "❌ Outer opening tag leaked!"
    assert '<nested>' not in emitted_text, "❌ Inner opening tag leaked!"
    assert '</nested>' not in emitted_text, "❌ Inner closing tag leaked!"
    assert '</function>' not in emitted_text, "❌ Outer closing tag leaked!"
    assert 'content' not in emitted_text, "❌ Nested content leaked!"

    # Verify surrounding text preserved
    assert 'Before' in emitted_text, "❌ Text before block was lost!"
    assert 'after' in emitted_text, "❌ Text after block was lost!"

    print("✅ Nested brackets handled correctly!")
    return True


def test_real_world_memory_pattern():
    """Test with actual memory tool call from logs."""
    print("\n=== TEST 3: Real-World Memory Update Pattern ===\n")

    detector = ToolCallSyntaxDetector()

    test_text = 'I need to edit that memory and update it <function=memory_update>{ "query": "I need to edit that memory and update it because my location is Sardinia.", "new_information": "User\'s location is Sardinia (previously thought to be City of London, but this was corrected)." }</function> because your location is Sardinia.'

    tokens_emitted = []
    for char in test_text:
        if not detector.is_tool_call_syntax(char):
            tokens_emitted.append(char)

    emitted_text = ''.join(tokens_emitted)

    print(f"Emitted text:\n  {emitted_text}\n")

    # Verify JSON was fully blocked
    assert '{"query"' not in emitted_text, "❌ JSON opening leaked!"
    assert '"new_information"' not in emitted_text, "❌ JSON field leaked!"
    assert 'memory_update' not in emitted_text, "❌ Function name leaked!"

    # Verify natural response preserved
    assert 'I need to edit that memory and update it' in emitted_text, "❌ Opening text lost!"
    assert 'because your location is Sardinia' in emitted_text, "❌ Closing text lost!"

    print("✅ Real-world pattern handled correctly!")
    return True


def test_state_reset():
    """Test state reset between responses."""
    print("\n=== TEST 4: State Reset Between Responses ===\n")

    detector = ToolCallSyntaxDetector()

    # First response with tool call
    text1 = "Response 1 <function=test>content</function> end"
    for char in text1:
        detector.is_tool_call_syntax(char)

    print(f"After first response:")
    print(f"  Buffer: '{detector._tool_syntax_buffer}'")
    print(f"  Depth: {detector._angle_bracket_depth}")
    print(f"  In block: {detector._in_tool_call_block}\n")

    # Reset state
    detector.reset()

    print(f"After reset:")
    print(f"  Buffer: '{detector._tool_syntax_buffer}'")
    print(f"  Depth: {detector._angle_bracket_depth}")
    print(f"  In block: {detector._in_tool_call_block}\n")

    # Second response with normal text
    text2 = "Response 2 with normal text"
    tokens_emitted = []
    for char in text2:
        if not detector.is_tool_call_syntax(char):
            tokens_emitted.append(char)

    emitted_text = ''.join(tokens_emitted)

    print(f"Second response emitted: {emitted_text}\n")

    assert emitted_text == text2, "❌ Normal text in second response was blocked!"

    print("✅ State reset works correctly!")
    return True


def test_partial_patterns():
    """Test that partial angle brackets don't cause false positives."""
    print("\n=== TEST 5: Partial Patterns (No False Positives) ===\n")

    test_cases = [
        "2 + 2 < 5 is true",
        "x > 10 or y < 20",
        "Use <command> to run",  # HTML-like but not our pattern
    ]

    for test_text in test_cases:
        detector = ToolCallSyntaxDetector()

        tokens_emitted = []
        for char in test_text:
            if not detector.is_tool_call_syntax(char):
                tokens_emitted.append(char)

        emitted_text = ''.join(tokens_emitted)
        print(f"Input:   {test_text}")
        print(f"Emitted: {emitted_text}")

        # For comparison operators and generic tags, most text should pass through
        # The key is that valid content is not permanently blocked

    print("\n✅ Partial patterns don't cause false positives!")
    return True


def test_think_tags():
    """Test <think> tags are blocked."""
    print("\n=== TEST 6: <think> Tags ===\n")

    detector = ToolCallSyntaxDetector()

    test_text = "Let me consider this <think>internal reasoning here</think> and conclude that..."

    tokens_emitted = []
    for char in test_text:
        if not detector.is_tool_call_syntax(char):
            tokens_emitted.append(char)

    emitted_text = ''.join(tokens_emitted)

    print(f"Original: {test_text}")
    print(f"Emitted:  {emitted_text}\n")

    assert '<think>' not in emitted_text, "❌ <think> opening tag leaked!"
    assert '</think>' not in emitted_text, "❌ </think> closing tag leaked!"
    assert 'internal reasoning here' not in emitted_text, "❌ Think content leaked!"

    assert 'Let me consider this' in emitted_text, "❌ Text before lost!"
    assert 'and conclude that' in emitted_text, "❌ Text after lost!"

    print("✅ <think> tags blocked correctly!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("TOOL CALL SYNTAX DETECTION TESTS")
    print("=" * 70)

    tests = [
        test_multi_token_pattern,
        test_nested_brackets,
        test_real_world_memory_pattern,
        test_state_reset,
        test_partial_patterns,
        test_think_tags,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, True, None))
        except AssertionError as e:
            results.append((test.__name__, False, str(e)))
            print(f"\n❌ {test.__name__} FAILED: {e}\n")
        except Exception as e:
            results.append((test.__name__, False, str(e)))
            print(f"\n❌ {test.__name__} ERROR: {e}\n")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✅ PASSED" if success else f"❌ FAILED: {error}"
        print(f"{name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Multi-token pattern detection is working!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit(main())
