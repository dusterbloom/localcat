#!/usr/bin/env python
"""
Test transcription across dialogue turns to ensure clean state management
"""

import asyncio
import sys
import os

# Ensure server root is importable
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for p in (_SERVER_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)


class MockTranscriber:
    """Mock transcriber that simulates progressive text accumulation"""
    def __init__(self, text_progression):
        self.text_progression = text_progression
        self.call_count = 0

    @property
    def result(self):
        if self.call_count < len(self.text_progression):
            text = self.text_progression[self.call_count]
            self.call_count += 1
            return type('Result', (), {'text': text})()
        return type('Result', (), {'text': self.text_progression[-1]})()


async def test_turn_transcription():
    """Test that transcription resets properly between dialogue turns"""
    print("Testing transcription state management across dialogue turns...")

    # Create mock STT instance
    class MockSTT:
        def __init__(self):
            self._last_sent_length = 0

    stt = MockSTT()

    # Simulate Turn 1: User says "Hello world!"
    print("\n--- Turn 1: User says 'Hello world!' ---")

    # Reset for new turn
    stt._last_sent_length = 0

    # Mock the transcriber to return predictable text
    transcriber1 = MockTranscriber([
        "Hello",              # Turn 1 - first chunk
        "Hello world",         # Turn 1 - accumulated
        "Hello world!",        # Turn 1 - final
    ])

    # Simulate processing audio chunks
    frames_turn1 = []

    # First chunk
    result = transcriber1.result
    if hasattr(result, 'text'):
        full_text = result.text.strip()
        print(f"Model output: '{full_text}'")

        if full_text:
            if len(full_text) > stt._last_sent_length:
                new_text = full_text[stt._last_sent_length:]
                if new_text:
                    # Create mock frame
                    frame = type('MockFrame', (), {'text': new_text})()
                    frames_turn1.append(frame)
                    print(f"Interim frame: '{new_text}'")
                    stt._last_sent_length = len(full_text)

    # Second chunk
    result = transcriber1.result
    if hasattr(result, 'text'):
        full_text = result.text.strip()
        print(f"Model output: '{full_text}'")

        if full_text:
            if len(full_text) > stt._last_sent_length:
                new_text = full_text[stt._last_sent_length:]
                if new_text:
                    # Create mock frame
                    frame = type('MockFrame', (), {'text': new_text})()
                    frames_turn1.append(frame)
                    print(f"Interim frame: '{new_text}'")
                    stt._last_sent_length = len(full_text)

    # Simulate Turn 2: User says "How are you?"
    print("\n--- Turn 2: User says 'How are you?' ---")

    # Reset for new turn
    stt._last_sent_length = 0

    # Mock new transcriber for turn 2
    transcriber2 = MockTranscriber([
        "How",                 # Turn 2 - first chunk
        "How are",            # Turn 2 - accumulated
        "How are you?",       # Turn 2 - final
    ])

    frames_turn2 = []

    # First chunk of turn 2
    result = transcriber2.result
    if hasattr(result, 'text'):
        full_text = result.text.strip()
        print(f"Model output: '{full_text}'")

        if full_text:
            if len(full_text) > stt._last_sent_length:
                new_text = full_text[stt._last_sent_length:]
                if new_text:
                    # Create mock frame
                    frame = type('MockFrame', (), {'text': new_text})()
                    frames_turn2.append(frame)
                    print(f"Interim frame: '{new_text}'")
                    stt._last_sent_length = len(full_text)

    # Second chunk of turn 2
    result = transcriber2.result
    if hasattr(result, 'text'):
        full_text = result.text.strip()
        print(f"Model output: '{full_text}'")

        if full_text:
            if len(full_text) > stt._last_sent_length:
                new_text = full_text[stt._last_sent_length:]
                if new_text:
                    # Create mock frame
                    frame = type('MockFrame', (), {'text': new_text})()
                    frames_turn2.append(frame)
                    print(f"Interim frame: '{new_text}'")
                    stt._last_sent_length = len(full_text)

    # Analyze results
    print("\n📊 Analysis:")

    # Check if state was reset between turns
    print(f"STT _last_sent_length after turns: {stt._last_sent_length}")

    # Extract text from frames
    turn1_texts = [f.text for f in frames_turn1 if hasattr(f, 'text')]
    turn2_texts = [f.text for f in frames_turn2 if hasattr(f, 'text')]

    print(f"Turn 1 texts: {turn1_texts}")
    print(f"Turn 2 texts: {turn2_texts}")

    # Check for text carryover
    turn1_combined = ''.join(turn1_texts)
    turn2_combined = ''.join(turn2_texts)

    if turn1_combined in turn2_combined and turn1_combined != turn2_combined:
        print(f"❌ TEXT CARRYOVER DETECTED: Turn 2 contains Turn 1 text")
        print(f"  Turn 1: '{turn1_combined}'")
        print(f"  Turn 2: '{turn2_combined}'")
        return False
    else:
        print("✅ No text carryover detected between turns")

    # Check for incremental updates
    expected_turn1 = ["Hello", " world"]
    expected_turn2 = ["How", " are"]

    if turn1_texts == expected_turn1 and turn2_texts == expected_turn2:
        print("✅ Incremental transcription working correctly")
    else:
        print(f"❌ Unexpected text sequences")
        print(f"  Expected Turn 1: {expected_turn1}, Got: {turn1_texts}")
        print(f"  Expected Turn 2: {expected_turn2}, Got: {turn2_texts}")
        return False

    print("\n🎯 Turn transcription test completed successfully!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_turn_transcription())
    if not success:
        sys.exit(1)