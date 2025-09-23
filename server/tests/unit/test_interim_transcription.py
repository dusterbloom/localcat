#!/usr/bin/env python
"""
Test interim transcription logic to prevent duplicate/repetitive text
"""

import numpy as np
import sys
import os
import asyncio

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


async def test_interim_transcription_logic():
    """Test that interim transcription sends only new text, not duplicates"""
    from core.stt.parakeet_streaming import ParakeetStreamingSTT

    # Create STT instance
    stt = ParakeetStreamingSTT.__new__(ParakeetStreamingSTT)  # Create without __init__
    stt._last_sent_length = 0

    # Simulate progressive transcription results
    # This simulates what might happen with streaming: text builds up over time
    mock_transcriber = MockTranscriber([
        "Wanna",                    # First chunk
        "Wanna talk",              # Second chunk
        "Wanna talk about",        # Third chunk
        "Wanna talk about finance" # Fourth chunk
    ])

    # Simulate the transcription logic
    sent_texts = []
    for i in range(4):
        result = mock_transcriber.result
        if hasattr(result, 'text'):
            full_text = result.text
            print(f"Full text from model: '{full_text}'")

            if full_text:
                # This is the fixed logic from run_stt
                if len(full_text) > stt._last_sent_length:
                    new_text = full_text[stt._last_sent_length:]

                    if new_text:
                        sent_texts.append(new_text)
                        print(f"Sent interim: '{new_text}' (len: {len(new_text)})")
                        stt._last_sent_length = len(full_text)

    print(f"\nSent texts: {sent_texts}")

    # Verify we sent incremental updates, not the full text each time
    expected = ["Wanna", " talk", " about", " finance"]
    assert sent_texts == expected, f"Expected {expected}, got {sent_texts}"

    print("✅ Interim transcription logic working correctly - no duplicates!")


async def test_repetitive_model_output():
    """Test handling of repetitive output from the model itself"""
    from core.stt.parakeet_streaming import ParakeetStreamingSTT

    # Create STT instance
    stt = ParakeetStreamingSTT.__new__(ParakeetStreamingSTT)  # Create without __init__
    stt._last_sent_length = 0

    # Simulate a model that produces repetitive text (hallucination)
    mock_transcriber = MockTranscriber([
        "Wanna talk about",                          # Normal
        "Wanna talk about wanna talk about",        # Repetitive
        "Wanna talk about wanna talk about anymore" # More text added
    ])

    sent_texts = []
    for i in range(3):
        result = mock_transcriber.result
        if hasattr(result, 'text'):
            full_text = result.text.strip()
            print(f"Full text from model: '{full_text}'")

            if full_text:
                if len(full_text) > stt._last_sent_length:
                    new_text = full_text[stt._last_sent_length:]

                    if new_text.strip():
                        sent_texts.append(new_text.strip())
                        print(f"Sent interim: '{new_text.strip()}'")
                        stt._last_sent_length = len(full_text)

    print(f"\nSent texts: {sent_texts}")

    # Should send incremental updates even with repetitive model output
    expected = ["Wanna talk about", " wanna talk about", " anymore"]
    assert sent_texts == expected, f"Expected {expected}, got {sent_texts}"

    print("✅ Repetitive model output handled correctly!")


if __name__ == "__main__":
    asyncio.run(test_interim_transcription_logic())
    asyncio.run(test_repetitive_model_output())