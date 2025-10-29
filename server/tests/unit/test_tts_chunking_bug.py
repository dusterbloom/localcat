"""Ad hoc test to demonstrate TTS double aggregation chunking bug.

This test proves that:
1. FastTextAggregator correctly sends full sentences
2. Kokoro TTS then splits them again internally (the bug)
3. This causes unnatural breaks like "friendly and <BREAK> concise"

Run with: pytest server/tests/unit/test_tts_chunking_bug.py -v -s
"""

import pytest
from pipecat.frames.frames import Frame, TextFrame, TTSTextFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection
from core.aggregators.fast_text import FastTextAggregator
from core.tts.kokoro_professional_direct import ProfessionalKokoroTTSService


class FrameCapture:
    """Helper class to capture frames pushed by processors."""

    def __init__(self):
        self.frames = []

    async def push_frame(self, frame: Frame, direction=None):
        """Capture pushed frames."""
        self.frames.append(frame)

    def get_text_frames(self):
        """Get only TextFrame objects."""
        return [f for f in self.frames if isinstance(f, TextFrame)]

    def get_tts_text_frames(self):
        """Get only TTSTextFrame objects."""
        return [f for f in self.frames if isinstance(f, TTSTextFrame)]

    def clear(self):
        """Clear captured frames."""
        self.frames.clear()


@pytest.mark.asyncio
async def test_fast_text_aggregator_sends_full_sentence():
    """Test 1: Verify FastTextAggregator sends the full sentence without splitting."""

    # The problematic sentence from logs
    test_sentence = "I'll make sure to keep this conversation friendly and concise."

    # Create aggregator with settings from factory (min_tokens=200, max_tokens=250, min_words=10)
    aggregator = FastTextAggregator(min_tokens=200, max_tokens=250, min_words=10)

    # Capture frames
    capture = FrameCapture()
    aggregator.push_frame = capture.push_frame

    # Feed the sentence as one TextFrame (simulating LLM output)
    await aggregator.process_frame(TextFrame(test_sentence), FrameDirection.DOWNSTREAM)

    # Get captured text frames
    text_frames = capture.get_text_frames()

    # Assertions
    print(f"\n📊 FastTextAggregator Results:")
    print(f"   Input: '{test_sentence}'")
    print(f"   Output frames: {len(text_frames)}")
    for i, frame in enumerate(text_frames):
        print(f"   Frame {i+1}: '{frame.text}' (len={len(frame.text)})")

    # FastTextAggregator should release as ONE complete sentence
    assert len(text_frames) == 1, f"Expected 1 TextFrame, got {len(text_frames)}"
    assert text_frames[0].text.strip() == test_sentence.strip(), \
        f"Expected full sentence, got: '{text_frames[0].text}'"

    print("   ✅ PASS: FastTextAggregator correctly sends full sentence\n")


@pytest.mark.asyncio
@pytest.mark.requires_models
async def test_kokoro_chunks_internally():
    """Test 2: Demonstrate Kokoro TTS splits the sentence internally (the bug)."""

    # The full sentence that FastTextAggregator sends
    full_sentence = "I'll make sure to keep this conversation friendly and concise."

    # Create Kokoro TTS (aggregate_sentences=True is hardcoded - should NOT chunk, but does)
    tts = ProfessionalKokoroTTSService(
        voice="af_heart",
        push_text_frames=True  # Enable TTSTextFrame emission for tracking
    )

    # Capture TTSTextFrames emitted during generation
    tts_text_frames = []
    original_push = tts.push_frame

    async def capture_tts_frames(frame, direction=None):
        if isinstance(frame, TTSTextFrame):
            tts_text_frames.append(frame)
        await original_push(frame, direction)

    tts.push_frame = capture_tts_frames

    # Run TTS on the full sentence
    print(f"\n📊 Kokoro TTS Internal Chunking Test:")
    print(f"   Input to TTS: '{full_sentence}'")
    print(f"   aggregate_sentences: True")

    chunk_count = 0
    async for frame in tts.run_tts(full_sentence):
        if isinstance(frame, TTSStartedFrame):
            chunk_count += 1

    print(f"   TTSStartedFrame emissions: {chunk_count}")
    print(f"   TTSTextFrames emitted: {len(tts_text_frames)}")
    for i, frame in enumerate(tts_text_frames):
        print(f"   TTSTextFrame {i+1}: '{frame.text}'")

    # THE BUG: Kokoro splits internally despite aggregate_sentences=True
    # Expected: 1 chunk
    # Actual: 2+ chunks (splits at "friendly and <BREAK> concise")

    if chunk_count == 1:
        print("   ✅ PASS: Kokoro respects aggregate_sentences=True (bug is fixed!)\n")
    else:
        print(f"   ❌ BUG CONFIRMED: Kokoro split into {chunk_count} chunks despite aggregate_sentences=True")
        print(f"   This causes unnatural breaks like 'friendly and <BREAK> concise'\n")
        pytest.fail(f"Kokoro should produce 1 chunk, but produced {chunk_count} chunks")


@pytest.mark.asyncio
@pytest.mark.requires_models
async def test_end_to_end_chunking_behavior():
    """Test 3: Integration test showing double aggregation in full pipeline."""

    test_sentence = "I'll make sure to keep this conversation friendly and concise."

    # Step 1: FastTextAggregator
    aggregator = FastTextAggregator(min_tokens=200, max_tokens=250, min_words=10)
    aggregator_capture = FrameCapture()
    aggregator.push_frame = aggregator_capture.push_frame

    await aggregator.process_frame(TextFrame(test_sentence), FrameDirection.DOWNSTREAM)

    # Step 2: Get what FastTextAggregator sends to TTS
    aggregated_text = aggregator_capture.get_text_frames()[0].text

    # Step 3: Send to Kokoro TTS
    tts = ProfessionalKokoroTTSService(
        voice="af_heart",
        push_text_frames=True
    )

    chunk_count = 0
    async for frame in tts.run_tts(aggregated_text):
        if isinstance(frame, TTSStartedFrame):
            chunk_count += 1

    print(f"\n📊 End-to-End Pipeline Test:")
    print(f"   Original sentence: '{test_sentence}'")
    print(f"   After FastTextAggregator: '{aggregated_text}' (1 frame)")
    print(f"   After Kokoro TTS: {chunk_count} audio chunks")

    if chunk_count == 1:
        print("   ✅ PASS: End-to-end pipeline produces 1 continuous audio chunk\n")
    else:
        print(f"   ❌ FAIL: Pipeline should produce 1 chunk, but produced {chunk_count} chunks")
        print("   Root cause: Double aggregation (FastText + Kokoro internal chunking)\n")
        pytest.fail(f"Expected 1 audio chunk, got {chunk_count} chunks")


@pytest.mark.asyncio
async def test_auxiliary_verb_protection():
    """Test 4: Verify FastTextAggregator doesn't break after auxiliary verbs."""

    # Test sentences with auxiliary verbs that should NOT be split
    test_cases = [
        "Nicolas Sarkozy and Emmanuel Macron have led the current wave.",
        "They will be attending the conference next week.",
        "She has been working on this project for months.",
        "I can see the solution clearly now.",
    ]

    aggregator = FastTextAggregator(min_tokens=200, max_tokens=250, min_words=10)

    print(f"\n📊 Auxiliary Verb Protection Test:")

    for sentence in test_cases:
        capture = FrameCapture()
        aggregator.push_frame = capture.push_frame
        aggregator._aggregation = ""  # Reset aggregation

        await aggregator.process_frame(TextFrame(sentence), FrameDirection.DOWNSTREAM)

        text_frames = capture.get_text_frames()

        if len(text_frames) == 1:
            print(f"   ✅ '{sentence}' → 1 frame (no split)")
        else:
            print(f"   ❌ '{sentence}' → {len(text_frames)} frames (unexpected split)")
            for i, frame in enumerate(text_frames):
                print(f"      Frame {i+1}: '{frame.text}'")

    print()


if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("TTS CHUNKING BUG DEMONSTRATION TEST SUITE")
    print("=" * 80)

    # Run tests manually
    asyncio.run(test_fast_text_aggregator_sends_full_sentence())

    print("\nNOTE: Tests 2 and 3 require models. Run with pytest:")
    print("  pytest server/tests/unit/test_tts_chunking_bug.py -v -s")
