"""Test for LLM TextFrame duplication bug.

This test verifies that DirectMLXLLMServiceWithTools doesn't generate duplicate
LLMTextFrames during streaming generation.

Run with: pytest server/tests/unit/test_llm_duplication_bug.py -v -s
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pipecat.frames.frames import Frame, LLMTextFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from core.llm.direct_mlx_llm_with_tools import DirectMLXLLMServiceWithTools


@pytest.mark.asyncio
async def test_llm_no_duplicate_textframes():
    """Test that DirectMLXLLMServiceWithTools doesn't produce duplicate TextFrames."""

    # Mock the model loading completely to avoid downloading from HuggingFace
    with patch('core.llm.direct_mlx_llm.mlx_lm.load') as mock_load:
        with patch('core.llm.direct_mlx_llm_with_tools.mlx_lm') as mock_mlx_lm_tools:
            # Mock model and tokenizer objects
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.chat_template = "test template"
            mock_load.return_value = (mock_model, mock_tokenizer)

            # Create LLM service instance
            llm = DirectMLXLLMServiceWithTools(
                model="test-model",
                model_path="test-path"
            )

    # Mock the mlx_lm.stream_generate to simulate token generation
    mock_tokens = [
        Mock(text="Hello"),
        Mock(text=" there"),
        Mock(text="!"),
        Mock(text=""),
    ]

    # Create a mock context
    context = OpenAILLMContext(
        messages=[
            {"role": "user", "content": "Say hello"}
        ]
    )

    collected_frames = []

    # Mock mlx_lm and related dependencies
    with patch('core.llm.direct_mlx_llm_with_tools.mlx_lm') as mock_mlx_lm:
        with patch('core.llm.direct_mlx_llm_with_tools.MLX_GLOBAL_LOCK'):
            # Mock the stream_generate function
            mock_mlx_lm.stream_generate.return_value = mock_tokens

            # Mock make_sampler
            mock_mlx_lm.make_sampler.return_value = Mock()

            # Collect frames from the LLM
            async for frame in llm._process_context(context):
                if isinstance(frame, LLMTextFrame):
                    collected_frames.append(frame)
                    print(f"📝 LLMTextFrame: '{frame.text}' (ID={id(frame)})")

    # Verify no duplicate text content
    texts = [frame.text for frame in collected_frames]
    print(f"📊 Generated texts: {texts}")

    # Check for duplicates
    seen_texts = set()
    duplicates_found = []
    for i, text in enumerate(texts):
        if text in seen_texts:
            duplicates_found.append((i, text))
        else:
            seen_texts.add(text)

    # Assertions
    assert len(collected_frames) == 3, f"Expected 3 TextFrames, got {len(collected_frames)}"
    assert len(duplicates_found) == 0, f"Found duplicate texts: {duplicates_found}"

    print("✅ PASS: No duplicate LLMTextFrames generated")


@pytest.mark.asyncio
async def test_llm_handles_repeated_tokens():
    """Test that LLM handles legitimate repeated tokens correctly."""

    # Create LLM service instance
    llm = DirectMLXLLMServiceWithTools(
        model="test-model",
        model_path="test-path"
    )

    # Mock tokens that include legitimate repetitions (like "and and")
    mock_tokens = [
        Mock(text="I like"),
        Mock(text=" and"),
        Mock(text=" and"),
        Mock(text=" coffee"),
        Mock(text="."),
    ]

    context = OpenAILLMContext(
        messages=[
            {"role": "user", "content": "Say something with repeated words"}
        ]
    )

    collected_frames = []

    with patch('core.llm.direct_mlx_llm_with_tools.mlx_lm') as mock_mlx_lm:
        with patch('core.llm.direct_mlx_llm_with_tools.MLX_GLOBAL_LOCK'):
            mock_mlx_lm.stream_generate.return_value = mock_tokens
            mock_mlx_lm.make_sampler.return_value = Mock()

            async for frame in llm._process_context(context):
                if isinstance(frame, LLMTextFrame):
                    collected_frames.append(frame)

    texts = [frame.text for frame in collected_frames]
    print(f"📊 Generated texts with legitimate repetitions: {texts}")

    # Should allow legitimate repeated tokens but prevent duplicate frames
    assert len(collected_frames) == 5, f"Expected 5 TextFrames, got {len(collected_frames)}"

    # Check that "and" appears twice (legitimate repetition) but frames are different
    and_frames = [frame for frame in collected_frames if frame.text == "and"]
    assert len(and_frames) == 2, f"Expected 2 'and' frames, got {len(and_frames)}"

    # Verify the frames have different IDs (they're different frame objects)
    assert id(and_frames[0]) != id(and_frames[1]), "Frames should be different objects"

    print("✅ PASS: Legitimate repeated tokens handled correctly")


@pytest.mark.asyncio
async def test_llm_generation_id_tracking():
    """Test that generation IDs are properly tracked across multiple generations."""

    llm = DirectMLXLLMServiceWithTools(
        model="test-model",
        model_path="test-path"
    )

    # First generation
    with patch('core.llm.direct_mlx_llm_with_tools.mlx_lm') as mock_mlx_lm:
        with patch('core.llm.direct_mlx_llm_with_tools.MLX_GLOBAL_LOCK'):
            mock_mlx_lm.stream_generate.return_value = [Mock(text="First"), Mock(text=" gen")]
            mock_mlx_lm.make_sampler.return_value = Mock()

            context1 = OpenAILLMContext(messages=[{"role": "user", "content": "First"}])
            gen1_frames = []

            async for frame in llm._process_context(context1):
                if isinstance(frame, LLMTextFrame):
                    gen1_frames.append(frame)

    gen1_id = llm._generation_id
    print(f"📊 Generation 1 ID: {gen1_id}")

    # Second generation
    with patch('core.llm.direct_mlx_llm_with_tools.mlx_lm') as mock_mlx_lm:
        with patch('core.llm.direct_mlx_llm_with_tools.MLX_GLOBAL_LOCK'):
            mock_mlx_lm.stream_generate.return_value = [Mock(text="Second"), Mock(text=" gen")]
            mock_mlx_lm.make_sampler.return_value = Mock()

            context2 = OpenAILLMContext(messages=[{"role": "user", "content": "Second"}])
            gen2_frames = []

            async for frame in llm._process_context(context2):
                if isinstance(frame, LLMTextFrame):
                    gen2_frames.append(frame)

    gen2_id = llm._generation_id
    print(f"📊 Generation 2 ID: {gen2_id}")

    # Verify generation IDs are different
    assert gen2_id > gen1_id, f"Generation ID should increment. Got gen1={gen1_id}, gen2={gen2_id}"

    # Verify deduplication tracking is reset between generations
    assert llm._last_generated_text == "", "Last generated text should be reset after generation"

    print("✅ PASS: Generation ID tracking works correctly")


if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("LLM DUPLICATION BUG TEST SUITE")
    print("=" * 80)

    asyncio.run(test_llm_no_duplicate_textframes())
    asyncio.run(test_llm_handles_repeated_tokens())
    asyncio.run(test_llm_generation_id_tracking())

    print("\nAll tests passed! 🎉")