#!/usr/bin/env python3
"""
Quick verification test for LLM duplication fix.

Tests the DirectMLXLLMServiceWithTools fix using the real model from .env
to verify that duplicate TextFrames are no longer generated.

Run with: source .venv/bin/activate && python test_llm_fix_verification.py
"""

import asyncio
import os
from dotenv import load_dotenv
from pipecat.frames.frames import Frame, LLMTextFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from core.llm.direct_mlx_llm_with_tools import DirectMLXLLMServiceWithTools

# Load environment variables
load_dotenv()

async def test_llm_no_duplicate_frames():
    """Test that our fix prevents duplicate LLMTextFrames."""

    print("🔬 Testing LLM duplication fix with real model...")

    # Get model from environment
    model_name = os.getenv('LLM_MODEL', 'mlx-community/Qwen3-1.7B-8bit')
    print(f"📋 Using model: {model_name}")

    # Create LLM service with real model
    llm = DirectMLXLLMServiceWithTools(model=model_name)

    print("✅ LLM service created successfully")

    # Test context
    context = OpenAILLMContext(
        messages=[
            {"role": "user", "content": "Say hello briefly"}
        ]
    )

    print("🧪 Starting LLM generation...")

    collected_frames = []
    frame_ids = []

    try:
        # Collect frames from LLM
        async for frame in llm._process_context(context):
            if isinstance(frame, LLMTextFrame):
                frame_id = id(frame)
                frame_ids.append(frame_id)
                collected_frames.append(frame)

                # Check for duplicate content
                content = frame.text.strip()
                if content:  # Only check non-empty frames
                    print(f"📝 Frame {len(collected_frames)}: '{content}' (ID: {frame_id})")

                    # Check if this content was seen before
                    prev_contents = [f.text.strip() for f in collected_frames[:-1]]
                    if content in prev_contents:
                        print(f"🚨 DUPLICATE DETECTED: '{content}' was already generated!")
                        return False

        print(f"✅ Generated {len(collected_frames)} TextFrames")

        # Verify no empty frames or obvious duplicates
        non_empty_frames = [f for f in collected_frames if f.text.strip()]
        print(f"📊 Non-empty frames: {len(non_empty_frames)}")

        # Check for consecutive duplicate IDs (shouldn't happen with our fix)
        consecutive_duplicates = 0
        for i in range(1, len(frame_ids)):
            if frame_ids[i] == frame_ids[i-1]:
                consecutive_duplicates += 1
                print(f"🚨 CONSECUTIVE DUPLICATE FRAME ID: {frame_ids[i]}")

        if consecutive_duplicates == 0:
            print("✅ No consecutive duplicate frame IDs")
            return True
        else:
            print(f"❌ Found {consecutive_duplicates} consecutive duplicates")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

    finally:
        # Cleanup
        if hasattr(llm, '_model') and llm._model:
            del llm._model
        if hasattr(llm, '_tokenizer') and llm._tokenizer:
            del llm._tokenizer

async def main():
    """Main test function."""
    print("=" * 80)
    print("🚀 LLM DUPLICATION FIX VERIFICATION")
    print("=" * 80)

    success = await test_llm_no_duplicate_frames()

    print("\n" + "=" * 80)
    if success:
        print("🎉 SUCCESS: LLM duplication fix is working!")
        print("   - No duplicate TextFrames generated")
        print("   - Each frame has unique ID")
        print("   - Clean token streaming")
    else:
        print("❌ FAILURE: LLM duplication fix needs adjustment")
        print("   - Check if duplicate tokens are still being generated")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())