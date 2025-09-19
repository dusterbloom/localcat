#!/usr/bin/env python3
"""
Debug script to test TranscriptionFrame flow through the pipeline
"""

import asyncio
import os
from dotenv import load_dotenv

from pipecat.frames.frames import TranscriptionFrame, StartFrame, EndFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

from hotpath_processor import HotPathMemoryProcessor

load_dotenv(override=True)

async def test_transcription_flow():
    """Test if TranscriptionFrames flow from HotMem to LLM"""

    print("🔍 Testing TranscriptionFrame pipeline flow...")

    # Create LLM service
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        max_tokens=100,
        stream=False
    )

    # Create context
    context = OpenAILLMContext([
        {"role": "system", "content": "You are a helpful assistant. Respond with 'I heard you!' to any message."}
    ])

    context_aggregator = llm.create_context_aggregator(context)

    # Create HotMem processor
    memory = HotPathMemoryProcessor(
        sqlite_path=":memory:",
        lmdb_dir=None,
        user_id="test-user",
        enable_metrics=True,
        context_aggregator=context_aggregator
    )

    # Create simple pipeline: HotMem -> Context -> LLM
    pipeline = Pipeline([
        memory,
        context_aggregator.user(),
        llm,
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    async def send_test_frames():
        """Send test frames to the pipeline"""
        print("📤 Sending StartFrame...")
        await task.queue_frames([StartFrame()])

        await asyncio.sleep(0.1)

        print("📤 Sending TranscriptionFrame: 'Hello can you hear me?'")
        test_frame = TranscriptionFrame(
            text="Hello can you hear me?",
            user_id="test-user",
            timestamp="0.0"
        )
        await task.queue_frames([test_frame])

        await asyncio.sleep(2)  # Give time for LLM response

        print("📤 Sending EndFrame...")
        await task.queue_frames([EndFrame()])

    # Run the pipeline
    runner = PipelineRunner()

    # Start pipeline and send frames
    await asyncio.gather(
        runner.run(task),
        send_test_frames()
    )

if __name__ == "__main__":
    asyncio.run(test_transcription_flow())