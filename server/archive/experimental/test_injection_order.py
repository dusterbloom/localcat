#!/usr/bin/env python3
"""
Verify HotMem injection order and role (mem0-style), plus storage/retrieval sanity.

This test runs without a full Pipeline by subclassing the processor to capture
outgoing frames and driving `process_frame` directly with Start → Transcription → User messages.
"""

import asyncio
import os
from typing import List, Tuple

from loguru import logger

# Prefer mem0-like system injection for this test
os.environ["HOTMEM_INJECT_ROLE"] = "system"
os.environ["HOTMEM_INJECT_HEADER"] = "[Memory]"
os.environ["HOTMEM_SQLITE"] = "test_inject.db"
os.environ["HOTMEM_LMDB_DIR"] = "test_inject.lmdb"

from hotpath_processor import HotPathMemoryProcessor
from pipecat.frames.frames import StartFrame, TranscriptionFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection


class CapturingProcessor(HotPathMemoryProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out: List[Tuple[str, object]] = []

    async def push_frame(self, frame, direction):
        # Capture without forwarding anywhere else
        self.out.append((type(frame).__name__, frame))

    # Bypass TaskManager for unit testing
    def create_task(self, coroutine, name: str | None = None):
        import asyncio
        return asyncio.create_task(coroutine)

    def cancel_task(self, task):
        try:
            task.cancel()
        except Exception:
            pass


async def run():
    # Create processor
    proc = CapturingProcessor(
        sqlite_path=os.getenv("HOTMEM_SQLITE"),
        lmdb_dir=os.getenv("HOTMEM_LMDB_DIR"),
        user_id="test-user",
    )

    # 1) StartFrame
    await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    # 2) User transcription (final)
    t = TranscriptionFrame("My dog's name is Potola", user_id="u", timestamp="t")
    # Mark as final so HotMem extracts and prepares bullets
    setattr(t, "is_final", True)
    await proc.process_frame(t, FrameDirection.DOWNSTREAM)

    # 3) Aggregator sends user LLMMessagesFrame → should inject memory frame BEFORE this
    user_batch = LLMMessagesFrame([{"role": "user", "content": "Hi there"}])
    await proc.process_frame(user_batch, FrameDirection.DOWNSTREAM)

    # Assertions (simple prints for now)
    names = [n for (n, _) in proc.out]
    # Expect order: StartFrame ... Injected LLMMessagesFrame ... Original LLMMessagesFrame
    injected_idx = None
    original_idx = None
    for i, (n, f) in enumerate(proc.out):
        if n == "LLMMessagesFrame":
            # The first LLMMessagesFrame after transcription should be the injected memory
            if injected_idx is None:
                injected_idx = i
                # Validate role and content
                msgs = getattr(f, "messages", [])
                assert msgs and msgs[0].get("role") == os.getenv("HOTMEM_INJECT_ROLE", "user"), "Injected role mismatch"
                assert "potola" in (msgs[0].get("content", "").lower()), "Injected content should include Potola"
            else:
                original_idx = i
                break

    assert injected_idx is not None and original_idx is not None, "Did not observe injected + original LLMMessagesFrames"
    assert injected_idx < original_idx, "Injected memory must precede original user batch"

    # Storage smoke-check
    # Ensure pending writes are persisted before reading
    proc.store.flush()
    edges = proc.store.get_all_edges()
    assert any("potola" in d for (_, _, d, _) in edges), "Storage missing Potola edge"

    # Retrieval (non-mutating) check
    preview = proc.hot.preview_bullets("What is my dog's name?")
    bullets = " ".join(preview["bullets"]).lower()
    assert "potola" in bullets, "Preview bullets should mention Potola"

    print("\n✅ Injection order, role, storage, and retrieval verified.")

    # Cleanup
    import shutil
    try:
        if os.path.exists("test_inject.db"):
            os.remove("test_inject.db")
        if os.path.exists("test_inject.lmdb"):
            shutil.rmtree("test_inject.lmdb")
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}")


if __name__ == "__main__":
    asyncio.run(run())
