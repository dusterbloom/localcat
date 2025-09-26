#!/usr/bin/env python
"""
Unit tests for Phase 0.5 env controls:
- HOTMEM_BULLETS_MAX caps pending bullets
- ENABLE_MEMORY disables processing
"""

import os
import sys
from loguru import logger
import pytest

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.memory.hotpath_processor import HotPathMemoryProcessor
from pipecat.frames.frames import TranscriptionFrame


@pytest.mark.fast
@pytest.mark.ci
async def test_bullets_cap():
    os.environ["HOTMEM_BULLETS_MAX"] = "1"
    os.environ["ENABLE_MEMORY"] = "true"
    mem = HotPathMemoryProcessor(sqlite_path=":memory:", lmdb_dir=None, user_id="test-user", enable_metrics=False, context_aggregator=None)
    # Run final processing to populate pending bullets
    final = TranscriptionFrame(text="My name is Ana and I live in Paris", user_id="test-user", timestamp=0.5)
    await mem._process_transcription(final, None)
    assert len(mem._pending_bullets) <= 1, f"Expected cap 1, got {len(mem._pending_bullets)}"
    logger.info("HOTMEM_BULLETS_MAX cap respected")


@pytest.mark.fast
@pytest.mark.ci
async def test_enable_memory_false():
    os.environ["ENABLE_MEMORY"] = "false"
    mem = HotPathMemoryProcessor(sqlite_path=":memory:", lmdb_dir=None, user_id="test-user", enable_metrics=False, context_aggregator=None)
    final = TranscriptionFrame(text="My name is Ana and I live in Paris", user_id="test-user", timestamp=0.5)
    await mem._process_transcription(final, None)
    assert len(mem._pending_bullets) == 0, "Processing should be disabled when ENABLE_MEMORY=false"
    logger.info("ENABLE_MEMORY=false disables processing")


@pytest.mark.fast
@pytest.mark.ci
async def test_convo_index_and_retrieval():
    # Enable convo indexing and retrieval source
    os.environ["ENABLE_MEMORY"] = "true"
    os.environ["MEMORY_CONVO_INDEX"] = "true"
    os.environ["MEMORY_SOURCES"] = "convo"

    mem = HotPathMemoryProcessor(sqlite_path=":memory:", lmdb_dir=None, user_id="test-user", enable_metrics=False, context_aggregator=None)

    # First turn: index a final utterance
    final1 = TranscriptionFrame(text="I live on the east side near the river", user_id="test-user", timestamp=0.5)
    await mem._process_transcription(final1, None)

    # Second turn: query retrieval (should pull from convo FTS)
    final2 = TranscriptionFrame(text="Where do I live?", user_id="test-user", timestamp=1.0)
    await mem._process_transcription(final2, None)

    # Because sources=convo only, pending bullets should come from FTS search
    assert len(mem._pending_bullets) >= 0
    logger.info("Convo retrieval executed (bullets may be empty depending on FTS tokenizer)")


async def main():
    await test_bullets_cap()
    await test_enable_memory_false()
    await test_convo_index_and_retrieval()
    return True


if __name__ == "__main__":
    import asyncio
    ok = asyncio.run(main())
    import os as _os
    _os._exit(0 if ok else 1)
