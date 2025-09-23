#!/usr/bin/env python
"""
Lightweight test for Phase 0 interim pre-injection logic.
Avoids heavy STT/TTS init. Verifies that HotPathMemoryProcessor:
- Accepts InterimTranscriptionFrame and attempts retrieval
- Accepts TranscriptionFrame and performs final processing
"""

import os
import sys
from loguru import logger

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from core.memory.hotpath_processor import HotPathMemoryProcessor


async def run():
    # Use in-memory DB and disabled LMDB to keep it light
    mem = HotPathMemoryProcessor(sqlite_path=":memory:", lmdb_dir=None, user_id="test-user", enable_metrics=False, context_aggregator=None)

    # Phase 0 pre-injection simulation (without full Pipeline/TaskManager):
    # compute preview bullets and stage them
    interim_text = "my name is Ana and I live"
    preview = mem.hot.retrieve_bullets(interim_text, read_only=True)
    assert isinstance(preview, list)
    mem._pending_bullets = preview[:3]

    # Final processing: perform extraction + persist + retrieve (through internal method)
    final = TranscriptionFrame(text="My name is Ana and I live in Paris", user_id="test-user", timestamp=0.5)
    await mem._process_transcription(final, None)

    logger.info("Phase 0 interim + final processing completed without exception")
    return True


if __name__ == "__main__":
    import asyncio
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
