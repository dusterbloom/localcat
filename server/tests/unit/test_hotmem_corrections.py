#!/usr/bin/env python
"""Regression tests for HotMem fact correction and retrieval labeling."""

import os
import sys

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")

for path in (_SERVER_ROOT, _PIPECAT_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

from pipecat.frames.frames import TranscriptionFrame

from hotpath_processor import HotPathMemoryProcessor


async def run():
    mem = HotPathMemoryProcessor(
        sqlite_path=":memory:",
        lmdb_dir=None,
        user_id="test-user",
        enable_metrics=False,
        context_aggregator=None,
    )

    first = TranscriptionFrame(text="My dog's name is Gus.", user_id="test-user", timestamp=0.0)
    await mem._process_transcription(first, None)

    correction = TranscriptionFrame(text="No, my dog's name is Po.", user_id="test-user", timestamp=1.0)
    await mem._process_transcription(correction, None)

    bullets = mem.hot.retrieve_bullets("Tell me about my dog", read_only=True)

    assert bullets, "Expected retrieval to return at least one bullet"
    joined = " ".join(b.lower() for b in bullets)
    assert "po" in joined, "Updated fact should be surfaced"
    assert "gus" not in joined, "Superseded fact should be demoted"

    # Graph-derived bullets must carry the explicit label for clarity
    assert all(b.startswith("• [graph]") for b in bullets), f"Unexpected bullet labels: {bullets}"

    # Entity index should no longer contain the stale destination
    assert ("dog", "name", "gus") not in mem.hot.entity_index["dog"], "Old fact still present in entity index"

    return True


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(run())
    sys.exit(0 if success else 1)
