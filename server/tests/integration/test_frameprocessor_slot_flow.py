#!/usr/bin/env python
"""
Integration test: FrameProcessor slot-aware flow
Simulate the exact conversation order through FrameProcessor:
 - Seed a declarative color fact
 - Ask the color question
Assert that injected bullets are slot-aligned (color only).
"""

import os
import sys
import pytest

_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for p in (_SERVER_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from loguru import logger


@pytest.mark.fast
@pytest.mark.asyncio
async def test_frameprocessor_slot_flow_color_only():
    os.environ['MEMORY_SOURCES'] = 'convo,graph,summary'
    os.environ['MEMORY_CONVO_INDEX'] = 'true'
    os.environ['MEMORY_MAX_BULLETS'] = '2'
    os.environ['AUDIO_INTELLIGENCE_ENABLED'] = 'false'
    os.environ['ENABLE_MEMORY'] = 'true'

    from core.memory.hotpath_processor import HotPathMemoryProcessor
    from pipecat.frames.frames import TranscriptionFrame

    mem = HotPathMemoryProcessor(sqlite_path=":memory:", lmdb_dir=None, user_id="test-user", enable_metrics=False, context_aggregator=None)

    # Seed declarative color
    await mem.frame_processor._process_transcription("My favorite color is yellow.")

    # Ask the color question
    await mem.frame_processor._process_transcription("What is my favorite color?")

    # Inspect last injected bullets via the context injector
    # Without a context aggregator in this test, injection does not happen,
    # so check pending bullets captured by the injector.
    bullets = mem.context_injector._pending_bullets
    text = "\n".join(bullets).lower()
    logger.info(f"Injected bullets: {text}")
    assert 'favorite color' in text
    assert 'favorite number' not in text
    assert 'favorite music' not in text
