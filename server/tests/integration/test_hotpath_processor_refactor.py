import os

os.environ.setdefault("PIPECAT_DISABLE_IMPORT", "1")

import pytest

from core.memory.config_manager import MemoryConfiguration
from core.memory.frame_processor import (
    FrameDirection,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from core.memory.hotpath_processor import HotPathMemoryProcessor, MemoryContextReadyFrame


class RecordingProcessor(HotPathMemoryProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitted = []

    async def push_frame(self, frame, direction):
        self.emitted.append((frame, direction))


class StubStore:
    def __init__(self):
        self.mentions = []
        self.flushed = 0

    def enqueue_mention(self, *args):
        self.mentions.append(args)

    def flush_if_needed(self):
        self.flushed += 1

    def get_metrics(self):
        return {}


class StubHotMemory:
    def __init__(self, store):
        self.store = store
        self.processed = []
        self.retrieve_calls = []
        self.agent_eid = None
        self.current_user_id = None
        self.current_session_id = None

    def prewarm(self, lang):
        return

    def rebuild_from_store(self):
        return

    def process_turn(self, text, session_id, turn_id, focus="standard", intent=None):
        self.processed.append((text, session_id, turn_id))
        return ["Likes pizza"], []

    def retrieve_bullets(self, query, read_only=True, intent=None):
        self.retrieve_calls.append(query)
        return ["Likes pizza"]

    def get_metrics(self):
        return {"total_ms": {"p95": 10}}


class DummyContext:
    def __init__(self):
        self._messages = [{"role": "system", "content": "[Persona]\nBe helpful."}]

    def get_messages(self):
        return list(self._messages)

    def set_messages(self, messages):
        self._messages = list(messages)


class DummyAggregatorUser:
    def __init__(self, context):
        self.context = context


class DummyAggregator:
    def __init__(self):
        self._context = DummyContext()

    def user(self):
        return DummyAggregatorUser(self._context)


@pytest.mark.asyncio
async def test_hotpath_processor_integration_flow():
    config = MemoryConfiguration(
        enabled=True,
        bullets_max=2,
        interim_min_words=1,
        summarization_enabled=False,
        handshake_enabled=True,
    )
    store = StubStore()
    hot = StubHotMemory(store)
    aggregator = DummyAggregator()

    processor = RecordingProcessor(
        config=config,
        context_aggregator=aggregator,
        hot_memory=hot,
        memory_store=store,
    )

    start = StartFrame()
    interim = InterimTranscriptionFrame()
    interim.text = "interim words enough"

    final = TranscriptionFrame()
    final.is_final = True
    final.text = "I like pizza"

    await processor.process_frame(start, FrameDirection.DOWNSTREAM)
    await processor.process_frame(interim, FrameDirection.DOWNSTREAM)
    await processor.process_frame(final, FrameDirection.DOWNSTREAM)

    # Memory bullets injected into aggregator context
    messages = aggregator.user().context.get_messages()
    assert any(processor.config.inject_header in msg.get("content", "") for msg in messages)

    # MemoryContextReadyFrame emitted
    handshake_frames = [frame for frame, _ in processor.emitted if isinstance(frame, MemoryContextReadyFrame)]
    assert handshake_frames

    # Hot memory received processing calls
    assert hot.processed
    assert hot.retrieve_calls
