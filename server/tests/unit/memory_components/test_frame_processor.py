import os

os.environ.setdefault("PIPECAT_DISABLE_IMPORT", "1")

import pytest

from core.memory.config_manager import MemoryConfiguration
from core.memory.frame_processor import (
    FrameDirection,
    MemoryFrameProcessor,
    TranscriptionFrame,
)


class StubContextInjector:
    def __init__(self):
        self.pending = []
        self.inject_calls = 0
        self.reset_calls = 0

    def set_pending_bullets(self, bullets):
        self.pending = list(bullets)

    def should_refresh_injection(self):
        return bool(self.pending)

    async def inject_memory_context(self):
        self.inject_calls += 1
        return True

    def reset_turn_state(self):
        self.reset_calls += 1


class StubSessionManager:
    def __init__(self):
        self.session_id = "session-1"
        self.user_eid = "alice"
        self.turns = 0
        self.recorded = []

    def increment_turn(self):
        self.turns += 1

    def record_turn_metrics(self, elapsed_ms):
        self.recorded.append(elapsed_ms)
        return {"session_turns": self.turns}


class StubStore:
    def __init__(self):
        self.mentions = []
        self.flushed = 0

    def enqueue_mention(self, *args):
        self.mentions.append(args)

    def flush_if_needed(self):
        self.flushed += 1


class StubHotMemory:
    def __init__(self):
        self.store = StubStore()
        self.processed = []

    def process_turn(self, text, session_id, turn_id, focus="standard", intent=None, prosody_features=None):
        self.processed.append((text, session_id, turn_id, focus))
        return ["Likes pizza"], [("alice", "likes", "pizza")]


@pytest.mark.asyncio
async def test_final_transcription_processes_turn_and_injects():
    config = MemoryConfiguration(enabled=True)
    context_injector = StubContextInjector()
    session_manager = StubSessionManager()
    hot_memory = StubHotMemory()

    processor = MemoryFrameProcessor(
        config=config,
        context_injector=context_injector,
        session_manager=session_manager,
        hot_memory=hot_memory,
        intent_service=None,
    )

    frame = TranscriptionFrame()
    frame.is_final = True
    frame.text = "I like pizza"

    emitted = []
    async for outgoing in processor.process_frame(frame, FrameDirection.DOWNSTREAM):
        emitted.append(outgoing)

    # The original frame should flow through
    assert emitted[0] is frame
    # Hot memory received the turn
    assert hot_memory.processed
    # Context injection executed
    assert context_injector.inject_calls == 1
    # Conversation mention stored
    assert hot_memory.store.mentions
    # Session turn recorded
    assert session_manager.turns == 1
