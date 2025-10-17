import os

os.environ.setdefault("PIPECAT_DISABLE_IMPORT", "1")

import pytest

from core.memory.background_summarizer import BackgroundSummarizer
from core.memory.config_manager import MemoryConfiguration


class StubStore:
    def __init__(self):
        self.notes = []
        self.flushed = False

    def enqueue_mention(self, *args):
        self.notes.append(args)

    def flush_if_needed(self):
        self.flushed = True


class StubHotMemory:
    def __init__(self, store):
        self.store = store


@pytest.mark.asyncio
async def test_summarize_turns_stores_summary(monkeypatch):
    config = MemoryConfiguration(
        summarization_enabled=True,
        summary_window_mode="turn_pairs",
        summary_turn_pairs=2,
    )
    store = StubStore()
    hot = StubHotMemory(store)
    summarizer = BackgroundSummarizer(hot_memory=hot, config=config, store=store)

    async def fake_llm(text: str):
        return "User likes pizza"

    monkeypatch.setattr(summarizer, "_get_conversation_chunks", lambda *args, **kwargs: [("User likes pizza", 0)])
    monkeypatch.setattr(summarizer, "_call_summarizer_llm", fake_llm)

    result = await summarizer.summarize_turns(turn_id=2, session_id="session-1")

    assert result is True
    assert store.notes
    assert store.flushed
