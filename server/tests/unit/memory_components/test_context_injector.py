import os

os.environ.setdefault("PIPECAT_DISABLE_IMPORT", "1")

import pytest

from core.memory.config_manager import MemoryConfiguration
from core.memory.context_injector import ContextInjector
from core.memory.context_formatter import ContextFormatter


class StubHotMemory:
    def __init__(self, bullets=None):
        self._bullets = bullets or ["Likes pizza", "Lives in SF"]
        self.current_session_id = None
        self.current_user_id = None

    def retrieve_bullets(self, query, read_only=True, intent=None):
        return list(self._bullets)


class DummyContext:
    def __init__(self):
        self._messages = [
            {"role": "system", "content": "[Persona]\nHelpful assistant."},
            {"role": "user", "content": "Hi"},
        ]

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
async def test_retrieve_and_inject_memory_context():
    config = MemoryConfiguration(bullets_max=2)
    aggregator = DummyAggregator()
    hot = StubHotMemory()
    injector = ContextInjector(
        hot_memory=hot,
        config=config,
        formatter=ContextFormatter(
            max_bullets=config.bullets_max,
            inject_role=config.inject_role,
            inject_header=config.inject_header,
        ),
        context_aggregator=aggregator,
    )

    bullets = await injector.retrieve_and_prepare_bullets("What do I like?", read_only=True)
    assert bullets == ["Likes pizza", "Lives in SF"]

    success = await injector.inject_memory_context()
    assert success

    messages = aggregator.user().context.get_messages()
    assert any(config.inject_header in msg.get("content", "") for msg in messages)
