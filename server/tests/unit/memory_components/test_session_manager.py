import os

os.environ.setdefault("PIPECAT_DISABLE_IMPORT", "1")

import pytest

from core.memory.config_manager import MemoryConfiguration
from core.memory.session_manager import SessionManager


class DummyContext:
    def __init__(self):
        self._messages = []

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


def test_ensure_session_header_inserts_and_updates():
    config = MemoryConfiguration()
    aggregator = DummyAggregator()
    manager = SessionManager(
        session_id="session-1",
        user_eid="alice",
        agent_eid="bot",
        config=config,
    )

    manager.ensure_session_header(aggregator)

    messages = aggregator.user().context.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"].startswith("[Session Context]")

    # Update call should replace existing header
    manager.ensure_session_header(aggregator)
    messages_after = aggregator.user().context.get_messages()
    assert len(messages_after) == 1


def test_ensure_session_header_respects_ephemeral_mode():
    config = MemoryConfiguration(ephemeral_mode=True)
    aggregator = DummyAggregator()
    manager = SessionManager(
        session_id="session-ephemeral",
        user_eid="ephemeral",
        agent_eid="bot",
        config=config,
    )

    # Pre-populate a header to ensure it gets removed
    aggregator.user().context.set_messages(
        [{"role": "system", "content": "[Session Context]\nUser: ephemeral"}]
    )

    manager.ensure_session_header(aggregator)

    assert aggregator.user().context.get_messages() == []
