import os

os.environ.setdefault("PIPECAT_DISABLE_IMPORT", "1")

import pytest

from core.memory.config_manager import (
    MemoryConfiguration,
    get_memory_config,
    reload_memory_config,
)


def test_from_env_reads_values(monkeypatch):
    monkeypatch.setenv("MEMORY_BULLETS_MAX", "5")
    monkeypatch.setenv("MEMORY_SOURCES", "convo,graph")
    monkeypatch.setenv("MEMORY_MAX_TURN_PAIRS", "6")

    config = MemoryConfiguration.from_env()

    assert config.bullets_max == 5
    assert config.sources == ["convo", "graph"]
    assert config.max_turn_pairs == 6


def test_validate_warns_on_outliers():
    config = MemoryConfiguration(
        bullets_max=0,
        retrieval_timeout_ms=200,
        sources=[],
        token_budget=50,
    )

    config.sources = []
    warnings = config.validate()

    assert any("bullets_max" in warning for warning in warnings)
    assert any("retrieval_timeout_ms" in warning for warning in warnings)
    assert any("No retrieval sources" in warning for warning in warnings)
    assert any("token_budget" in warning for warning in warnings)


def test_singleton_helpers(monkeypatch):
    monkeypatch.delenv("MEMORY_BULLETS_MAX", raising=False)
    reload_memory_config()

    first = get_memory_config()
    second = get_memory_config()

    assert first is second
    assert isinstance(first, MemoryConfiguration)
