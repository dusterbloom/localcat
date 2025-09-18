#!/usr/bin/env python3
"""SurrealDB smoke test (optional).

Skips automatically if the surrealdb client is not installed or if
SURREALDB_URL is not configured.
"""

import os
import pytest


surreal = pytest.importorskip("surrealdb", reason="surrealdb client not installed")

if not os.getenv("SURREALDB_URL"):
    pytest.skip("SURREALDB_URL not set; skipping surreal smoke", allow_module_level=True)


@pytest.mark.asyncio
async def test_surreal_smoke():
    # Minimal write/read roundtrip on mention table (used by FTS)
    from adapters.surreal_memory_store import SurrealMemoryStore

    store = SurrealMemoryStore.from_env()
    # Enqueue a mention and flush
    store.enqueue_mention("session:test", "hello surreal", 1, "session:test", 1)
    await store.flush_async()

    # Query back via async path
    rows = await store.search_fts_detailed_async("hello", 5)
    assert rows, "Expected at least one FTS row"
    assert any("hello surreal" in r[0] for r in rows)

