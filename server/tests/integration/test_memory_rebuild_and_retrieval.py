#!/usr/bin/env python3
"""
Integration tests: rebuild-from-store, retrieval, and session storage

Validates that:
- Seeded graph edges persist to MemoryStore and can be rebuilt into RAM indices
- Retrieval returns meaningful bullets after rebuild (no cold start issue)
- Verbatim messages and summaries store correctly in SessionStore
"""

import os
import sys
import time
import contextlib

import pytest

# Make imports work regardless of CWD (repo root or server/)
HERE = os.path.dirname(__file__)
SERVER_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_ROOT, ".."))
for p in (SERVER_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade
from components.session.session_store import get_session_store, reset_session_store


@contextlib.contextmanager
def chdir(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def test_rebuild_from_store_and_retrieval(tmp_path):
    """Seeds edges, rebuilds indices, then retrieves relevant bullets."""
    with chdir(tmp_path.as_posix()):
        # Prepare durable paths
        sqlite_path = tmp_path / "memory.db"
        lmdb_dir = tmp_path / "graph.lmdb"
        store = MemoryStore(Paths(sqlite_path=str(sqlite_path), lmdb_dir=str(lmdb_dir)))

        # Seed a few edges directly (simulate previously persisted facts)
        now = int(time.time() * 1000)
        store.observe_edge("you", "lives_in", "Sardinia", 0.9, now)
        store.observe_edge("you", "works_at", "Acme", 0.8, now + 1)
        store.flush()

        # Facade + rebuild from store to warm up RAM indices
        facade = HotMemoryFacade(store)
        facade.rebuild_from_store()

        # Validate indices populated
        assert "you" in facade.entity_index
        assert any(tr[1] == "lives_in" for tr in facade.entity_index["you"])

        # Run retrieval for a question that should use memory
        result = facade.retriever.retrieve_context("Where do I live?", ["you"], 1)
        assert isinstance(result.bullets, list)
        assert len(result.bullets) >= 1
        assert any("sardinia" in b.lower() for b in result.bullets)


def test_verbatim_and_summary_storage(tmp_path):
    """Verifies verbatim messages and session summaries are stored."""
    with chdir(tmp_path.as_posix()):
        # Isolate session DB under tmp
        reset_session_store()
        session_db = tmp_path / "sessions.db"
        sess = get_session_store(str(session_db))

        # Create session and store conversation
        session_id = sess.create_session("tester")
        sess.add_message(session_id, "user", "Hello there", 1)
        sess.add_message(session_id, "assistant", "Hi! How can I help?", 1)
        sess.add_message(session_id, "user", "I live in Sardinia", 2)
        sess.add_message(session_id, "assistant", "Great, noted.", 2)

        # Add a summary
        sess.add_session_summary(session_id, "User lives in Sardinia; greeted and acknowledged.", "auto")

        # Validate persisted data
        convo = sess.get_session_conversation(session_id)
        meta = sess.get_session_metadata(session_id)
        assert len(convo) == 4
        assert meta.summary is not None and "Sardinia" in meta.summary


def test_fts_summary_aids_retrieval(tmp_path):
    """Stores a summary in MemoryStore FTS and verifies it contributes to retrieval bullets."""
    with chdir(tmp_path.as_posix()):
        sqlite_path = tmp_path / "memory.db"
        lmdb_dir = tmp_path / "graph.lmdb"
        store = MemoryStore(Paths(sqlite_path=str(sqlite_path), lmdb_dir=str(lmdb_dir)))

        # Insert a summary mention like the periodic summarizer would
        sid = "session_test"
        ts = int(time.time() * 1000)
        store.enqueue_mention(eid=f"summary:{sid}", text="You live in Sardinia and work at Acme.", ts=ts, sid=sid, tid=2)
        store.flush()

        # Facade + rebuild so retriever has indices (even if graph is empty)
        facade = HotMemoryFacade(store)
        facade.rebuild_from_store()

        # Retrieval should surface a summary-derived bullet
        result = facade.retriever.retrieve_context("Where do I live?", ["you"], 1)
        assert isinstance(result.bullets, list)
        assert len(result.bullets) >= 1
        assert any("sardinia" in b.lower() for b in result.bullets)


if __name__ == "__main__":
    # Run standalone without pytest/conftest to avoid unrelated import issues
    import tempfile
    from pathlib import Path

    print("Running standalone integration checks...")
    with tempfile.TemporaryDirectory() as d1:
        test_rebuild_from_store_and_retrieval(Path(d1))
        print("✅ rebuild_from_store_and_retrieval passed")
    with tempfile.TemporaryDirectory() as d2:
        test_verbatim_and_summary_storage(Path(d2))
        print("✅ verbatim_and_summary_storage passed")
    with tempfile.TemporaryDirectory() as d3:
        test_fts_summary_aids_retrieval(Path(d3))
        print("✅ fts_summary_aids_retrieval passed")
    print("🎉 All standalone integration checks passed")
