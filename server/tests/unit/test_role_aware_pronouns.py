import os
import pytest


@pytest.mark.unit
def test_user_first_person_maps_to_user_and_second_person_to_agent():
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.memory_hotpath import HotMemory

    # In-memory store; no LMDB
    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store)

    # Provide role-aware IDs
    hot.user_eid = "you:peppi"
    hot.agent_eid = "agent:locat"

    # User utterance
    text = "I thank you for your help."
    bullets, triples = hot.process_turn(text, session_id="sess-1", turn_id=1)

    # Expect a triple where subject is the user and object is the agent
    assert any(
        s.startswith("you:") and (r == "thank" or r == "v:thank") and d.startswith("agent:")
        for (s, r, d) in triples
    ), f"Missing expected thank triple in {triples}"


@pytest.mark.unit
def test_user_second_person_copula_maps_to_agent_subject():
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.memory_hotpath import HotMemory

    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store)
    hot.user_eid = "you:peppi"
    hot.agent_eid = "agent:locat"

    text = "You are helpful."
    bullets, triples = hot.process_turn(text, session_id="sess-1", turn_id=2)

    # Expect (agent:locat, is, helpful)
    assert any(
        s.startswith("agent:") and r == "is" and d.startswith("helpful")
        for (s, r, d) in triples
    ), f"Missing expected copula triple in {triples}"


@pytest.mark.unit
def test_bullet_humanization_displays_names_not_ids():
    from core.memory.retrieval import Retrieval

    class _Host:
        pass

    host = _Host()
    r = Retrieval(host)

    # Directly test humanization path
    human = r._humanize_fact("you:peppi", "thank", "agent:locat")
    assert "peppi" in human and "locat" in human and "you:" not in human and "agent:" not in human

