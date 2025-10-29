#!/usr/bin/env python
import os
import tempfile
import sys
import pytest


_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for p in (_SERVER_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)


try:
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.memory_hotpath import HotMemory
    HOTMEM_AVAILABLE = True
except Exception:
    HOTMEM_AVAILABLE = False


@pytest.mark.fast
def test_slot_aware_color_query_filters_non_color_convo():
    if not HOTMEM_AVAILABLE:
        pytest.skip("HotMem not available")

    os.environ['MEMORY_SOURCES'] = 'convo,graph,summary'
    os.environ['MEMORY_CONVO_INDEX'] = 'true'
    os.environ['MEMORY_MAX_BULLETS'] = '2'

    td = tempfile.mkdtemp(prefix='hotmem_test_')
    store = MemoryStore(Paths(sqlite_path=os.path.join(td, 'mem.db'), lmdb_dir=None))
    hot = HotMemory(store)

    s = 'test_sess'
    turn = 1
    seeds = [
        "I want you to know that my favorite number is 63.",
        "I want you to know that my favorite music is rock and roll.",
        "I want you to remember that my favorite color is yellow.",
    ]
    for t in seeds:
        hot.process_turn(t, s, turn)
        turn += 1

    bullets, triples = hot.process_turn("What is my favorite color?", s, turn)
    text = "\n".join(bullets)
    # Should include only color domain; must not include number/music
    assert 'favorite color' in text.lower()
    assert 'favorite number' not in text.lower()
    assert 'favorite music' not in text.lower()


@pytest.mark.fast
def test_uk_variant_colour_is_canonicalized():
    if not HOTMEM_AVAILABLE:
        pytest.skip("HotMem not available")

    os.environ['MEMORY_SOURCES'] = 'convo,graph,summary'
    os.environ['MEMORY_CONVO_INDEX'] = 'true'
    os.environ['MEMORY_MAX_BULLETS'] = '2'

    td = tempfile.mkdtemp(prefix='hotmem_test_')
    store = MemoryStore(Paths(sqlite_path=os.path.join(td, 'mem.db'), lmdb_dir=None))
    hot = HotMemory(store)

    s = 'uk_sess'
    hot.process_turn("My favourite colour is yellow.", s, 1)
    bullets, triples = hot.process_turn("What is my favourite colour?", s, 2)
    text = "\n".join(bullets).lower()
    assert 'favourite colour' in text or 'favorite color' in text


@pytest.mark.fast
def test_number_only_seed_then_color_query_returns_empty():
    if not HOTMEM_AVAILABLE:
        pytest.skip("HotMem not available")

    os.environ['MEMORY_SOURCES'] = 'convo,graph,summary'
    os.environ['MEMORY_CONVO_INDEX'] = 'true'
    os.environ['MEMORY_MAX_BULLETS'] = '2'

    td = tempfile.mkdtemp(prefix='hotmem_test_')
    store = MemoryStore(Paths(sqlite_path=os.path.join(td, 'mem.db'), lmdb_dir=None))
    hot = HotMemory(store)

    s = 'only_num'
    hot.process_turn("My favorite number is 63.", s, 1)
    bullets, _ = hot.process_turn("What is my favorite color?", s, 2)
    # No cross-slot leakage
    assert len(bullets) == 0


@pytest.mark.fast
def test_slot_number_and_music_queries_are_slot_aligned():
    if not HOTMEM_AVAILABLE:
        pytest.skip("HotMem not available")

    os.environ['MEMORY_SOURCES'] = 'convo,graph,summary'
    os.environ['MEMORY_CONVO_INDEX'] = 'true'
    os.environ['MEMORY_MAX_BULLETS'] = '2'

    td = tempfile.mkdtemp(prefix='hotmem_test_')
    store = MemoryStore(Paths(sqlite_path=os.path.join(td, 'mem.db'), lmdb_dir=None))
    hot = HotMemory(store)

    s = 'mix'
    turn = 1
    seeds = [
        "My favorite number is 63.",
        "My favorite music is rock and roll.",
        "My favorite color is yellow.",
    ]
    for t in seeds:
        hot.process_turn(t, s, turn)
        turn += 1

    b_num, _ = hot.process_turn("What is my favorite number?", s, turn); turn += 1
    num_text = "\n".join(b_num).lower()
    assert 'favorite number' in num_text
    assert 'favorite color' not in num_text
    assert 'favorite music' not in num_text

    b_music, _ = hot.process_turn("What is my favorite music?", s, turn)
    music_text = "\n".join(b_music).lower()
    assert 'favorite music' in music_text
    assert 'favorite color' not in music_text
    assert 'favorite number' not in music_text
