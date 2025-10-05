#!/usr/bin/env python3
import time

from core.memory.hotmem_service import HotMemService
from core.audio.prosody_analyzer import ProsodyFeatures


def test_prosody_pass_through_and_edge_meta():
    svc = HotMemService(user_id="test-user", sqlite_path=":memory:", lmdb_dir=None)
    # Provide prosody for the next turn
    pros = ProsodyFeatures(
        pitch_mean=180.0,
        pitch_std=20.0,
        pitch_slope=-12.0,
        intensity_mean=60.0,
        intensity_peak=72.0,
        speaking_rate=4.2,
        pause_count=0,
        duration_sec=1.6,
        certainty_modifier=0.2,
    )
    svc.set_prosody_for_turn(pros)

    # Store a simple message that yields a triple
    svc._store_messages([{"role": "user", "content": "John read the book."}])

    # Verify meta contains prosody_certainty
    cur = svc.store.sql.cursor()
    rows = cur.execute("SELECT meta FROM edge").fetchall()
    assert rows, "No edges persisted"
    meta_json = rows[0][0] or "{}"
    assert "prosody_certainty" in meta_json, f"Missing prosody in meta: {meta_json}"

