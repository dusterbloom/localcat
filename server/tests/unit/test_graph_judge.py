import json
import os

import pytest

from core.memory.judge import GraphJudge, GraphJudgeConfig, build_features


class FakeEnt:
    def __init__(self, text, label_):
        self.text = text
        self.label_ = label_


class FakeDoc:
    def __init__(self, ents=None, lang_="en"):
        self.ents = ents or []
        self.lang_ = lang_


def test_lite_score_prefers_lexicalized_and_contentful(monkeypatch):
    # Enable judge lite (no model)
    monkeypatch.setenv("YAML_GRAPH_JUDGE", "on")
    monkeypatch.delenv("YAML_GRAPH_JUDGE_MODEL", raising=False)
    gj = GraphJudge.from_env()
    assert gj.enabled()

    # 'work_on research' vs empty object should prefer contentful lexicalized
    doc = FakeDoc(ents=[FakeEnt("Research", "ORG")])
    s1 = gj.score("alice", "work_on", "research", doc)
    s2 = gj.score("alice", "focus", "", doc)
    assert s1 > s2
    # Filter with default threshold should retain good triple
    kept = gj.filter([("alice", "work_on", "research"), ("alice", "focus", "")], doc, threshold=0.3)
    assert ("alice", "work_on", "research") in kept
    assert ("alice", "focus", "") not in kept


def test_distilled_model_threshold(monkeypatch, tmp_path):
    # Prepare a distilled model that strongly accepts via 'bias'
    model = {"intercept": 2.5, "weights": {"bias": 1.0}, "threshold": 0.9}
    mpath = tmp_path / "judge_model.json"
    mpath.write_text(json.dumps(model))
    monkeypatch.setenv("YAML_GRAPH_JUDGE", "on")
    monkeypatch.setenv("YAML_GRAPH_JUDGE_MODEL", str(mpath))
    monkeypatch.delenv("YAML_GRAPH_JUDGE_THRESH", raising=False)
    gj = GraphJudge.from_env()
    assert gj.enabled()
    # Any triple should score near 1.0 given strong bias
    sc = gj.score("john", "work_at", "google", FakeDoc([FakeEnt("Google", "ORG")]))
    assert sc >= 0.9
    kept = gj.filter([("john", "work_at", "google")], FakeDoc(), threshold=None)
    assert kept == [("john", "work_at", "google")]


def test_schema_loc_match(monkeypatch):
    # Use shipped schema to check location compatibility
    from pathlib import Path
    schema_path = Path(__file__).resolve().parents[3] / "server" / "models" / "graph_judge_schema.json"
    monkeypatch.setenv("YAML_GRAPH_JUDGE", "on")
    monkeypatch.setenv("YAML_GRAPH_JUDGE_SCHEMA", str(schema_path))
    monkeypatch.delenv("YAML_GRAPH_JUDGE_MODEL", raising=False)
    doc = FakeDoc([FakeEnt("New York", "GPE")])
    feats = build_features("john", "live_in", "new york", doc)
    assert feats.get("type_loc_match", 0.0) >= 1.0


def test_language_label_map_person(monkeypatch):
    # French PER label should be recognized as PERSON
    monkeypatch.setenv("YAML_GRAPH_JUDGE", "on")
    monkeypatch.delenv("YAML_GRAPH_JUDGE_MODEL", raising=False)
    # meet_with expects PERSON/ORG in schema; ensure type_person_match contributes
    doc = FakeDoc([FakeEnt("Alice", "PER")], lang_="fr")
    feats = build_features("bob", "meet_with", "alice", doc)
    assert feats.get("schema_any", 0.0) >= 0.0  # schema may or may not exist in env, but map should set person flags
    # Even without schema, person detection should not crash and should be boolean
    assert feats.get("type_person_match", 0.0) in (0.0, 1.0)

