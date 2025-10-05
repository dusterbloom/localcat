from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


Triple = Tuple[str, str, str]


class GraphJudgeConfig:
    def __init__(self, enabled: bool, model_path: Optional[str], threshold: float):
        self.enabled = enabled
        self.model_path = model_path
        self.threshold = threshold
        self.model: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_env() -> "GraphJudgeConfig":
        enabled = os.getenv("YAML_GRAPH_JUDGE", "off").strip().lower() not in {"off", "false", "0"}
        model_path = os.getenv("YAML_GRAPH_JUDGE_MODEL", "").strip() or None
        thresh_env = os.getenv("YAML_GRAPH_JUDGE_THRESH", "").strip()
        try:
            threshold = float(thresh_env) if thresh_env else 0.40
        except Exception:
            threshold = 0.40
        cfg = GraphJudgeConfig(enabled, model_path, threshold)
        if model_path:
            try:
                cfg.model = json.loads(Path(model_path).read_text())
                # Allow model-embedded threshold when env isn’t specified
                if not thresh_env and isinstance(cfg.model, dict) and "threshold" in cfg.model:
                    cfg.threshold = float(cfg.model["threshold"])  # type: ignore[index]
            except Exception:
                cfg.model = None
        return cfg


def _safe_lower(x: Optional[str]) -> str:
    return (x or "").strip().lower()


class SchemaRegistry:
    """Optional relation→object-type schema loaded from JSON.

    JSON format example:
    {
      "live_in": ["GPE","LOC","FAC"],
      "works_at": ["ORG"],
      "travel_to": ["GPE","LOC"]
    }
    """

    def __init__(self, path: Optional[str] = None):
        self.map: Dict[str, Set[str]] = {}
        # Built-in light defaults (EN NER labels)
        builtin: Dict[str, List[str]] = {
            "live_in": ["GPE", "LOC", "FAC"],
            "arrive_at": ["FAC", "GPE"],
            "arrive_in": ["GPE", "LOC"],
            "travel_to": ["GPE", "LOC"],
            "works_at": ["ORG"],
            "present_to": ["ORG", "PERSON"],
            "meet_with": ["PERSON", "ORG"],
        }
        for k, v in builtin.items():
            self.map[_safe_lower(k)] = set(v)
        # Load external map if provided
        if path:
            try:
                data = json.loads(Path(path).read_text())
                for k, v in data.items():
                    if isinstance(v, list):
                        self.map[_safe_lower(k)] = set(str(x) for x in v)
            except Exception:
                pass

    @staticmethod
    def from_env() -> "SchemaRegistry":
        return SchemaRegistry(os.getenv("YAML_GRAPH_JUDGE_SCHEMA", "").strip() or None)

    def expected_types(self, rel: str) -> Set[str]:
        return self.map.get(_safe_lower(rel), set())


class LanguageLabelMap:
    """Map spaCy NER labels to coarse types for multiple languages.

    Coarse types we care about: PERSON, ORG, LOC (includes GPE/FAC).
    """

    def __init__(self):
        # Defaults largely align with spaCy label conventions
        self.map: Dict[str, Dict[str, Set[str]]] = {
            # English
            "en": {
                "PERSON": {"PERSON"},
                "ORG": {"ORG", "NORP"},
                "LOC": {"GPE", "LOC", "FAC"},
            },
            # German / French / Spanish / Italian: typically PER, LOC, ORG, MISC
            "de": {
                "PERSON": {"PER", "PERSON"},
                "ORG": {"ORG"},
                "LOC": {"LOC", "GPE", "FAC"},
            },
            "fr": {
                "PERSON": {"PER", "PERSON"},
                "ORG": {"ORG"},
                "LOC": {"LOC", "GPE", "FAC"},
            },
            "es": {
                "PERSON": {"PER", "PERSON"},
                "ORG": {"ORG"},
                "LOC": {"LOC", "GPE", "FAC"},
            },
            "it": {
                "PERSON": {"PER", "PERSON"},
                "ORG": {"ORG"},
                "LOC": {"LOC", "GPE", "FAC"},
            },
        }

    def detect(self, doc: Any) -> Tuple[bool, bool, bool]:
        """Return (has_person, has_org, has_loc) for any entity in doc matching object text later.
        We do not yet filter by object span here; filtering happens in build_features.
        """
        try:
            lang = _safe_lower(getattr(doc, "lang_", "en"))
        except Exception:
            lang = "en"
        # Override via env if needed
        env_lang = _safe_lower(os.getenv("YAML_GRAPH_JUDGE_LANG_OVERRIDE", ""))
        if env_lang:
            lang = env_lang
        labels = self.map.get(lang, self.map.get("en", {}))
        has_person = False
        has_org = False
        has_loc = False
        try:
            for ent in getattr(doc, "ents", []) or []:
                lab = getattr(ent, "label_", "")
                if not lab:
                    continue
                labu = lab.upper()
                if labu in labels.get("PERSON", set()):
                    has_person = True
                if labu in labels.get("ORG", set()):
                    has_org = True
                if labu in labels.get("LOC", set()):
                    has_loc = True
        except Exception:
            pass
        return has_person, has_org, has_loc


def build_features(s: str, r: str, d: str, doc: Any) -> Dict[str, float]:
    """Feature function shared by Lite and Distilled judges.
    Avoids expensive ops; tolerates missing NER.
    """
    r = _safe_lower(r)
    d = _safe_lower(d)
    feats: Dict[str, float] = {
        "bias": 1.0,
        "lexicalized": 1.0 if any(r.endswith(suf) for suf in ("_on", "_in", "_to", "_from", "_with", "_for", "_into")) else 0.0,
        "len_d": min(len(d) / 32.0, 1.5),
        "very_short_d": 1.0 if 0 < len(d) <= 3 else 0.0,
        "empty_d": 1.0 if len(d) == 0 else 0.0,
        "rel_strong": 1.0 if r in {"work_on", "focus_on", "agree_on", "agree_with", "agree_to", "result_in", "stem_from", "lead_to", "apply_to", "apply_for", "comply_with", "adhere_to", "engage_in", "engage_with", "enter_into", "consist_of", "consist_in"} else 0.0,
    }
    generic_objs = {"thing", "things", "stuff", "issue", "issues", "something", "anything", "everything"}
    feats["generic_head"] = 1.0 if any(w in generic_objs for w in d.split()) else 0.0
    feats["pron_d"] = 1.0 if d in {"it", "this", "that", "there", "something", "anything"} else 0.0

    # Minimal type hints using NER (optional)
    # Language-aware entity type cues
    has_person_any, has_org_any, has_loc_any = LanguageLabelMap().detect(doc)
    has_loc = False
    has_org = False
    has_person = False
    try:
        for ent in getattr(doc, "ents", []) or []:
            txt = _safe_lower(getattr(ent, "text", ""))
            if txt and txt in d:
                lab = getattr(ent, "label_", "")
                labu = lab.upper()
                # Use language-aware booleans but require surface containment
                if has_loc_any and labu in {"GPE", "LOC", "FAC"}:
                    has_loc = True
                if has_org_any and labu in {"ORG", "NORP"}:
                    has_org = True
                if has_person_any and labu in {"PERSON", "PER"}:
                    has_person = True
    except Exception:
        pass

    # Relation schema compatibility hints (cheap priors)
    schema = SchemaRegistry.from_env()
    exp = schema.expected_types(r)
    # Checks by coarse type buckets available via spaCy NER
    loc_ok = any(t in exp for t in ("GPE", "LOC", "FAC"))
    org_ok = any(t in exp for t in ("ORG", "NORP"))
    per_ok = any(t in exp for t in ("PERSON", "PER"))
    feats["type_loc_match"] = 1.0 if (loc_ok and has_loc) else 0.0
    feats["type_org_match"] = 1.0 if (org_ok and has_org) else 0.0
    feats["type_person_match"] = 1.0 if (per_ok and has_person) else 0.0
    feats["schema_any"] = 1.0 if exp else 0.0

    # Contentfulness: ratio of alphabetic tokens (approx)
    alpha = sum(ch.isalpha() for ch in d)
    feats["content_ratio"] = (alpha / max(1.0, len(d))) if d else 0.0
    return feats


def lite_score(feats: Dict[str, float]) -> float:
    weights = {
        "bias": 0.10,
        "lexicalized": 0.60,
        "len_d": 0.20,
        "very_short_d": -0.60,
        "empty_d": -1.00,
        "generic_head": -0.80,
        "pron_d": -0.70,
        "rel_strong": 0.70,
        "type_loc_match": 0.60,
        "type_org_match": 0.60,
        "type_person_match": 0.50,
        "schema_any": 0.05,
        "content_ratio": 0.30,
    }
    score = sum(weights.get(k, 0.0) * v for k, v in feats.items())
    score = 0.5 + 0.5 * score
    return 0.0 if score < 0.0 else 1.0 if score > 1.0 else score


class GraphJudge:
    def __init__(self, cfg: GraphJudgeConfig):
        self.cfg = cfg

    @staticmethod
    def from_env() -> "GraphJudge":
        return GraphJudge(GraphJudgeConfig.from_env())

    def enabled(self) -> bool:
        return self.cfg.enabled

    def score(self, s: str, r: str, d: str, doc: Any) -> float:
        feats = build_features(s, r, d, doc)
        if self.cfg.model and isinstance(self.cfg.model, dict) and "weights" in self.cfg.model:
            w = self.cfg.model.get("weights", {})
            b = float(self.cfg.model.get("intercept", 0.0))
            z = b + sum(float(w.get(k, 0.0)) * float(v) for k, v in feats.items())
            try:
                return 1.0 / (1.0 + math.exp(-z))
            except OverflowError:
                return 1.0 if z > 0 else 0.0
        return lite_score(feats)

    def filter(self, triples: List[Triple], doc: Any, threshold: Optional[float] = None) -> List[Triple]:
        if not self.enabled():
            return list(triples)
        thr = self.cfg.threshold if threshold is None else threshold
        out: List[Triple] = []
        for s, r, d in triples:
            if self.score(s, r, d, doc) >= thr:
                out.append((s, r, d))
        return out
