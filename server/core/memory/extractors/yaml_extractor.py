"""
YAMLExtractor: Adapter implementing Extractor interface using YAMLRuntime.

Usage (dev/tests only):
    extractor = YAMLExtractor(yaml_path="/path/to/ASI1_proposal.yaml")
    entities, triples, neg, doc = extractor.extract(text, lang="en")

This is not wired into the hot path; it is for the YAML superiority proof.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Optional
from loguru import logger

from .base import Extractor
from .yaml_runtime import YAMLRuntime


def _canon_entity_text(text: str) -> str:
    t = (text or "").strip().lower()
    for det in ("the", "a", "an", "my", "your", "his", "her", "our", "their", "its"):
        if t.startswith(det + " "):
            t = t[len(det) + 1 :]
            break
    if t.endswith("'s"):
        t = t[:-2]
    if t in {"i", "me", "my", "mine", "myself"}:
        return "you"
    return t


class YAMLExtractor(Extractor):
    def __init__(self, yaml_path: str):
        self.runtime = YAMLRuntime(yaml_path)

    def prewarm(self, lang: str = "en") -> None:
        # YAMLRuntime lazily loads spaCy per call; nothing to prewarm here.
        pass

    def extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        try:
            entities, triples, neg, doc = self.runtime.extract(text, lang)
            return entities, triples, neg, doc
        except Exception as e:
            logger.warning(f"YAML extraction failed: {e}")
            return [], [], 0, None

    def refine(self, text: str, triples: List[Tuple[str, str, str]], doc: Any) -> List[Tuple[str, str, str]]:
        # Keep simple for dev: drop trivial relations and scaffolding
        out = []
        stop_rel = {"and", "tell", "say"}
        stop_ent = {"it", "this", "that"}
        for s, r, d in triples:
            s2, d2 = _canon_entity_text(s), _canon_entity_text(d)
            r2 = (r or "").strip().lower()
            if not s2 or not d2 or r2 in stop_rel or s2 in stop_ent or d2 in stop_ent:
                continue
            out.append((s2, r2, d2))
        # de‑dupe
        seen = set()
        uniq = []
        for tr in out:
            if tr not in seen:
                uniq.append(tr)
                seen.add(tr)
        return uniq

    def refine_entities(self, text: str, entities: List[str]) -> List[str]:
        out = []
        seen = set()
        for e in entities:
            ce = _canon_entity_text(e)
            if ce and ce not in seen:
                out.append(ce)
                seen.add(ce)
        return out
