"""
YAMLExtractor: Adapter implementing Extractor interface using YAMLRuntime.

Usage (dev/tests only):
    extractor = YAMLExtractor(yaml_path="/path/to/ASI1_proposal.yaml")
    entities, triples, neg, doc = extractor.extract(text, lang="en")

This is not wired into the hot path; it is for the YAML superiority proof.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Optional, Dict
import os
from loguru import logger

from .base import Extractor
from .yaml_runtime import YAMLRuntime
from core.memory.judge import GraphJudge
from core.memory.canonicalize import canonicalize_triple


def _canon_entity_text(text: str) -> str:
    t = (text or "").strip().lower()
    # strip common determiners across en/es/fr/de/it
    for det in (
        "the", "a", "an", "my", "your", "his", "her", "our", "their", "its",
        "el", "la", "los", "las", "un", "una", "unos", "unas",  # es
        "le", "la", "les", "un", "une", "des", "l'", "l’",          # fr
        "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "eines",  # de
        "il", "lo", "la", "i", "gli", "le", "uno", "una", "un"  # it
    ):
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
        # Post-filter and cap density per sentence to align with MEMORY_SYSTEM_MAP
        stop_rel = {"and", "tell", "say"}
        stop_ent = {"it", "this", "that"}

        # 1) Basic cleanup and normalization
        cleaned: List[Tuple[str, str, str, int]] = []  # (s, r, d, sent_idx)
        sents = list(getattr(doc, "sents", []) or []) if doc is not None else []

        def sent_index_for(s: str, r: str, d: str) -> int:
            # Heuristic: prefer sentence containing relation, else subject, else object
            if not sents:
                return 0
            srch = [(r or "").lower(), (s or "").lower(), (d or "").lower()]
            for i, sp in enumerate(sents):
                txt = sp.text.lower()
                if srch[0] and srch[0] in txt:
                    return i
            for i, sp in enumerate(sents):
                txt = sp.text.lower()
                if srch[1] and srch[1] in txt:
                    return i
            for i, sp in enumerate(sents):
                txt = sp.text.lower()
                if srch[2] and srch[2] in txt:
                    return i
            return 0

        for s, r, d in triples:
            s2, d2 = _canon_entity_text(s), _canon_entity_text(d)
            r2 = (r or "").strip().lower()
            if not s2 or not r2 or r2 in stop_rel or s2 in stop_ent:
                continue
            # Drop degenerate copula self-pairs (e.g., 'car is car')
            if r2 in {"is", "be"} and s2 == d2:
                continue
            if d2 in stop_ent:
                d2 = ""
            # Canonicalize relation/object (verb+prep folding, copula)
            s2, r2, d2 = canonicalize_triple(s2, r2, d2)
            si = sent_index_for(s2, r2, d2)
            cleaned.append((s2, r2, d2, si))

        # Regex assist for copula adjectival (ensures 'the car is red' ⇒ (car, be, red))
        import re as _re
        pattern = _re.compile(r"\b(?:the|a|an)\s+([A-Za-z][\w-]*)\s+(?:is|are|was|were)\s+([A-Za-z][\w-]*)\b", _re.IGNORECASE)
        for m in pattern.finditer(text or ""):
            s2 = _canon_entity_text(m.group(1))
            d2 = _canon_entity_text(m.group(2))
            if s2 and d2 and s2 != d2:
                si = sent_index_for(s2, "be", d2)
                if (s2, "be", d2, si) not in cleaned:
                    cleaned.append((s2, "be", d2, si))

        # 2) Optional singularization for subjects/objects (env‑gated, default ON in eval)
        def singularize_word(w: str) -> str:
            wl = w.lower()
            # simple, conservative rules
            if len(wl) <= 3:
                return w
            if wl.endswith("ies") and len(wl) > 4:
                return w[:-3] + "y"
            if wl.endswith("sses") or wl.endswith("xes") or wl.endswith("ches") or wl.endswith("shes") or wl.endswith("zes"):
                return w[:-2]
            if wl.endswith("s") and not wl.endswith("ss"):
                return w[:-1]
            return w

        def singularize_phrase(p: str) -> str:
            if not p:
                return p
            toks = p.split()
            if not toks:
                return p
            # only singularize the head (last token)
            toks[-1] = singularize_word(toks[-1])
            return " ".join(toks)

        do_sing = (os.getenv("YAML_SINGULARIZE", "off").strip().lower() not in {"off", "false", "0"})

        # 2) De‑dupe early
        dedup_seen = set()
        deduped: List[Tuple[str, str, str, int]] = []
        for s2, r2, d2, si in cleaned:
            if do_sing:
                s2 = singularize_phrase(s2)
                d2 = singularize_phrase(d2)
            key = (s2, r2, d2)
            if key in dedup_seen:
                continue
            dedup_seen.add(key)
            deduped.append((s2, r2, d2, si))

        # Optional GraphJudge filtering (modular) with gray-zone logging
        gj = GraphJudge.from_env()
        if gj.enabled():
            judged: List[Tuple[str, str, str, int]] = []
            # Gray-zone logging controls
            try:
                import json as _json
                from pathlib import Path as _Path
            except Exception:
                _json = None  # type: ignore
                _Path = None  # type: ignore

            gray_band_env = os.getenv("YAML_GRAPH_JUDGE_GRAY_BAND", "0.10").strip()
            try:
                gray_band = float(gray_band_env or 0.10)
            except Exception:
                gray_band = 0.10
            log_path_env = os.getenv("YAML_GRAPH_JUDGE_GRAYZONE_LOG", "data/judge_grayzone.jsonl").strip()

            # Compute base dir for relative logs (server/)
            base_dir = None
            try:
                from pathlib import Path as _Path2
                base_dir = _Path2(__file__).resolve().parents[3]
            except Exception:
                base_dir = None

            log_path = None
            if _Path is not None:
                try:
                    p = _Path(log_path_env)
                    if not p.is_absolute():
                        if base_dir is not None:
                            p = base_dir / p
                    # ensure parent dir exists
                    p.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
                    log_path = p
                except Exception:
                    log_path = None

            thr = gj.cfg.threshold
            SAFE_RELS = {"be", "is", "lives_in", "works_at", "founded"}
            for s2, r2, d2, si in deduped:
                if r2 in SAFE_RELS:
                    judged.append((s2, r2, d2, si))
                    sc = thr  # treat as accepted for gray-zone logging
                else:
                    sc = gj.score(s2, r2, d2, doc)
                    if sc >= thr:
                        judged.append((s2, r2, d2, si))
                # Log gray-zone (|score - thr| <= band)
                try:
                    if log_path is not None and _json is not None and abs(sc - thr) <= gray_band:
                        rec = {
                            "text": text,
                            "triple": [s2, r2, d2],
                            "score": sc,
                            "threshold": thr,
                        }
                        with open(str(log_path), "a", encoding="utf-8") as f:
                            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            deduped = judged

        # Optional: disable density caps for true upper-bound evaluation
        caps_env = os.getenv("YAML_DENSITY_CAPS", "on").strip().lower()
        caps_off = caps_env in ("off", "false", "0")

        if caps_off:
            # Return judged/deduped facts without per-sentence or global caps
            return [(s2, r2, d2) for (s2, r2, d2, _si) in deduped]

        # Optional tiny scorer to bias retention (env‑gated)
        def _pref_score(tr: Tuple[str, str, str, int]) -> float:
            s3, r3, d3, _ = tr
            base = 0.0
            lexicalized_suffix = ("_on", "_in", "_to", "_from", "_with", "_for")
            if any(r3.endswith(suf) for suf in lexicalized_suffix):
                base += 3.0
            core_rels = {"affect", "increase", "decrease", "result_in", "lead_to", "agree_on", "focus_on", "work_on"}
            if r3 in core_rels:
                base += 4.0
            if d3 and len(d3) <= 24:
                base += 1.0
            if s3 in stop_ent or (d3 and d3 in stop_ent):
                base -= 2.0
            return base

        use_scorer = (os.getenv("YAML_SCORER", "on").strip().lower() not in {"off", "false", "0"})

        # 3) Density control per sentence (default ON)
        MAX_PRED_PER_SENT = 2
        MAX_DISC_PER_SENT = 1
        MAX_EDGES_PER_DOC = 8

        def is_discourse(rel: str) -> bool:
            return any(rel.endswith(suf) for suf in ("_after", "_when", "_then", "_because_of", "_contrast", "_if", "_result"))

        # Simple priority for retention; align loosely with retrieval weights
        PRIORITY: dict = {
            "lives_in": 100,
            "works_at": 95,
            "founded": 90,
            "is": 70,
            "be": 70,
        }

        def score(tr: Tuple[str, str, str, int]) -> float:
            _, r3, d3, _ = tr
            base = PRIORITY.get(r3, 50.0)
            # Prefer concrete objects over empty
            if d3:
                base += 5.0
            # Penalize discourse a bit so we keep predicates when capped
            if is_discourse(r3):
                base -= 10.0
            if use_scorer:
                base += _pref_score(tr)
            return base

        # Group by sentence
        by_sent: dict[int, List[Tuple[str, str, str, int]]] = {}
        for tr in deduped:
            by_sent.setdefault(tr[3], []).append(tr)

        kept: List[Tuple[str, str, str]] = []
        for si, items in by_sent.items():
            preds = [t for t in items if not is_discourse(t[1])]  # type: ignore[index]
            discs = [t for t in items if is_discourse(t[1])]      # type: ignore[index]
            preds.sort(key=score, reverse=True)
            discs.sort(key=score, reverse=True)
            for t in preds[:MAX_PRED_PER_SENT]:
                kept.append((t[0], t[1], t[2]))  # type: ignore[index]
            for t in discs[:MAX_DISC_PER_SENT]:
                kept.append((t[0], t[1], t[2]))  # type: ignore[index]

        # 4) Global cap (stable order by score)
        kept_sorted = sorted({k for k in kept}, key=lambda x: PRIORITY.get(x[1], 50.0), reverse=True)
        if len(kept_sorted) > MAX_EDGES_PER_DOC:
            kept_sorted = kept_sorted[:MAX_EDGES_PER_DOC]
        return kept_sorted

    def refine_entities(self, text: str, entities: List[str]) -> List[str]:
        out = []
        seen = set()
        for e in entities:
            ce = _canon_entity_text(e)
            if ce and ce not in seen:
                out.append(ce)
                seen.add(ce)
        return out
