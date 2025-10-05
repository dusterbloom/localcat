"""
Dev-only YAML runtime for text→graph extraction.

Purpose
-------
- Load an authoring YAML (e.g., ASI1_proposal.yaml) and enable a subset of
  Level‑1 patterns plus micro slices of Level‑2/3 for evaluation.
- Zero impact on hot path: this is not imported by default; tests/CLI use it.

Design
------
- We parse the YAML only to detect which high‑level rules are present
  and to keep pattern names/descriptions available for reporting.
- Matching is implemented with compact, deterministic handlers keyed by
  known pattern names (e.g., UNIVERSAL_SVO_ACTIVE, UNIVERSAL_COPULA_NOMINAL).
- Micro‑L2: small pronominal/definite‑NP coref within a 3‑sentence window.
- Micro‑L3: language‑hotspot stubs that activate only if lang matches and
  a relevant rule exists; they fail safe if the language model isn’t loaded.

Note: This module is intentionally conservative; the goal is to prove
superiority in tests before moving to a compiled registry for the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from loguru import logger

import spacy
from spacy.tokens import Doc, Token


# ----------------------------- Data structures -----------------------------

@dataclass
class EnabledPatterns:
    l1: Set[str]
    l2: Set[str]
    l3_lang: Dict[str, Set[str]]


def _safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _canon_entity_text(text: str) -> str:
    t = _safe_lower(text)
    # strip leading determiners and possessives; basic, fast
    for det in ("the", "a", "an", "my", "your", "his", "her", "our", "their", "its"):
        if t.startswith(det + " "):
            t = t[len(det) + 1 :]
            break
    if t.endswith("'s"):
        t = t[:-2]
    # pronoun normalization (user→you)
    if t in {"i", "me", "my", "mine", "myself"}:
        return "you"
    return t


class YAMLRuntime:
    """Minimal interpreter for a subset of the ASI1 YAML spec.

    Supports:
    - L1: UNIVERSAL_SVO_ACTIVE, UNIVERSAL_COPULA_NOMINAL (initial set)
    - Micro‑L2: PRONOMINAL_3SG_RESOLUTION, DEFINITE_NP_COREFERENCE (bounded)
    - Micro‑L3 (stubs): ES_GUSTAR_PSYCH_VERBS, DE_SEPARABLE_PREFIX_VERBS, FR_CLITIC_PRONOUN_CLIMBING, ZH_SERIAL_VERB_CONSTRUCTIONS
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.spec: Dict[str, Any] = {}
        self.enabled = EnabledPatterns(l1=set(), l2=set(), l3_lang={})
        self._load_yaml()

    def _load_yaml(self) -> None:
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                self.spec = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load YAML spec '{self.yaml_path}': {e}")
            self.spec = {}

        # Detect enabled L1 patterns
        for item in self.spec.get("core_patterns", []) or []:
            name = item.get("name")
            if name:
                self.enabled.l1.add(str(name))

        # Detect enabled L2 patterns
        for item in self.spec.get("coreference_system", []) or []:
            name = item.get("name")
            if name:
                self.enabled.l2.add(str(name))

        # Detect L3 per language
        for lang, items in (self.spec.get("language_extensions", {}) or {}).items():
            rules: Set[str] = set()
            for it in items or []:
                nm = it.get("name")
                if nm:
                    rules.add(str(nm))
            if rules:
                self.enabled.l3_lang[lang] = rules

        logger.info(
            "YAMLRuntime enabled: L1=%d, L2=%d, L3_lang=%s",
            len(self.enabled.l1),
            len(self.enabled.l2),
            list(self.enabled.l3_lang.keys()),
        )

    # ----------------------------- Extraction API -----------------------------

    def extract(self, text: str, lang: str = "en") -> Tuple[List[str], List[Tuple[str, str, str]], int, Optional[Doc]]:
        """Return (entities, triples, neg_count, doc)."""
        nlp = self._get_nlp(lang)
        if not nlp:
            return [], [], 0, None

        doc = nlp(text)
        entities: Set[str] = set()
        triples: List[Tuple[str, str, str]] = []
        neg_count = 0

        # L1 matchers (subset)
        if "UNIVERSAL_SVO_ACTIVE" in self.enabled.l1:
            self._match_svo_active(doc, entities, triples)
        if "UNIVERSAL_COPULA_NOMINAL" in self.enabled.l1:
            self._match_copula_nominal(doc, entities, triples)

        # Count negations quickly
        for tok in doc:
            if tok.dep_ == "neg":
                neg_count += 1

        # Micro‑L2 coref pass (bounded)
        if {
            "PRONOMINAL_3SG_RESOLUTION",
            "DEFINITE_NP_COREFERENCE",
        } & self.enabled.l2:
            self._micro_coref_rewrite(doc, triples)

        # Micro‑L3 stubs (lang‑gated)
        rules = self.enabled.l3_lang.get(lang, set())
        if rules:
            self._micro_l3(doc, triples, lang, rules)

        # Collect entities from triples
        for s, _, d in triples:
            entities.add(_canon_entity_text(s))
            entities.add(_canon_entity_text(d))

        return list(entities), triples, neg_count, doc

    # ----------------------------- L1 handlers -----------------------------

    def _match_svo_active(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]]):
        for tok in doc:
            if tok.pos_ == "VERB":
                subj = None
                obj = None
                for ch in tok.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                    elif ch.dep_ in {"obj", "dobj"}:
                        obj = ch
                if subj and obj:
                    s = _canon_entity_text(subj.text)
                    r = _safe_lower(tok.lemma_)
                    d = _canon_entity_text(obj.text)
                    triples.append((s, r, d))

    def _match_copula_nominal(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]]):
        # Look for attr/acomp attached to cop or AUX
        for tok in doc:
            if tok.dep_ in {"attr", "acomp"}:
                head = tok.head
                subj = None
                for ch in head.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                        break
                if subj:
                    s = _canon_entity_text(subj.text)
                    d = _canon_entity_text(tok.text)
                    triples.append((s, "is", d))

    # ----------------------------- Micro‑L2 coref -----------------------------

    def _micro_coref_rewrite(self, doc: Doc, triples: List[Tuple[str, str, str]]):
        """Resolve simple 3rd‑person pronouns and definite NPs within last 3 sentences.

        Strategy: build a small antecedent list (proper nouns / noun chunks) in a
        rolling window; replace pronoun subjects/objects in triples when confident.
        """
        # Build sentence starts for windowing
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
        antecedents: List[str] = []
        for sent in sents[-3:]:
            # Named entities and strong noun chunks first
            for ent in getattr(sent, "ents", []) or []:
                antecedents.append(_canon_entity_text(ent.text))
            for chunk in getattr(sent, "noun_chunks", []) or []:
                antecedents.append(_canon_entity_text(chunk.text))

        if not antecedents:
            return

        pronouns = {"he", "she", "him", "her", "it", "they", "them", "his", "hers", "its", "their"}

        def resolve(token_text: str) -> Optional[str]:
            t = _safe_lower(token_text)
            if t in pronouns and antecedents:
                # choose the most recent antecedent
                return antecedents[-1]
            # definite NP coref ("the X") → prefer recent X
            if t.startswith("the "):
                base = t[4:]
                for cand in reversed(antecedents):
                    if base in cand or cand in base:
                        return cand
            return None

        # Rewrite triples inplace when confident
        for i, (s, r, d) in enumerate(triples):
            ns = resolve(s) or s
            nd = resolve(d) or d
            triples[i] = (ns, r, nd)

    # ----------------------------- Micro‑L3 stubs -----------------------------

    def _micro_l3(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str, rules: Set[str]):
        if lang == "es" and "ES_GUSTAR_PSYCH_VERBS" in rules:
            for tok in doc:
                if tok.pos_ == "VERB" and _safe_lower(tok.lemma_) in {"gustar", "encantar", "interesar"}:
                    experiencer = None
                    theme = None
                    for ch in tok.children:
                        if ch.dep_ in {"iobj", "obl"}:
                            experiencer = ch
                        elif ch.dep_ in {"nsubj", "obj"}:
                            theme = ch
                    if experiencer and theme:
                        s = _canon_entity_text(experiencer.text)
                        d = _canon_entity_text(theme.text)
                        triples.append((s, "like", d))

        if lang == "de" and "DE_SEPARABLE_PREFIX_VERBS" in rules:
            # Basic reconstruction via compound:prt/obl:prt
            for tok in doc:
                if tok.pos_ == "VERB":
                    prefix = None
                    subj = None
                    obj = None
                    for ch in tok.children:
                        if ch.dep_ in {"compound:prt", "obl:prt", "prt"}:
                            prefix = ch.text.lower()
                        elif ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                        elif ch.dep_ in {"obj", "dobj"}:
                            obj = ch
                    if prefix and subj and obj:
                        full = f"{prefix}_{_safe_lower(tok.lemma_)}"
                        triples.append((_canon_entity_text(subj.text), full, _canon_entity_text(obj.text)))

        if lang == "fr" and "FR_CLITIC_PRONOUN_CLIMBING" in rules:
            # Lightweight: treat clitic + main verb as relation
            for tok in doc:
                if tok.pos_ == "VERB":
                    subj = None
                    obj = None
                    for ch in tok.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                        elif ch.dep_ in {"obj"}:
                            obj = ch
                    if subj and obj:
                        triples.append((_canon_entity_text(subj.text), _safe_lower(tok.lemma_), _canon_entity_text(obj.text)))

        if lang == "zh" and "ZH_SERIAL_VERB_CONSTRUCTIONS" in rules:
            # Minimal: create purpose relation between first two verbs if consecutive
            verbs = [t for t in doc if t.pos_ == "VERB"]
            if len(verbs) >= 2:
                v1, v2 = verbs[0], verbs[1]
                subj = None
                for ch in v1.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                        break
                if subj:
                    triples.append((_canon_entity_text(subj.text), f"{_safe_lower(v1.lemma_)}_in_order_to", _safe_lower(v2.lemma_)))

    # ----------------------------- Utilities -----------------------------

    def _get_nlp(self, lang: str) -> Optional[spacy.Language]:
        try:
            if lang == "en":
                # Disable heavy pipes; minimal for deps
                return spacy.load("en_core_web_sm")
            # Best effort: attempt to load language‑specific small models
            return spacy.load(f"{lang}_core_news_sm")
        except Exception as e:
            logger.warning(f"spaCy model for lang='{lang}' unavailable: {e}")
            return None
