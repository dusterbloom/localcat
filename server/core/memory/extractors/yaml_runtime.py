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
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from loguru import logger
import re

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
    # strip leading determiners and possessives; multi-lingual set
    for det in (
        "the", "a", "an", "my", "your", "his", "her", "our", "their", "its",
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "le", "la", "les", "un", "une", "des", "l'", "l’",
        "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "eines",
        "il", "lo", "la", "i", "gli", "le", "uno", "una", "un",
    ):
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
        raw_text: Optional[str] = None
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                self.spec = yaml.safe_load(raw_text) or {}
        except Exception as e:
            logger.info(
                "Non-standard YAML for '%s' (%s). Falling back to tolerant scan.",
                self.yaml_path,
                str(e).splitlines()[0],
            )
            self.spec = {}

        def apply_spec(spec: Dict[str, Any]) -> None:
            # Merge patterns from a parsed spec
            for item in spec.get("core_patterns", []) or []:
                name = item.get("name") or item.get("rule_id")
                if name:
                    self.enabled.l1.add(str(name))

            for item in spec.get("coreference_system", []) or []:
                name = item.get("name") or item.get("rule_id")
                if name:
                    self.enabled.l2.add(str(name))

            lang_map = {
                "english": "en",
                "spanish": "es",
                "german": "de",
                "french": "fr",
                "chinese": "zh",
            }
            for lang, items in (spec.get("language_extensions", {}) or {}).items():
                norm_lang = lang_map.get(_safe_lower(lang), lang)
                rules: Set[str] = set(self.enabled.l3_lang.get(lang, set()))
                for it in items or []:
                    nm = None
                    if isinstance(it, dict):
                        nm = it.get("name") or it.get("rule_id")
                    elif isinstance(it, str):
                        nm = it
                    if nm:
                        rules.add(str(nm))
                if rules:
                    self.enabled.l3_lang[norm_lang] = rules

        if self.spec:
            # Strict path using parsed YAML
            apply_spec(self.spec)

            # Optional includes: list of relative file paths to merge
            includes = self.spec.get("includes") or []
            if isinstance(includes, list):
                base_dir = os.path.dirname(self.yaml_path)
                for inc in includes:
                    if not isinstance(inc, str):
                        continue
                    inc_path = os.path.join(base_dir, inc)
                    try:
                        with open(inc_path, "r", encoding="utf-8") as f:
                            inc_text = f.read()
                        inc_spec = yaml.safe_load(inc_text) or {}
                        if isinstance(inc_spec, dict):
                            apply_spec(inc_spec)
                    except Exception as ie:
                        logger.warning(f"Failed to load include '{inc_path}': {ie}")
        elif raw_text:
            # Tolerant scanner when YAML syntax is non‑standard
            section = None  # 'core_patterns', 'coreference_system', 'language_extensions'
            current_lang = None
            for line in raw_text.splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if s.startswith('core_patterns:'):
                    section = 'core_patterns'
                    current_lang = None
                    continue
                if s.startswith('coreference_system:'):
                    section = 'coreference_system'
                    current_lang = None
                    continue
                if s.startswith('language_extensions:'):
                    section = 'language_extensions'
                    current_lang = None
                    continue
                # Language headers under language_extensions
                if section == 'language_extensions' and s.endswith(':') and not s.startswith('-'):
                    # e.g., 'spanish:' or 'german:'
                    current_lang = s[:-1].strip()
                    if current_lang:
                        self.enabled.l3_lang.setdefault(current_lang, set())
                    continue
                # Capture names: lines like "- name: \"XYZ\""
                if 'name:' in s:
                    try:
                        # Extract text after name:
                        idx = s.index('name:') + 5
                        val = s[idx:].strip().strip('"\'')
                    except Exception:
                        val = None
                    if val:
                        if section == 'core_patterns':
                            self.enabled.l1.add(val)
                        elif section == 'coreference_system':
                            self.enabled.l2.add(val)
                        elif section == 'language_extensions' and current_lang:
                            self.enabled.l3_lang.setdefault(current_lang, set()).add(val)

        # Fallback defaults if YAML couldn't be parsed or yielded no patterns
        if not self.enabled.l1:
            self.enabled.l1.update({
                "UNIVERSAL_SVO_ACTIVE",
                "UNIVERSAL_COPULA_NOMINAL",
            })

        logger.info(
            f"YAMLRuntime enabled -> L1:{sorted(list(self.enabled.l1))} "
            f"L2:{sorted(list(self.enabled.l2))} L3_lang:{sorted(list(self.enabled.l3_lang.keys()))}"
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

        # L1 matchers (call only if handler is available)
        if "UNIVERSAL_SVO_ACTIVE" in self.enabled.l1 and hasattr(self, "_match_svo_active"):
            self._match_svo_active(doc, entities, triples, lang)
        if "UNIVERSAL_SVO_PASSIVE" in self.enabled.l1 and hasattr(self, "_match_svo_passive"):
            self._match_svo_passive(doc, entities, triples, lang)
        if "UNIVERSAL_COPULA_NOMINAL" in self.enabled.l1 and hasattr(self, "_match_copula_nominal"):
            self._match_copula_nominal(doc, entities, triples)
        if "UNIVERSAL_COPULA_ADJECTIVAL" in self.enabled.l1 and hasattr(self, "_match_copula_adjectival"):
            self._match_copula_adjectival(doc, entities, triples)
        if "UNIVERSAL_COORD_SUBJECT" in self.enabled.l1 and hasattr(self, "_match_coord_subject"):
            self._match_coord_subject(doc, entities, triples, lang)
        if "UNIVERSAL_COORD_OBJECT" in self.enabled.l1 and hasattr(self, "_match_coord_object"):
            self._match_coord_object(doc, entities, triples, lang)
        if "UNIVERSAL_COORD_VERB" in self.enabled.l1 and hasattr(self, "_match_coord_verb"):
            self._match_coord_verb(doc, entities, triples, lang)
        if "UNIVERSAL_COORD_MIXED" in self.enabled.l1 and hasattr(self, "_match_coord_mixed"):
            self._match_coord_mixed(doc, entities, triples, lang)
        if "UNIVERSAL_DITRANSITIVE_GIVE" in self.enabled.l1 and hasattr(self, "_match_ditransitive_give"):
            self._match_ditransitive_give(doc, entities, triples, lang)
        if "UNIVERSAL_DITRANSITIVE_COMMUNICATE" in self.enabled.l1 and hasattr(self, "_match_ditransitive_communicate"):
            self._match_ditransitive_communicate(doc, entities, triples, lang)
        if "UNIVERSAL_CONTROL_VERB" in self.enabled.l1 and hasattr(self, "_match_control_verb"):
            self._match_control_verb(doc, entities, triples, lang)
        if "UNIVERSAL_CCOMP_EMBEDDING" in self.enabled.l1 and hasattr(self, "_match_ccomp_embedding"):
            self._match_ccomp_embedding(doc, entities, triples, lang)
        if "UNIVERSAL_MODAL_VERBS" in self.enabled.l1 and hasattr(self, "_match_modal_verbs"):
            self._match_modal_verbs(doc, entities, triples, lang)
        if "UNIVERSAL_RELATIVE_CLAUSE" in self.enabled.l1 and hasattr(self, "_match_relative_clauses"):
            self._match_relative_clauses(doc, triples, lang)
        if "UNIVERSAL_TEMPORAL_ADVERBIALS" in self.enabled.l1 and hasattr(self, "_match_temporal_adverbials"):
            self._match_temporal_adverbials(doc, entities, triples, lang)
        if "UNIVERSAL_SPATIAL_PREPOSITIONS" in self.enabled.l1 and hasattr(self, "_match_spatial_prepositions"):
            self._match_spatial_prepositions(doc, entities, triples, lang)
        if "UNIVERSAL_QUANTIFIER_SCOPE" in self.enabled.l1 and hasattr(self, "_match_quantifier_scope"):
            self._match_quantifier_scope(doc, entities, triples, lang)
        if "UNIVERSAL_NEGATION_SCOPE" in self.enabled.l1 and hasattr(self, "_match_negation_scope"):
            neg_count += self._match_negation_scope(doc, entities, triples, lang)
        if "UNIVERSAL_PROGRESSIVE_ASPECT" in self.enabled.l1 and hasattr(self, "_match_progressive_aspect"):
            self._match_progressive_aspect(doc, entities, triples, lang)
        if "UNIVERSAL_PERFECT_ASPECT" in self.enabled.l1 and hasattr(self, "_match_perfect_aspect"):
            self._match_perfect_aspect(doc, entities, triples, lang)

        # Fallbacks to improve coverage in edge parses
        self._fallback_copula_adjectival(doc, triples)
        self._ensure_intransitive_subjects(doc, triples)
        # Regex-based safety nets
        # Copula adjectival can add missing 'N be ADJ' even if other triples exist
        self._regex_copula_adjectival(text, triples)
        # Intransitive regex only if nothing extracted
        if not triples:
            self._regex_intransitive(text, triples)
        # Sanitize degenerate copula self-pairs like ('car','is','car')
        self._sanitize_copula_pairs(triples)
        # Appositive-as-edge (env-gated, conservative)
        self._emit_appos_edges(doc, triples, lang)

        # Count negations quickly
        for tok in doc:
            if tok.dep_ == "neg":
                neg_count += 1

        # Micro‑L2 coref pass (bounded)
        if {
            "PRONOMINAL_3SG_RESOLUTION",
            "DEFINITE_NP_COREFERENCE",
        } & self.enabled.l2:
            coref_env = os.getenv("YAML_COREF", "on").strip().lower()
            if coref_env not in {"off", "false", "0"}:
                # Appositive aliasing before pronoun rewrite to stabilize mentions
                self._micro_appositive_alias(doc, triples)
                self._micro_coref_rewrite(doc, triples, lang)

        # Micro‑L2 discourse/temporal relations
        if {
            "DISCOURSE_CONNECTIVE_RESOLUTION",
            "TEMPORAL_EVENT_CHAINING",
        } & self.enabled.l2:
            self._micro_discourse_temporal(doc, triples, lang)

        # Micro‑L2 entity clustering/aliasing
        if {"ENTITY_CLUSTER_MERGING"} & self.enabled.l2:
            self._micro_cluster_entities(triples)

        # Micro‑L3 stubs (lang‑gated)
        rules = self.enabled.l3_lang.get(lang, set())
        if rules:
            self._micro_l3(doc, triples, lang, rules)
            # Apply a second coref rewrite to resolve pronouns introduced by L3
            if {"PRONOMINAL_3SG_RESOLUTION", "DEFINITE_NP_COREFERENCE"} & self.enabled.l2:
                self._micro_coref_rewrite(doc, triples, lang)

        # Optional: Light-verb rewriting (conservative; env-gated)
        self._rewrite_light_verbs(triples, lang)
        # Optional: Nominalization rewriting (conservative; env-gated)
        self._rewrite_nominals(doc, triples, lang)

        # Collect entities from triples
        for s, _, d in triples:
            entities.add(_canon_entity_text(s))
            entities.add(_canon_entity_text(d))

        return list(entities), triples, neg_count, doc

    # ----------------------------- L1 handlers -----------------------------

    def _match_svo_passive(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str) -> None:
        return

    def _match_svo_active(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        
        
        def _enrich_object_phrase(tok: Token) -> str:
            # Return only attached prepositional phrases (no base noun), e.g., 'of climate change on ecosystems'
            parts: List[str] = []
            for ch in tok.children:
                if ch.dep_ == "prep":
                    prep = ch.text.lower()
                    pobj = None
                    for gc in ch.children:
                        if gc.dep_ in {"pobj", "obj"}:
                            pobj = gc
                            break
                    if pobj is not None:
                        # Skip gerund/verb pobj (adjunct-like: 'for expanding', 'by doing')
                        try:
                            if getattr(pobj, "pos_", "") == "VERB":
                                continue
                        except Exception:
                            pass
                        det = None
                        for gc in pobj.children:
                            if gc.dep_ in {"det", "det:poss"}:
                                det = gc.text.lower()
                                break
                        target = pobj.text.lower()
                        parts.append(f"{prep} {det + ' ' if det else ''}{target}".strip())
            return " ".join(parts)

        def _expand_noun_with_modifiers(n: Token) -> str:
            # Include left-side compounds and adjectives; ignore determiners
            # Prefer spaCy noun_chunks when available
            try:
                for chunk in getattr(n.doc, "noun_chunks", []) or []:
                    if getattr(chunk, "root", None) is n:
                        phrase = chunk.text.lower()
                        return _canon_entity_text(phrase)
            except Exception:
                pass
            collected = [n]
            try:
                for lc in n.lefts:
                    if lc.dep_ in {"compound", "amod", "nummod"}:
                        if lc.dep_ == "amod":
                            stop_amod = {
                                "various", "several", "numerous", "multiple", "different",
                                "technological", "innovative", "traditional", "digital",
                                "comprehensive", "extensive", "new", "record"
                            }
                            if _safe_lower(lc.text) in stop_amod:
                                continue
                        collected.append(lc)
            except Exception:
                pass
            toks = sorted({t.i: t for t in collected}.values(), key=lambda t: t.i)
            phrase = " ".join(t.text.lower() for t in toks)
            return _canon_entity_text(phrase)

        def _xcomp_phrase(xc: Token) -> str:
            # Build "verb obj" phrase from xcomp verb and its object if present
            head = _safe_lower(getattr(xc, "lemma_", getattr(xc, "text", "")))
            dobj = None
            for ch in xc.children:
                if ch.dep_ in {"obj", "dobj"}:
                    dobj = ch
                    break
            if dobj is not None:
                obj_text = self._object_text_from(xc, dobj)
                return f"{head} {obj_text}".strip()
            return head

        def _object_text(v: Token, tok: Token) -> str:
            """Return canonical object text.
            - If `tok` is the pobj of a preposition attached to verb `v`, include the preposition: 'in garden'.
            - Else, enrich from object-internal PPs and include local noun modifiers.
            """
            prep_policy: Dict[str, Dict[str, str]] = {
                # policy: 'keep' → keep prep in object; 'drop' → drop prep from object
                "expand": {"into": "keep"},
                "move": {"to": "keep", "from": "keep"},
                "arrive": {"at": "keep", "in": "keep"},
                "return": {"to": "keep", "from": "keep"},
                "live": {"in": "keep", "at": "keep"},
                "invest": {"in": "drop"},
                "apply": {"for": "drop", "to": "drop"},
                "provide": {"with": "drop", "for": "drop"},
                "approve": {"of": "drop"},
                "object": {"to": "drop"},
                "belong": {"to": "drop"},
                "base": {"on": "keep"},
                "consist": {"of": "drop", "in": "drop"},
                "search": {"for": "drop"},
                "benefit": {"from": "drop", "to": "drop"},
                "enter": {"into": "drop"},
                "comply": {"with": "drop"},
                "adhere": {"to": "drop"},
                "engage": {"with": "drop", "in": "drop"},
            }
            try:
                if tok.head is not None and tok.head.dep_ == "prep" and tok.head.head.i == v.i:
                    np = _expand_noun_with_modifiers(tok)
                    vlem = _safe_lower(getattr(v, "lemma_", getattr(v, "text", "")))
                    prep = tok.head.text.lower()
                    pol = prep_policy.get(vlem, {}).get(prep)
                    if pol == "drop":
                        return np
                    # default keep unless policy says drop
                    return f"{prep} {np}".strip()
            except Exception:
                pass
            # If tok is an open complement, return its verb+object phrase
            if tok.dep_ == "xcomp" and tok.pos_ == "VERB":
                return _xcomp_phrase(tok)
            # for bare objects, include modifiers and internal prepositional attachments
            base_np = _expand_noun_with_modifiers(tok)
            extra_pp = _enrich_object_phrase(tok)
            if extra_pp and base_np:
                # _enrich_object_phrase returns 'base + pps' already when parts exist; avoid double base
                if extra_pp.startswith(base_np + " "):
                    return extra_pp
                return f"{base_np} {extra_pp}".strip()
            return base_np or _enrich_object_phrase(tok)

        def _enrich_from_verb_preps(v: Token, obj_tok: Optional[Token]) -> str:
            parts: List[str] = []
            # Conservative, verb-aware filter for argument-like prepositions
            verb_lemma = _safe_lower(getattr(v, "lemma_", getattr(v, "text", "")))
            pp_scorer_on = os.getenv("YAML_PP_SCORER", "off").strip().lower() not in {"off", "false", "0"}

            def _pp_keep_score(verb: str, prep: str, pobj_text: str, pobj_pos: str) -> float:
                # Tiny linear scorer; positive means keep
                score = -0.05
                prep_w = {
                    "on": 0.35, "in": 0.30, "with": 0.25, "to": 0.25,
                    "for": 0.15, "at": 0.10, "about": 0.05, "into": 0.15, "from": 0.10,
                }.get(prep, 0.0)
                score += prep_w
                if verb in {"work", "focus", "agree", "result", "stem", "lead", "apply", "benefit",
                            "comply", "adhere", "engage", "enter", "provide", "approve", "belong",
                            "base", "consist", "invest", "search", "depend", "rely", "present",
                            "talk", "speak", "listen", "compare", "refer", "discuss", "deal"}:
                    score += 0.25
                toks = (pobj_text or "").split()
                if len(toks) <= 3:
                    score += 0.15
                elif len(toks) >= 7:
                    score -= 0.10
                if pobj_pos in {"NOUN", "PROPN"}:
                    score += 0.10
                elif pobj_pos == "PRON":
                    score -= 0.15
                low = (pobj_text or "").lower()
                if low in {"it", "this", "that", "there"}:
                    score -= 0.25
                return score

            verb_allowed: Dict[str, Set[str]] = {
                # project/work domains
                "work": {"on", "with"},
                "build": {"on", "with"},
                "collaborate": {"with"},
                "contribute": {"to"},
                # dependency/focus
                "depend": {"on"},
                "rely": {"on", "upon"},
                "focus": {"on"},
                "insist": {"on"},
                "specialize": {"in"},
                "succeed": {"in"},
                # perception/communication
                "look": {"at", "into", "for"},
                "listen": {"to"},
                "talk": {"to", "with", "about"},
                "speak": {"to", "with", "about"},
                "refer": {"to"},
                "compare": {"to", "with"},
                "report": {"on"},
                "present": {"to", "on"},
                "explain": {"to"},
                "discuss": {"with"},
                "deal": {"with"},
                "assist": {"with"},
                "help": {"with"},
                "provide": {"with", "for"},
                "approve": {"of"},
                "object": {"to"},
                "belong": {"to"},
                "base": {"on"},
                "consist": {"of", "in"},
                "search": {"for"},
                # participation/intention
                "apply": {"for", "to"},
                "participate": {"in"},
                "plan": {"for"},
                "adhere": {"to"},
                "comply": {"with"},
                "engage": {"in", "with"},
                "enter": {"into"},
                # belief/finance
                "believe": {"in"},
                "invest": {"in"},
                "benefit": {"from"},
                "result": {"in", "from"},
                "stem": {"from"},
                "lead": {"to"},
                # temporal/waiting/agreements
                "wait": {"for"},
                "agree": {"to", "with", "on"},
                # movement/location
                "move": {"to", "from"},
                "travel": {"to"},
                "live": {"in", "at"},
                "relocate": {"to"},
                "arrive": {"at", "in"},
                "return": {"to", "from"},
            }
            generic_allowed = {"to", "for", "on", "in", "into", "from", "about", "at", "with"}
            for ch in v.children:
                if ch.dep_ == "prep":
                    prep = ch.text.lower()
                    allowed_preps = verb_allowed.get(verb_lemma, generic_allowed)
                    if prep not in allowed_preps:
                        continue
                    pobj = None
                    for gc in ch.children:
                        if gc.dep_ in {"pobj", "obj"}:
                            pobj = gc
                            break
                    if pobj is not None:
                        if obj_tok is not None and pobj.i == obj_tok.i:
                            continue
                        # Prefer deeper NP when pobj is a gerund/verb (e.g., 'focus on discussing approaches' -> 'on approaches')
                        pobj_text = ""
                        try:
                            if getattr(pobj, "pos_", "") == "VERB":
                                inner_obj = None
                                for c2 in pobj.children:
                                    if c2.dep_ in {"obj", "dobj"}:
                                        inner_obj = c2
                                        break
                                if inner_obj is not None:
                                    pobj_text = _expand_noun_with_modifiers(inner_obj)
                            if not pobj_text:
                                pobj_text = _expand_noun_with_modifiers(pobj)
                        except Exception:
                            pobj_text = _expand_noun_with_modifiers(pobj)
                        if pp_scorer_on:
                            try:
                                pobj_pos = getattr(pobj, "pos_", "")
                            except Exception:
                                pobj_pos = ""
                            if _pp_keep_score(verb_lemma, prep, pobj_text, pobj_pos) < 0.10:
                                continue
                        parts.append(f"{prep} {pobj_text}")
            return " ".join(parts)
        # Helper: previous sentence subject heuristic (for zero anaphora)
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
        sent_spans = [(s.start, s.end) for s in sents]

        def sent_index(tok: Token) -> int:
            i = tok.i
            for idx, (a, b) in enumerate(sent_spans):
                if a <= i < b:
                    return idx
            return 0

        def find_prev_subject(idx: int) -> Optional[str]:
            if idx <= 0:
                return None
            prev = sents[idx - 1]
            # prefer explicit nsubj/csubj; else any strong noun chunk
            for t in prev:
                if t.dep_ in {"nsubj", "csubj"}:
                    return _canon_entity_text(t.text)
            try:
                for chunk in getattr(prev, "noun_chunks", []) or []:
                    return _canon_entity_text(chunk.text)
            except Exception:
                # Some languages (e.g., zh) don't implement noun_chunks
                pass
            for ent in getattr(prev, "ents", []) or []:
                return _canon_entity_text(ent.text)
            return None

        pro_drop_langs = {"es", "it", "pt", "fr", "zh", "ja", "ko", "ar", "tr"}

        subject_deps = {"nsubj", "csubj", "sb"}
        object_deps = {"obj", "dobj", "oa"}
        passive_subj_dep = {"nsubjpass"}
        if lang == "en":
            object_deps.add("xcomp")

        def _collect_conj(head_tok: Optional[Token], pos_ok: Set[str]) -> List[Token]:
            if head_tok is None:
                return []
            stack = [head_tok]
            seen: Set[int] = set()
            out: List[Token] = []
            while stack:
                t = stack.pop()
                if t.i in seen:
                    continue
                seen.add(t.i)
                if t.pos_ in pos_ok:
                    out.append(t)
                for ch in t.children:
                    if ch.dep_ == "conj":
                        stack.append(ch)
                # also traverse to the coordination head
                try:
                    if t.dep_ == "conj" and t.head is not None:
                        stack.append(t.head)
                except Exception:
                    pass
            # uniqueness by index, preserve doc order
            out = sorted({tok.i: tok for tok in out}.values(), key=lambda x: x.i)
            return out

        def _inherit_from_head_verb(v: Token) -> Tuple[Optional[Token], Optional[Token]]:
            """For coordinated verbs, inherit subj/obj from head verb if missing.

            Returns (subj_tok, obj_tok) found on ancestor verb, else (None, None).
            """
            head = v.head
            visited: Set[int] = set()
            while head is not None and head.i not in visited:
                visited.add(head.i)
                if head.pos_ in {"VERB", "AUX"}:
                    subj_tok = None
                    obj_tok = None
                    for ch in head.children:
                        if ch.dep_ in subject_deps and subj_tok is None:
                            subj_tok = ch
                        elif ch.dep_ in object_deps and obj_tok is None:
                            obj_tok = ch
                        elif ch.dep_ == "prep" and obj_tok is None:
                            # inherit prepositional object if exists
                            for gc in ch.children:
                                if gc.dep_ in {"pobj", "obj"}:
                                    obj_tok = gc
                                    break
                    if subj_tok is not None or obj_tok is not None:
                        return subj_tok, obj_tok
                # climb further if this is nested conj
                head = head.head if head.dep_ == "conj" else None
            return None, None

        made_for_token: Set[int] = set()
        added_triples: Set[Tuple[str, str, str]] = set()

        def _expand_punct_list(anchor: Token) -> List[Token]:
            # Heuristic expansion for punctuation-separated lists not marked as conj
            try:
                si = sent_index(anchor)
            except Exception:
                si = 0
            sent_start, sent_end = sent_spans[si]
            out: List[Token] = []
            max_take = 3
            taken = 0
            # look right for NOUN/PROPN preceded by comma or cc 'and/or'
            j = anchor.i + 1
            prev_sep = False
            while j < sent_end and taken < max_take:
                tj = doc[j]
                wj = _safe_lower(tj.text)
                if wj in {",", ";"} or tj.dep_ == "punct" or wj in {"and", "or"}:
                    prev_sep = True
                    j += 1
                    continue
                if prev_sep and getattr(tj, "pos_", "") in {"NOUN", "PROPN"}:
                    out.append(tj)
                    taken += 1
                    prev_sep = False
                    j += 1
                    continue
                # stop at a verb/clause boundary
                if getattr(tj, "pos_", "") in {"VERB", "AUX"}:
                    break
                j += 1
            return out
        for tok in doc:
            if tok.pos_ in {"VERB", "AUX"}:
                subj: Optional[Token] = None
                obj: Optional[Token] = None
                prt = None
                is_passive = False
                passive_subj_tok: Optional[Token] = None
                obj_from_prep = False
                obj_prep_text: Optional[str] = None
                aux_lemmas: Set[str] = set()
                for ch in tok.children:
                    if ch.dep_ in subject_deps:
                        subj = ch
                    elif ch.dep_ in object_deps:
                        obj = ch
                    elif ch.dep_ in passive_subj_dep:
                        # track passive subject separately so we can normalize flexibly
                        passive_subj_tok = ch
                        is_passive = True
                    elif ch.dep_ == "auxpass":
                        is_passive = True
                    elif ch.dep_ == "aux":
                        aux_lemmas.add(_safe_lower(getattr(ch, "lemma_", getattr(ch, "text", ""))))
                    elif ch.dep_ == "prt" and lang == "en":
                        prt = ch.text.lower()
                # Fallback passive detection: be/get + VBN
                if not is_passive:
                    try:
                        if ("be" in aux_lemmas or "get" in aux_lemmas) and getattr(tok, "tag_", "").upper() == "VBN":
                            is_passive = True
                    except Exception:
                        pass
                # If no direct object, pick prepositional object from verb
                if obj is None:
                    for ch in tok.children:
                        if ch.dep_ == "prep":
                            for gc in ch.children:
                                if gc.dep_ in {"pobj", "obj"}:
                                    obj = gc
                                    obj_from_prep = True
                                    obj_prep_text = ch.text.lower()
                                    break
                        if obj is not None:
                            break
                # If still missing subj/obj on coordinated verb, inherit from head verb
                if tok.dep_ == "conj" and (subj is None or obj is None):
                    inh_subj, inh_obj = _inherit_from_head_verb(tok)
                    if subj is None and inh_subj is not None:
                        subj = inh_subj
                    if obj is None and inh_obj is not None:
                        obj = inh_obj
                # Passive agent → subject
                if is_passive and subj is None:
                    agent = None
                    for ch in tok.children:
                        if ch.dep_ == "agent" or (ch.dep_ == "prep" and _safe_lower(ch.text) == "by"):
                            agent = ch
                            break
                    if agent is not None:
                        for gc in agent.children:
                            if gc.dep_ in {"pobj", "obj"}:
                                subj = gc
                                break
                # Control/raising: inherit subject for xcomp from governing verb
                if subj is None:
                    try:
                        head = tok.head
                        # ascend one level if current token is an open clausal complement
                        if tok.dep_ == "xcomp" and head is not None and head.pos_ in {"VERB", "AUX"}:
                            control_verb = _safe_lower(getattr(head, "lemma_", getattr(head, "text", "")))
                            # Object-control verbs (ask/tell/order/etc.): subject is the object of the governor
                            obj_control = {"ask", "tell", "order", "teach", "advise", "encourage", "force", "allow", "permit", "require", "urge", "invite"}
                            # Subject-control/raising verbs (want/expect/plan/try/decide/agree/hope/seem/appear/help)
                            subj_control = {"want", "expect", "plan", "try", "decide", "agree", "hope", "seem", "appear", "help"}
                            if control_verb in obj_control:
                                for ch in head.children:
                                    if ch.dep_ in object_deps:
                                        subj = ch
                                        break
                            else:
                                # default to subject-control: inherit subject of the governor
                                for ch in head.children:
                                    if ch.dep_ in subject_deps:
                                        subj = ch
                                        break
                    except Exception:
                        pass

                # Zero-anaphora subject recovery for pro-drop languages
                if subj is None and lang in pro_drop_langs:
                    idx = sent_index(tok)
                    prev_s = find_prev_subject(idx)
                    if prev_s:
                        s = prev_s
                    else:
                        s = None
                else:
                    s = _canon_entity_text(subj.text) if subj is not None else None

                base_rel = _safe_lower(tok.lemma_) or _safe_lower(tok.text)
                # Build relation; attach particle for phrasal verbs only
                rel_with_particle = f"{base_rel}_{prt}" if prt else base_rel
                # Special-case lexicalized verb+prep where relation commonly carries the prep
                if obj_from_prep:
                    if obj_prep_text == "on" and base_rel in {"work", "focus", "agree"}:
                        rel_with_particle = f"{base_rel}_on"
                    elif obj_prep_text == "in" and base_rel in {"result"}:
                        rel_with_particle = f"{base_rel}_in"
                    elif obj_prep_text == "from" and base_rel in {"stem"}:
                        rel_with_particle = f"{base_rel}_from"
                    elif obj_prep_text == "to" and base_rel in {"lead"}:
                        rel_with_particle = f"{base_rel}_to"
                    elif obj_prep_text == "to" and base_rel in {"benefit", "adhere", "apply"}:
                        rel_with_particle = f"{base_rel}_to"
                    elif obj_prep_text == "of" and base_rel in {"consist"}:
                        rel_with_particle = f"{base_rel}_of"
                    elif obj_prep_text == "in" and base_rel in {"consist"}:
                        rel_with_particle = f"{base_rel}_in"
                    elif obj_prep_text == "into" and base_rel in {"enter"}:
                        rel_with_particle = f"{base_rel}_into"
                    elif obj_prep_text == "with" and base_rel in {"comply", "engage"}:
                        rel_with_particle = f"{base_rel}_with"
                    elif obj_prep_text == "in" and base_rel in {"engage"}:
                        rel_with_particle = f"{base_rel}_in"

                if is_passive and subj is None and passive_subj_tok is not None:
                    # No agent: treat passive subject as S, and use verb-attached prepositional
                    # complements as the object (e.g., "implemented across departments").
                    s_text = _canon_entity_text(passive_subj_tok.text)
                    d_text = _enrich_from_verb_preps(tok, None)
                    # Prefer a concise single PP like "across departments"
                    if not d_text:
                        d_text = _canon_entity_text(obj.text) if obj is not None else ""
                    triple = (s_text, base_rel, d_text)
                    if triple not in added_triples:
                        triples.append(triple)
                        added_triples.add(triple)
                    made_for_token.add(tok.i)
                elif s is not None and obj is not None:
                    # Expand ditransitives by collecting all object-like dependents
                    object_like_deps = {"obj", "dobj", "oa", "iobj", "dative"}
                    obj_tokens: List[Token] = [ch for ch in tok.children if ch.dep_ in object_like_deps]
                    if not obj_tokens:
                        obj_tokens = [obj]
                    # Coordination expansion for subj and each object-like token
                    subj_list = _collect_conj(subj, {"NOUN", "PROPN", "PRON"}) or ([subj] if subj is not None else [])
                    expanded_objs: List[Token] = []
                    seen_o = set()
                    for ot in obj_tokens:
                        expanded = _collect_conj(ot, {"NOUN", "PROPN", "PRON"}) or [ot]
                        # Also consider punctuation-separated list items following the anchor
                        try:
                            expanded += _expand_punct_list(ot)
                        except Exception:
                            pass
                        for e in expanded:
                            if e.i not in seen_o:
                                expanded_objs.append(e)
                                seen_o.add(e.i)
                    for sj in subj_list:
                        s_j = _canon_entity_text(sj.text)
                        for oj in expanded_objs:
                            d_j = _object_text(tok, oj)
                            extra = _enrich_from_verb_preps(tok, oj)
                            if extra:
                                d_j = f"{d_j} {extra}".strip()
                            triple = (s_j, rel_with_particle, d_j)
                            if triple not in added_triples:
                                triples.append(triple)
                                added_triples.add(triple)
                    made_for_token.add(tok.i)
                elif s is not None:
                    # Subject-only predicate (no direct object); if xcomp exists, use it as object phrase
                    d_obj = ""
                    try:
                        for xc in tok.children:
                            if xc.dep_ == "xcomp" and getattr(xc, "pos_", "") == "VERB":
                                # build a concise verb-object phrase
                                x_dobj = None
                                for c2 in xc.children:
                                    if c2.dep_ in {"obj", "dobj"}:
                                        x_dobj = c2
                                        break
                                if x_dobj is None:
                                    for c2 in xc.children:
                                        if c2.dep_ == "prep":
                                            for c3 in c2.children:
                                                if c3.dep_ in {"pobj", "obj"}:
                                                    x_dobj = c3
                                                    break
                                        if x_dobj is not None:
                                            break
                                if x_dobj is not None:
                                    d_obj = f"{_safe_lower(getattr(xc, 'lemma_', getattr(xc, 'text', '')))} {self._object_text_from(xc, x_dobj)}".strip()
                                else:
                                    d_obj = _safe_lower(getattr(xc, 'lemma_', getattr(xc, 'text', '')))
                                break
                    except Exception:
                        pass
                    triple = (s, base_rel, d_obj)
                    if triple not in added_triples:
                        triples.append(triple)
                        added_triples.add(triple)
                    made_for_token.add(tok.i)

                # Emit separate xcomp events (e.g., 'decide to postpone' → 'postpone merger')
                if s is not None:
                    for ch in tok.children:
                        if ch.dep_ == "xcomp" and getattr(ch, "pos_", "") == "VERB":
                            # find direct object (or prepositional object) of the xcomp verb
                            x_obj = None
                            for c2 in ch.children:
                                if c2.dep_ in {"obj", "dobj"}:
                                    x_obj = c2
                                    break
                            if x_obj is None:
                                # try prepositional pobj under xcomp verb
                                for c2 in ch.children:
                                    if c2.dep_ == "prep":
                                        for c3 in c2.children:
                                            if c3.dep_ in {"pobj", "obj"}:
                                                x_obj = c3
                                                break
                                    if x_obj is not None:
                                        break
                            d_x = self._object_text_from(ch, x_obj) if x_obj is not None else ""
                            r_x = _safe_lower(getattr(ch, "lemma_", getattr(ch, "text", "")))
                            tri_x = (s, r_x, d_x)
                            if tri_x not in added_triples:
                                triples.append(tri_x)
                                added_triples.add(tri_x)

                # Emit events for gerund prepositional complements where appropriate
                # - focus on VERB+obj → emit VERB obj
                # - by VERB+obj → emit VERB obj (common for means)
                if s is not None:
                    lemma_tok = _safe_lower(getattr(tok, "lemma_", getattr(tok, "text", "")))
                    for pr in tok.children:
                        if pr.dep_ == "prep":
                            prep = _safe_lower(pr.text)
                            pobj = None
                            for c2 in pr.children:
                                if c2.dep_ in {"pobj", "obj"}:
                                    pobj = c2
                                    break
                            if pobj is not None and getattr(pobj, "pos_", "") == "VERB":
                                # choose only certain patterns to avoid noise
                                if (lemma_tok == "focus" and prep == "on") or prep == "by":
                                    # find object of gerund
                                    gobj = None
                                    for c3 in pobj.children:
                                        if c3.dep_ in {"obj", "dobj"}:
                                            gobj = c3
                                            break
                                    if gobj is None:
                                        for c3 in pobj.children:
                                            if c3.dep_ == "prep":
                                                for c4 in c3.children:
                                                    if c4.dep_ in {"pobj", "obj"}:
                                                        gobj = c4
                                                        break
                                            if gobj is not None:
                                                break
                                    d_g = self._object_text_from(pobj, gobj) if gobj is not None else ""
                                    r_g = _safe_lower(getattr(pobj, "lemma_", getattr(pobj, "text", "")))
                                    tri_g = (s, r_g, d_g)
                                    if tri_g not in added_triples:
                                        triples.append(tri_g)
                                        added_triples.add(tri_g)

        # Fallback for languages where small models mislabel deps (e.g., fr)
        if lang in {"fr", "zh", "es", "de", "it"}:
            sents_fb = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
            for s in sents_fb:
                verbs = [t for t in s if t.pos_ == "VERB"]
                if len(verbs) == 1 and verbs[0].i not in made_for_token:
                    v = verbs[0]
                    # choose nearest left subject-like token
                    left = None
                    for t in reversed(list(s)):
                        if t.i >= v.i:
                            continue
                        if t.pos_ in {"NOUN", "PROPN", "PRON"}:
                            left = t
                            break
                    right = None
                    for t in s:
                        if t.i <= v.i:
                            continue
                        if t.pos_ in {"NOUN", "PROPN"}:
                            right = t
                            break
                    if left is not None:
                        s_text = _canon_entity_text(left.text)
                        r = _safe_lower(v.lemma_) or _safe_lower(v.text)
                        d_text = _canon_entity_text(right.text) if right is not None else ""
                        triples.append((s_text, r, d_text))

    def _match_copula_nominal(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]]):
        # Look for attr/acomp attached to cop or AUX/raising verbs; use head lemma as relation (be/seem/etc.)
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
                    r = _safe_lower(getattr(head, "lemma_", getattr(head, "text", "is"))) or "be"
                    if s and d and s != d:
                        triples.append((s, r, d))

    def _match_relative_clauses(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str) -> None:
        """Extract main-edge facts from relative clauses in English.

        Heuristics (EN):
        - Find verbs labeled as relative clause heads (dep 'relcl' or 'acl:relcl').
        - If the relative pronoun is the subject of the relcl, map subject→head noun.
        - If the relative pronoun is the object, map object→head noun.
        - Prefer explicit non-pronominal subjects/objects when present.
        """
        added: Set[Tuple[str, str, str]] = set()
        rel_prons = {"who", "whom", "whose", "that", "which"}

        def is_rel_pron(t: Token) -> bool:
            w = _safe_lower(t.text)
            return (w in rel_prons) or (getattr(t, "tag_", "").startswith("W"))

        for tok in doc:
            dep = getattr(tok, "dep_", "")
            if dep not in {"relcl", "acl:relcl"}:
                continue
            head_noun = tok.head
            if head_noun is None or getattr(head_noun, "pos_", "") not in {"NOUN", "PROPN"}:
                continue
            verb = tok
            base_rel = _safe_lower(getattr(verb, "lemma_", getattr(verb, "text", "")))
            # Gather arguments in relcl
            subj = None
            obj = None
            for ch in verb.children:
                if ch.dep_ in {"nsubj", "csubj"}:
                    subj = ch
                elif ch.dep_ in {"obj", "dobj"}:
                    obj = ch
            # Case 1: subject is relative pronoun → subject = head noun
            if subj is not None and is_rel_pron(subj):
                s = _canon_entity_text(head_noun.text)
                if obj is not None:
                    # expand coordinated objects
                    objs = [obj]
                    try:
                        for c in obj.children:
                            if c.dep_ == "conj" and getattr(c, "pos_", "") in {"NOUN", "PROPN"}:
                                objs.append(c)
                    except Exception:
                        pass
                    for o_tok in objs:
                        d = self._object_text_from(verb, o_tok)
                        triple = (s, base_rel, d)
                        if triple not in added:
                            triples.append(triple)
                            added.add(triple)
                else:
                    # Intransitive relcl; produce subject-only
                    triple = (s, base_rel, "")
                    if triple not in added:
                        triples.append(triple)
                        added.add(triple)
                continue
            # Case 2: object is relative pronoun (direct object) → object = head noun
            if obj is not None and is_rel_pron(obj) and subj is not None:
                s = _canon_entity_text(subj.text)
                d = _canon_entity_text(head_noun.text)
                triple = (s, base_rel, d)
                if triple not in added:
                    triples.append(triple)
                    added.add(triple)
                continue
            # Case 2b: prepositional object is relative pronoun (pobj under prep)
            for ch in verb.children:
                if ch.dep_ == "prep":
                    prep = _safe_lower(ch.text)
                    pobj = None
                    for gc in ch.children:
                        if gc.dep_ in {"pobj", "obj"}:
                            pobj = gc
                            break
                    if pobj is not None and is_rel_pron(pobj) and subj is not None:
                        s = _canon_entity_text(subj.text)
                        d = _canon_entity_text(head_noun.text)
                        rel = f"{base_rel}_{prep}" if prep else base_rel
                        triple = (s, rel, d)
                        if triple not in added:
                            triples.append(triple)
                            added.add(triple)
                        break
            # Case 2c: pied‑piping via case-marked relative pronoun as oblique (e.g., 'with which')
            # Some parses attach PREP as 'case' to the relative pronoun (obl/obj under verb)
            if subj is not None:
                for ch in verb.children:
                    if ch.dep_ in {"obl", "obj"} and is_rel_pron(ch):
                        pp = None
                        try:
                            for cc in ch.children:
                                if getattr(cc, "dep_", "") == "case":
                                    cand = _safe_lower(getattr(cc, "text", ""))
                                    if cand in {"to", "for", "in", "at", "with", "on", "from", "into"}:
                                        pp = cand
                                        break
                        except Exception:
                            pp = None
                        if pp is not None:
                            s = _canon_entity_text(subj.text)
                            d = _canon_entity_text(head_noun.text)
                            rel = f"{base_rel}_{pp}"
                            triple = (s, rel, d)
                            if triple not in added:
                                triples.append(triple)
                                added.add(triple)
                            break
            # Case 4: possessive 'whose' in the relative clause → head has <noun>
            try:
                for ch in verb.subtree:
                    if _safe_lower(getattr(ch, "text", "")) == "whose" and getattr(ch, "dep_", "") == "poss":
                        poss_head = getattr(ch, "head", None)
                        if poss_head is not None and getattr(poss_head, "pos_", "") in {"NOUN", "PROPN"}:
                            triple = (_canon_entity_text(head_noun.text), "have", _canon_entity_text(poss_head.text))
                            if triple not in added:
                                triples.append(triple)
                                added.add(triple)
                        break
            except Exception:
                pass
            # Case 3: No explicit rel pron; fallback: if no subject, use head noun; if no object, use head noun
            if subj is None and obj is not None:
                s = _canon_entity_text(head_noun.text)
                d = self._object_text_from(verb, obj)
                triple = (s, base_rel, d)
                if triple not in added:
                    triples.append(triple)
                    added.add(triple)
            elif obj is None and subj is not None:
                s = _canon_entity_text(subj.text)
                d = _canon_entity_text(head_noun.text)
                triple = (s, base_rel, d)
                if triple not in added:
                    triples.append(triple)
                    added.add(triple)

    # ----------------------------- Micro‑L2 coref -----------------------------

    def _micro_coref_rewrite(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str):
        """Resolve simple 3rd‑person pronouns and definite NPs within last 3 sentences.

        Strategy: build a small antecedent list (proper nouns / noun chunks) in a
        rolling window; replace pronoun subjects/objects in triples when confident.
        """
        # Build sentence starts for windowing
        sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
        # antecedent tuple: (mention_text, is_person, head_lemma, gender, number)
        Ante = Tuple[str, bool, str, Optional[str], Optional[str]]
        antecedents: List[Ante] = []
        for sent in sents[-3:]:
            # Named entities and strong noun chunks first
            for ent in getattr(sent, "ents", []) or []:
                head = getattr(ent, "root", None)
                head_lemma = getattr(head, "lemma_", "") if head is not None else ""
                morph = getattr(head, "morph", None)
                gender = morph.get("Gender")[0] if morph and morph.get("Gender") else None
                number = morph.get("Number")[0] if morph and morph.get("Number") else None
                is_person = ent.label_ in {"PERSON", "PER"}
                antecedents.append((_canon_entity_text(ent.text), is_person, _safe_lower(head_lemma), gender, number))
            try:
                for chunk in getattr(sent, "noun_chunks", []) or []:
                    try:
                        is_pron = getattr(chunk.root, "pos_", "") == "PRON"
                    except Exception:
                        is_pron = False
                    if is_pron:
                        continue
                    # Only accept nominal chunks (avoid malformed chunks in some small models)
                    if getattr(chunk.root, "pos_", "") not in {"NOUN", "PROPN"}:
                        continue
                    head_lemma = _safe_lower(getattr(chunk.root, "lemma_", ""))
                    morph = getattr(chunk.root, "morph", None)
                    gender = morph.get("Gender")[0] if morph and morph.get("Gender") else None
                    number = morph.get("Number")[0] if morph and morph.get("Number") else None
                    antecedents.append((_canon_entity_text(chunk.text), False, head_lemma, gender, number))
            except Exception:
                # Ignore missing noun_chunks iterator
                pass

        if not antecedents:
            return

        pronouns = {"he", "she", "him", "her", "it", "they", "them", "his", "hers", "its", "their", "this", "that"}
        # Extend with a minimal set for es/fr/it
        if lang == "es":
            pronouns.update({"él", "ella", "lo", "la", "le", "ellos", "ellas"})
        if lang == "fr":
            pronouns.update({"il", "elle", "le", "la", "les", "lui", "eux"})
        if lang == "it":
            pronouns.update({"lui", "lei", "lo", "la", "gli", "le", "loro"})
        if lang == "de":
            pronouns.update({"er", "sie", "es", "ihn", "ihr", "ihm", "ihnen"})
        if lang == "zh":
            pronouns.update({"他", "她", "它", "他们", "她们", "它们"})

        # Cataphora lookahead: any upcoming proper noun can serve as antecedent
        future_proper_nouns: List[str] = []
        for t in doc:
            if t.pos_ == "PROPN":
                future_proper_nouns.append(_canon_entity_text(t.text))

        # Track seen subjects from existing triples to bias antecedents
        seen_subjects: Set[str] = set(_canon_entity_text(s) for s, _, _ in triples)

        def sent_of_text(tkn: str) -> Optional[int]:
            # naive: find first sentence containing this substring
            if not sents:
                return 0
            low = tkn.lower()
            for idx, sp in enumerate(sents):
                if low and low in sp.text.lower():
                    return idx
            return None

        def resolve(token_text: str) -> Optional[str]:
            t = _safe_lower(token_text)
            if t in pronouns and antecedents:
                # Prefer person/non-person compatibility, then recency
                prefer_person = t in {"he", "him", "she", "her", "il", "elle", "lui", "él", "ella", "lei"}
                prefer_non_person = t in {"it", "lo", "la", "le", "les"}
                # stricter gating for it/this/that
                strict_pronouns = {"it", "this", "that"}
                if t in strict_pronouns:
                    # must be same or previous sentence and non-person
                    cur_si = sent_of_text(token_text)
                    # fallback to last sentence index
                    if cur_si is None:
                        cur_si = len(sents) - 1
                    filtered: List[Ante] = []
                    for cand, is_person, head_lemma, gender, number in antecedents:
                        # exclude person-like antecedents
                        if is_person:
                            continue
                        # require proximity (same or last sentence)
                        si = sent_of_text(cand)
                        if si is None or si < cur_si - 1:
                            continue
                        filtered.append((cand, is_person, head_lemma, gender, number))
                    if filtered:
                        # temporarily narrow antecedents for this resolution
                        chosen = None
                        best_score = -1e9
                        for idx, (cand, is_person, head_lemma, gender, number) in enumerate(reversed(filtered)):
                            score = -idx * 0.2
                            if _canon_entity_text(cand) in seen_subjects:
                                score += 0.3
                            if score > best_score:
                                best_score = score
                                chosen = cand
                        if chosen:
                            return chosen
                # Scoring with mild weights: recency (-i), type match, head match, number/gender match
                best = None
                best_score = -1e9
                for idx, (cand, is_person, head_lemma, gender, number) in enumerate(reversed(antecedents)):
                    score = 0.0
                    # recency (closer = better)
                    score += -idx * 0.1
                    # prefer mentions we already used as subjects
                    if _canon_entity_text(cand) in seen_subjects:
                        score += 0.5
                    if prefer_person and is_person:
                        score += 0.6
                    if prefer_non_person and not is_person:
                        score += 0.4
                    # If pronoun string appears in candidate (rare), small boost
                    if t and t in cand:
                        score += 0.05
                    # Head match boost if token_text starts with 'the X' and head matches
                    if t.startswith("the "):
                        the_head = t[4:].split(" ")[0]
                        if the_head and the_head == head_lemma:
                            score += 0.3
                    # rough number/gender hints
                    if t in {"they", "them"} and (number == "Plur"):
                        score += 0.2
                    if t in {"he", "him", "il", "él"} and (gender == "Masc"):
                        score += 0.2
                    if t in {"she", "her", "elle", "ella", "lei"} and (gender == "Fem"):
                        score += 0.2
                    if score > best_score:
                        best_score = score
                        best = cand
                # Sanity filter: avoid returning long or verb-like phrases as antecedents
                def acceptable(x: Optional[str]) -> bool:
                    if not x:
                        return False
                    toks = x.split()
                    if len(toks) > 2:
                        return False
                    # Avoid candidates containing common clitics/pronouns in es/fr
                    bad_bits = {"lo", "la", "le", "les", "y", "en"}
                    for bb in bad_bits:
                        if f" {bb} " in f" {x} ":
                            return False
                    return True
                if acceptable(best):
                    return best
                # Fallback: pick most recent acceptable candidate by type
                for cand, is_person, head_lemma, gender, number in reversed(antecedents):
                    if prefer_person and is_person and acceptable(cand):
                        return cand
                for cand, _is_person, _h, _g, _n in reversed(antecedents):
                    if acceptable(cand):
                        return cand
                return best
            if t in pronouns and not antecedents and future_proper_nouns:
                # cataphora fallback: first proper noun in document
                return future_proper_nouns[0]
            # definite NP coref ("the X") → prefer recent X
            if t.startswith("the "):
                base = t[4:]
                for cand, _is_person, _h, _g, _n in reversed(antecedents):
                    if base in cand or cand in base:
                        return cand
            return None

        # Build list of previous concrete objects to help resolve object clitics (es/it/fr)
        prev_objects: List[str] = []

        # Rewrite triples inplace when confident
        for i, (s, r, d) in enumerate(triples):
            ns = resolve(s) or s
            nd = resolve(d)
            # If object pronoun resolves to subject, try previous distinct antecedent
            d_is_pron = _safe_lower(d) in pronouns
            if d_is_pron and (nd is None or nd == ns):
                for cand, _is_person, _h, _g, _n in reversed(antecedents):
                    if cand != ns:
                        nd = cand
                        break
            # Generic clitic fallback: if still unresolved or still pronoun-like, use last concrete object
            if (nd is None or _safe_lower(nd) in pronouns or len(nd.split()) > 2) and prev_objects:
                nd = prev_objects[-1]
            if nd is None:
                nd = d
            triples[i] = (ns, r, nd)
            # Track concrete objects for subsequent use
            if nd and _safe_lower(nd) not in pronouns and len(nd.split()) <= 3:
                prev_objects.append(nd)

    # ----------------------------- Shared NP helpers -----------------------------

    def _np_with_modifiers(self, n: Token) -> str:
        try:
            for chunk in getattr(n.doc, "noun_chunks", []) or []:
                if getattr(chunk, "root", None) is n:
                    return _canon_entity_text(chunk.text.lower())
        except Exception:
            pass
        collected = [n]
        try:
            for lc in n.lefts:
                if lc.dep_ in {"compound", "amod", "nummod"}:
                    collected.append(lc)
        except Exception:
            pass
        toks = sorted({t.i: t for t in collected}.values(), key=lambda t: t.i)
        phrase = " ".join(t.text.lower() for t in toks)
        return _canon_entity_text(phrase)

    def _only_internal_pps(self, n: Token) -> str:
        parts: List[str] = []
        for ch in n.children:
            if ch.dep_ == "prep":
                prep = ch.text.lower()
                pobj = None
                for gc in ch.children:
                    if gc.dep_ in {"pobj", "obj"}:
                        pobj = gc
                        break
                if pobj is not None:
                    det = None
                    for gc in pobj.children:
                        if gc.dep_ in {"det", "det:poss"}:
                            det = gc.text.lower()
                            break
                    target = self._np_with_modifiers(pobj)
                    parts.append(f"{prep} {target if target else pobj.text.lower()}")
        return " ".join(parts)

    def _object_text_from(self, v: Token, tok: Token) -> str:
        try:
            if tok.head is not None and tok.head.dep_ == "prep" and tok.head.head.i == v.i:
                np = self._np_with_modifiers(tok)
                return f"{tok.head.text.lower()} {np}".strip()
        except Exception:
            pass
        base_np = self._np_with_modifiers(tok)
        extra_pp = self._only_internal_pps(tok)
        if extra_pp and base_np:
            if extra_pp.startswith(base_np + " "):
                return extra_pp
            return f"{base_np} {extra_pp}".strip()
        return base_np or extra_pp

    def _micro_appositive_alias(self, doc: Doc, triples: List[Tuple[str, str, str]]) -> None:
        """Merge appositive aliases like 'Alice, a doctor,' → alias(Alice, doctor).

        Strategy: build union-find of appositive pairs and rewrite triples to the
        canonical (shortest) surface form.
        """
        pairs: List[Tuple[str, str]] = []
        for tok in doc:
            if tok.dep_ == "appos" and tok.head is not None:
                a = _canon_entity_text(tok.head.text)
                b = _canon_entity_text(tok.text)
                if a and b and a != b:
                    pairs.append((a, b))

        if not pairs:
            return
        # Union-Find
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            # prefer shorter canonical form
            parent[max(rx, ry, key=len)] = min(rx, ry, key=len)

        for a, b in pairs:
            union(a, b)

        # Build mapping to canonical
        canon: Dict[str, str] = {k: find(k) for k in list(parent.keys())}
        for i, (s, r, d) in enumerate(triples):
            s2 = canon.get(_canon_entity_text(s), s)
            d2 = canon.get(_canon_entity_text(d), d)
            triples[i] = (s2, r, d2)

    def _fallback_copula_adjectival(self, doc: Doc, triples: List[Tuple[str, str, str]]) -> None:
        """If parser missed complements, try to recover 'N be ADJ/N'."""
        # Build a set to avoid duplicate entries
        existing = set((s, r, d) for s, r, d in triples)
        for sent in (list(doc.sents) if doc.has_annotation("SENT_START") else [doc]):
            # Prefer explicit copula tokens
            copulas = [t for t in sent if _safe_lower(getattr(t, "lemma_", getattr(t, "text", ""))) in {"be", "is", "are", "was", "were"}]
            if not copulas:
                # Weak fallback: any AUX as copula
                copulas = [t for t in sent if t.pos_ == "AUX"]
            for cop in copulas:
                subj = None
                for ch in cop.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                        break
                if subj is None:
                    continue
                s = _canon_entity_text(subj.text)
                # Candidate complements
                comp = None
                # direct acomp/attr
                for ch in cop.children:
                    if ch.dep_ in {"acomp", "attr"} and _canon_entity_text(ch.text) != s:
                        comp = ch
                        break
                # scan to the right for first ADJ
                if comp is None:
                    for t in sent:
                        if t.i > cop.i and t.pos_ == "ADJ":
                            comp = t
                            break
                # if still none, take first NOUN to the right not equal to subject
                if comp is None:
                    for t in sent:
                        if t.i > cop.i and t.pos_ in {"NOUN", "PROPN"} and _canon_entity_text(t.text) != s:
                            comp = t
                            break
                # final resort: immediate token to the right (non-punct)
                if comp is None:
                    idx = cop.i + 1
                    while idx < len(doc):
                        t = doc[idx]
                        if t.is_punct:
                            idx += 1
                            continue
                        if _canon_entity_text(t.text) != s:
                            comp = t
                        break
                if comp is not None:
                    d = _canon_entity_text(comp.text)
                    if s and d and s != d and (s, "be", d) not in existing and (s, "is", d) not in existing:
                        triples.append((s, "be", d))
                        existing.add((s, "be", d))

    def _ensure_intransitive_subjects(self, doc: Doc, triples: List[Tuple[str, str, str]]) -> None:
        """Ensure intransitive verbs with subjects yield subject-only triples."""
        existing_pairs = set((s, r) for s, r, _ in triples)
        for t in doc:
            if t.pos_ in {"VERB", "AUX"}:
                subj = None
                obj = None
                for ch in t.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                    elif ch.dep_ in {"obj", "dobj", "iobj", "pobj"}:
                        obj = ch
                if subj is not None and obj is None:
                    s = _canon_entity_text(subj.text)
                    r = _safe_lower(getattr(t, "lemma_", getattr(t, "text", "")))
                    if (s, r) not in existing_pairs:
                        triples.append((s, r, ""))
                        existing_pairs.add((s, r))

    def _regex_copula_adjectival(self, text: str, triples: List[Tuple[str, str, str]]) -> None:
        """Regex fallback for 'The|A|An NOUN is ADJ' regardless of parse status."""
        existing = set((s, r, d) for s, r, d in triples)
        pattern = re.compile(r"\b(?:the|a|an)\s+([A-Za-z][\w-]*)\s+(?:is|are|was|were)\s+([A-Za-z][\w-]*)\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            s = _canon_entity_text(m.group(1))
            d = _canon_entity_text(m.group(2))
            if s and d and s != d and (s, "be", d) not in existing and (s, "is", d) not in existing:
                triples.append((s, "be", d))
                existing.add((s, "be", d))

    def _regex_intransitive(self, text: str, triples: List[Tuple[str, str, str]]) -> None:
        """Regex fallback for 'NOUN VERB' in simple SVO-less clauses."""
        m = re.search(r"\b(?:the|a|an)?\s*([A-Za-z][\w-]*)\s+([A-Za-z][\w-]*)s?\b", text, re.IGNORECASE)
        if m:
            s = _canon_entity_text(m.group(1))
            r = _safe_lower(m.group(2))
            if s and r:
                triples.append((s, r, ""))

    def _sanitize_copula_pairs(self, triples: List[Tuple[str, str, str]]) -> None:
        """Remove self-referential copula pairs and deduplicate variants."""
        out: List[Tuple[str, str, str]] = []
        seen: set = set()
        for s, r, d in triples:
            if r in {"is", "be"} and _canon_entity_text(s) == _canon_entity_text(d):
                continue
            key = (s, r, d)
            if key not in seen:
                out.append(key)
                seen.add(key)
        triples[:] = out

    # ----------------------------- Light verb rewriting -----------------------------

    def _rewrite_light_verbs(self, triples: List[Tuple[str, str, str]], lang: str) -> None:
        if lang != "en":
            return
        flag = os.getenv("YAML_LIGHT_VERBS", "off").strip().lower()
        if flag in {"off", "false", "0"}:
            return
        # Map (light_verb, object_head) -> predicate
        LV: Dict[Tuple[str, str], str] = {
            ("make", "decision"): "decide",
            ("make", "choice"): "choose",
            ("make", "plan"): "plan",
            ("make", "promise"): "promise",
            ("make", "attempt"): "attempt",
            ("make", "effort"): "try",
            ("make", "change"): "change",
            ("take", "walk"): "walk",
            ("take", "look"): "look",
            ("take", "nap"): "nap",
            ("take", "part"): "participate",
            ("take", "care"): "care",
            ("give", "presentation"): "present",
            ("give", "talk"): "talk",
            ("give", "call"): "call",
            ("give", "advice"): "advise",
            ("give", "permission"): "permit",
            ("give", "approval"): "approve",
            ("give", "support"): "support",
            ("have", "look"): "look",
            ("have", "dinner"): "dine",
            ("have", "conversation"): "converse",
            ("have", "meeting"): "meet",
        }
        updated: List[Tuple[str, str, str]] = []
        for s, r, d in triples:
            rb = _safe_lower(r)
            db = (d or "").strip().lower()
            # object head = first token before a space or preposition
            head = db.split(" ")[0] if db else ""
            new_r = LV.get((rb, head))
            if new_r:
                # drop the light-noun head from object; keep trailing modifiers (e.g., "on X")
                rest = db[len(head):].strip()
                updated.append((s, new_r, rest))
            else:
                updated.append((s, r, d))
        triples[:] = updated

    def _rewrite_nominals(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str) -> None:
        if lang != "en":
            return
        flag = os.getenv("YAML_NOMINALS", "off").strip().lower()
        if flag in {"off", "false", "0"}:
            return
        added: Set[Tuple[str, str, str]] = set()
        change_heads = {
            "increase": "increase",
            "decrease": "decrease",
            "reduction": "decrease",
            "growth": "increase",
        }
        nominal_verbs = {
            "implementation": "implement",
            "expansion": "expand",
            # conservative broadening for risk/responsibility/cause
            "risk": "risk_of",
            "hazard": "risk_of",
            "danger": "risk_of",
            "threat": "risk_of",
            "responsibility": "responsible_for",
            "liability": "responsible_for",
            "cause": "cause_of",
            # additional nominal heads that map to lexicalized relations when preps present
            # handled below with prep-aware mapping
        }
        affect_heads = {"effect", "effects", "impact", "influence"}

        def pobj_of(head: Token, prep_txt: str) -> Optional[Token]:
            for ch in head.children:
                if ch.dep_ == "prep" and _safe_lower(ch.text) == prep_txt:
                    for gc in ch.children:
                        if gc.dep_ in {"pobj", "obj"}:
                            return gc
            return None

        for tok in doc:
            if getattr(tok, "pos_", "") != "NOUN":
                continue
            lem = _safe_lower(getattr(tok, "lemma_", getattr(tok, "text", "")))
            if lem in change_heads:
                # Prefer 'of X in Y' → (X, inc/dec, 'in Y'); also allow 'increase of X' and agent 'by Y' when safe
                x_of = pobj_of(tok, "of")
                x_in = pobj_of(tok, "in")
                by = pobj_of(tok, "by")
                if x_of is not None and x_in is not None:
                    tri = (_canon_entity_text(x_of.text), change_heads[lem], f"in {_canon_entity_text(x_in.text)}")
                    if tri not in added:
                        triples.append(tri)
                        added.add(tri)
                elif x_in is not None:
                    tri = (_canon_entity_text(x_in.text), change_heads[lem], "")
                    if tri not in added:
                        triples.append(tri)
                        added.add(tri)
                elif x_of is not None:
                    obj = f"by {_canon_entity_text(by.text)}" if by is not None else ""
                    tri = (_canon_entity_text(x_of.text), change_heads[lem], obj)
                    if tri not in added:
                        triples.append(tri)
                        added.add(tri)
            elif lem in nominal_verbs or lem in {"agreement", "application", "compliance", "adherence", "engagement", "entry"}:
                # Preposition-aware nominal mappings for select heads
                rel = None
                obj_tok = None
                if lem == "agreement":
                    # agreement on/with/to X → agree_on/agree_with/agree_to X
                    for prep, rel_name in (("on", "agree_on"), ("with", "agree_with"), ("to", "agree_to")):
                        pt = pobj_of(tok, prep)
                        if pt is not None:
                            rel = rel_name
                            obj_tok = pt
                            break
                elif lem == "application":
                    # application to/for X → apply_to/apply_for X
                    for prep, rel_name in (("to", "apply_to"), ("for", "apply_for")):
                        pt = pobj_of(tok, prep)
                        if pt is not None:
                            rel = rel_name
                            obj_tok = pt
                            break
                elif lem == "compliance":
                    pt = pobj_of(tok, "with")
                    if pt is not None:
                        rel = "comply_with"
                        obj_tok = pt
                elif lem == "adherence":
                    pt = pobj_of(tok, "to")
                    if pt is not None:
                        rel = "adhere_to"
                        obj_tok = pt
                elif lem == "engagement":
                    for prep, rel_name in (("with", "engage_with"), ("in", "engage_in")):
                        pt = pobj_of(tok, prep)
                        if pt is not None:
                            rel = rel_name
                            obj_tok = pt
                            break
                elif lem == "entry":
                    pt = pobj_of(tok, "into")
                    if pt is not None:
                        rel = "enter_into"
                        obj_tok = pt
                else:
                    # default for implementation/expansion: prefer 'of'
                    obj_tok = pobj_of(tok, "of") or pobj_of(tok, "for") or pobj_of(tok, "on")
                    if obj_tok is not None:
                        rel = nominal_verbs.get(lem)
                if rel and obj_tok is not None:
                    tri = ("", rel, _canon_entity_text(obj_tok.text))
                    if tri not in added:
                        triples.append(tri)
                        added.add(tri)
            elif lem in affect_heads:
                x = pobj_of(tok, "of")
                y = pobj_of(tok, "on") or pobj_of(tok, "upon")
                if x is not None and y is not None:
                    tri = (_canon_entity_text(x.text), "affect", _canon_entity_text(y.text))
                    if tri not in added:
                        triples.append(tri)
                        added.add(tri)

    # ----------------------------- Micro‑L3 stubs -----------------------------

    def _micro_l3(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str, rules: Set[str]):
        if lang == "es" and "ES_GUSTAR_PSYCH_VERBS" in rules:
            for tok in doc:
                base = _safe_lower(getattr(tok, "lemma_", getattr(tok, "text", "")))
                if (tok.pos_ in {"VERB", "AUX"} or base in {"gustar", "encantar", "interesar", "parecer", "doler", "importar"}) and base in {"gustar", "encantar", "interesar", "parecer", "doler", "importar"}:
                    experiencer = None
                    theme = None
                    objs = [ch for ch in tok.children if ch.dep_ in {"obj", "iobj", "obl"}]
                    # theme is grammatical subject
                    for ch in tok.children:
                        if ch.dep_ == "nsubj":
                            theme = ch
                            break
                    # experiencer: prefer NP with 'a' case, then clitic pronoun
                    for ch in objs:
                        has_a = any(gc.dep_ == "case" and _safe_lower(gc.text) == "a" for gc in ch.children)
                        if has_a:
                            experiencer = ch
                            break
                    if not experiencer:
                        for ch in objs:
                            if ch.pos_ == "PRON":
                                experiencer = ch
                                break
                    if experiencer and theme:
                        s = _canon_entity_text(experiencer.text)
                        d = _canon_entity_text(theme.text)
                        triples.append((s, "like", d))

        if lang == "de" and "DE_SEPARABLE_PREFIX_VERBS" in rules:
            # Basic reconstruction via compound:prt/obl:prt
            for tok in doc:
                if tok.pos_ in {"VERB", "AUX"} or _safe_lower(tok.text) in {"an", "auf", "zu", "ein"}:
                    prefix = None
                    subj = None
                    obj = None
                    for ch in tok.children:
                            # 'svp' is separable verb particle in de_core_news
                        if ch.dep_ in {"compound:prt", "obl:prt", "prt", "svp"}:
                            prefix = ch.text.lower()
                        elif ch.dep_ in {"nsubj", "csubj", "sb"}:
                            subj = ch
                        elif ch.dep_ in {"obj", "dobj", "oa"}:
                            obj = ch
                    if prefix and subj and obj:
                        full = f"{prefix}_{_safe_lower(tok.lemma_)}"
                        triples.append((_canon_entity_text(subj.text), full, _canon_entity_text(obj.text)))

        if lang == "fr" and "FR_CLITIC_PRONOUN_CLIMBING" in rules:
            # Lightweight: treat clitic + main verb as relation. Drop clitic pronouns as objects.
            clitic_prons = {"le", "la", "les", "l'", "l’", "lui", "leur", "y", "en"}
            lemma_fallback = {
                "veux": "vouloir", "veut": "vouloir", "veulent": "vouloir", "voulons": "vouloir", "voulez": "vouloir",
                "vais": "aller", "va": "aller", "vont": "aller", "allons": "aller", "allez": "aller",
                "aime": "aimer", "aiment": "aimer", "aimons": "aimer", "aimez": "aimer",
            }

            def infer_location_from_context(tk: Token) -> Optional[str]:
                # Look left for an oblique with case in {à, chez, dans, sur, en} → NOUN/PROPN
                preps = {"à", "chez", "dans", "sur", "en"}
                for i in range(tk.i - 1, -1, -1):
                    t = tk.doc[i]
                    if t.dep_ in {"obl", "obl:loc"}:
                        # check case marker
                        has_loc = any((_safe_lower(c.text) in preps) for c in t.children if c.dep_ == "case")
                        if has_loc and t.pos_ in {"NOUN", "PROPN"}:
                            return _canon_entity_text(t.text)
                return None
            def main_verb(t: Token) -> Token:
                if t.pos_ == "VERB":
                    return t
                # climb to a verb if AUX
                cur = t
                seen = set()
                while cur.head is not None and cur.head != cur and cur.i not in seen:
                    seen.add(cur.i)
                    cur = cur.head
                    if cur.pos_ == "VERB":
                        return cur
                return t
            for tok in doc:
                if tok.pos_ in {"VERB", "AUX"} or _safe_lower(tok.text) in lemma_fallback or _safe_lower(tok.text) in {"veux", "veut", "vont", "va", "aime"}:
                    mv = main_verb(tok)
                    subj = None
                    obj = None
                    for ch in mv.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                        elif ch.dep_ in {"obj", "obl"}:
                            obj = ch
                    # also look on auxiliary for clitic pronouns
                    if obj is None and tok != mv:
                        for ch in tok.children:
                            if ch.dep_ in {"obj", "obl"}:
                                obj = ch
                                break
                    # also recover subject from AUX if missing
                    if subj is None and tok != mv:
                        for ch in tok.children:
                            if ch.dep_ in {"nsubj", "csubj"}:
                                subj = ch
                                break
                    if subj:
                        rel = _safe_lower(mv.lemma_)
                        # Fallback lemma from surface form when model lemmatization fails
                        if not rel:
                            rel = lemma_fallback.get(_safe_lower(mv.text), rel)
                        if obj is not None and (obj.pos_ == "PRON" and _safe_lower(obj.text) in clitic_prons):
                            dest = ""
                            if _safe_lower(obj.text) == "y":
                                loc = infer_location_from_context(mv)
                                dest = loc or ""
                            else:
                                # Try infer last nominal object in context
                                last_noun = None
                                for i in range(mv.i - 1, -1, -1):
                                    t = mv.doc[i]
                                    if t.pos_ in {"NOUN", "PROPN"}:
                                        last_noun = t
                                        break
                                if last_noun is not None:
                                    dest = _canon_entity_text(last_noun.text)
                            triples.append((_canon_entity_text(subj.text), rel, dest))
                        elif obj is not None:
                            triples.append((_canon_entity_text(subj.text), rel, _canon_entity_text(obj.text)))
                        else:
                            # Fallback: if any clitic pronoun token exists in sentence, emit subject-only edge
                            if any((_safe_lower(tk.text) in clitic_prons) for tk in doc):
                                # Try to infer object from previous context
                                last_noun = None
                                for i in range(mv.i - 1, -1, -1):
                                    t = mv.doc[i]
                                    if t.pos_ in {"NOUN", "PROPN"}:
                                        last_noun = t
                                        break
                                dest = _canon_entity_text(last_noun.text) if last_noun is not None else ""
                                triples.append((_canon_entity_text(subj.text), rel, dest))
                # Root-based fallback for mis-tagged verbs (e.g., 'veux' as NOUN)
                if tok.dep_ == "ROOT":
                    subj = None
                    for ch in tok.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break
                    if subj and any((_safe_lower(tk.text) in clitic_prons) for tk in doc):
                        rel = lemma_fallback.get(_safe_lower(tok.text), _safe_lower(tok.lemma_))
                        # Try to infer object from previous context in the doc
                        last_noun = None
                        for i in range(tok.i - 1, -1, -1):
                            t = tok.doc[i]
                            if t.pos_ in {"NOUN", "PROPN"}:
                                last_noun = t
                                break
                        dest = _canon_entity_text(last_noun.text) if last_noun is not None else ""
                        triples.append((_canon_entity_text(subj.text), rel or _safe_lower(tok.text), dest))

        if lang == "fr" and "FR_PARTITIVE_CONSTRUCTIONS" in rules:
            # Handle partitive clitic 'en' → drop object to empty while keeping predicate edge
            for tok in doc:
                if tok.pos_ == "VERB":
                    subj = None
                    has_en_obj = False
                    for ch in tok.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                        # FR models often tag 'obj' PRON as 'en'
                        if ch.dep_ == "obj" and _safe_lower(ch.text) == "en":
                            has_en_obj = True
                    if subj and has_en_obj:
                        triples.append((_canon_entity_text(subj.text), _safe_lower(tok.lemma_), ""))

        if lang == "zh" and "ZH_SERIAL_VERB_CONSTRUCTIONS" in rules:
            # Serial verbs: link V1→V2 with purpose/sequence; prefer first subject in clause
            verbs = [t for t in doc if t.pos_ == "VERB"]
            if len(verbs) >= 2:
                v1, v2 = verbs[0], verbs[1]
                subj = None
                for ch in v1.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                        break
                if subj:
                    # If V1 is motion verb like 去, link as purpose
                    if v1.lemma_ == "去" or v1.text == "去":
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(v1.lemma_)}_in_order_to", _safe_lower(v2.lemma_)))
                    else:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(v1.lemma_)}_then", _safe_lower(v2.lemma_)))

        if lang == "zh" and "ZH_TOPIC_COMMENT_STRUCTURE" in rules:
            # Detect simple Topic，Comment pattern and map 喜欢 → like
            # e.g., 北京，我喜欢。
            # Find the first fullwidth comma as a topic boundary
            tokens = list(doc)
            try:
                comma_idx = next(i for i,t in enumerate(tokens) if t.text in {"，", ","})
            except StopIteration:
                comma_idx = -1
            if comma_idx > 0:
                # pick last NOUN/PROPN before comma as topic
                topic_tok = None
                for t in tokens[:comma_idx][::-1]:
                    if t.pos_ in {"NOUN", "PROPN"}:
                        topic_tok = t
                        break
                if topic_tok:
                    # search for verb 喜欢 and its subject on the right side
                    subject_tok = None
                    like_tok = None
                    for t in tokens[comma_idx+1:]:
                        if t.lemma_ == "喜欢" or t.text == "喜欢":
                            like_tok = t
                        if t.dep_ in {"nsubj", "csubj"}:
                            subject_tok = t
                    if like_tok and subject_tok:
                        triples.append((_canon_entity_text(subject_tok.text), "like", _canon_entity_text(topic_tok.text)))

    # ----------------------------- Utilities -----------------------------

    def _get_nlp(self, lang: str) -> Optional[spacy.Language]:
        try:
            import os
            override = os.getenv(f"SPACY_MODEL_{lang.upper()}") or (os.getenv("SPACY_MODEL_EN") if lang == "en" else None) or os.getenv("SPACY_MODEL")
            if override:
                return spacy.load(override)
            if lang == "en":
                return spacy.load("en_core_web_sm")
            if lang == "zh":
                return spacy.load("zh_core_web_sm")
            return spacy.load(f"{lang}_core_news_sm")
        except Exception as e:
            logger.warning(f"spaCy model for lang='{lang}' unavailable: {e}")
            try:
                # Fallback to English model to keep pipeline functional
                return spacy.load("en_core_web_sm")
            except Exception as e2:
                logger.warning(f"Fallback to English model failed: {e2}")
                return None

    # ----------------------------- Micro‑L2 helpers -----------------------------

    def _micro_discourse_temporal(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str) -> None:
        def nearest_left_verb(idx: int) -> Optional[Token]:
            for i in range(idx - 1, -1, -1):
                t = doc[i]
                if t.pos_ == "VERB":
                    return t
            return None

        def nearest_right_verb(idx: int) -> Optional[Token]:
            for i in range(idx + 1, len(doc)):
                t = doc[i]
                if t.pos_ == "VERB":
                    return t
            return None

        def subject_of(verb: Token) -> Optional[Token]:
            for ch in verb.children:
                if ch.dep_ in {"nsubj", "csubj"}:
                    return ch
            return None

        cause_markers = {"because", "since", "porque", "parce", "parce que", "car"}
        result_markers = {"so", "therefore", "thus", "hence", "consequently", "donc", "alors"}
        contrast_markers = {"but", "however", "though", "although", "while", "toutefois", "cependant", "mais"}
        cond_markers = {"if", "unless", "si", "à moins que"}
        temp_after = {"after", "afterward", "afterwards", "después", "après", "然后", "後来"}
        temp_before = {"before", "beforehand", "antes", "avant"}
        temp_when = {"when", "whenever", "cuando", "quand", "当"}
        then_markers = {"then", "then,", "subsequently", "later", "luego", "puis", "然后"}

        cause_phrases = [("because", "of"), ("due", "to")]
        result_phrases = [("as", "a", "result"), ("as", "a", "consequence")]
        cause_phr_single = {"owing": "to"}
        purpose_phrases = [("in", "order", "to"), ("so", "that", "")]  # 'so that' two-word start; third is placeholder
        purpose_phrases_3 = [("so", "as", "to")]
        result_of_phrases = [("as", "a", "result", "of"), ("as", "a", "consequence", "of")]

        for i, tok in enumerate(doc):
            w = _safe_lower(tok.text)
            if w in cause_markers:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_because_of", _safe_lower(rv.lemma_)))
            # phrase: because of / due to
            for a, b in cause_phrases:
                if i + 1 < len(doc) and _safe_lower(doc[i].text) == a and _safe_lower(doc[i+1].text) == b:
                    lv = nearest_left_verb(i)
                    rv = nearest_right_verb(i+1)
                    if lv is not None and rv is not None:
                        subj = subject_of(lv)
                        if subj is not None:
                            triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_because_of", _safe_lower(rv.lemma_)))
            # phrase: as a result / as a consequence
            for a, b, c in result_phrases:
                if i + 2 < len(doc) and _safe_lower(doc[i].text) == a and _safe_lower(doc[i+1].text) == b and _safe_lower(doc[i+2].text) == c:
                    lv = nearest_left_verb(i)
                    rv = nearest_right_verb(i+2)
                    if lv is not None and rv is not None:
                        subj = subject_of(lv)
                        if subj is not None:
                            triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_result", _safe_lower(rv.lemma_)))
            # phrase: as a result/consequence of
            for a, b, c, d in result_of_phrases:
                if i + 3 < len(doc) and _safe_lower(doc[i].text) == a and _safe_lower(doc[i+1].text) == b and _safe_lower(doc[i+2].text) == c and _safe_lower(doc[i+3].text) == d:
                    lv = nearest_left_verb(i)
                    rv = nearest_right_verb(i+3)
                    if lv is not None and rv is not None:
                        subj = subject_of(lv)
                        if subj is not None:
                            triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_result", _safe_lower(rv.lemma_)))
            # phrase: in order to / so that / so as to
            for a, b, c in purpose_phrases:
                if a == "in":
                    if i + 2 < len(doc) and _safe_lower(doc[i].text) == a and _safe_lower(doc[i+1].text) == b and _safe_lower(doc[i+2].text) == c:
                        lv = nearest_left_verb(i)
                        rv = nearest_right_verb(i+2)
                        if lv is not None and rv is not None:
                            subj = subject_of(lv)
                            if subj is not None:
                                triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_in_order_to", _safe_lower(rv.lemma_)))
                elif a == "so":
                    if i + 1 < len(doc) and _safe_lower(doc[i].text) == "so" and _safe_lower(doc[i+1].text) == "that":
                        lv = nearest_left_verb(i)
                        rv = nearest_right_verb(i+1)
                        if lv is not None and rv is not None:
                            subj = subject_of(lv)
                            if subj is not None:
                                triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_so_that", _safe_lower(rv.lemma_)))
            for a, b, c in purpose_phrases_3:
                if i + 2 < len(doc) and _safe_lower(doc[i].text) == a and _safe_lower(doc[i+1].text) == b and _safe_lower(doc[i+2].text) == c:
                    lv = nearest_left_verb(i)
                    rv = nearest_right_verb(i+2)
                    if lv is not None and rv is not None:
                        subj = subject_of(lv)
                        if subj is not None:
                            triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_so_as_to", _safe_lower(rv.lemma_)))
            # single-token start + following token (owing to)
            if _safe_lower(tok.text) in cause_phr_single:
                nxt = cause_phr_single[_safe_lower(tok.text)]
                if i + 1 < len(doc) and _safe_lower(doc[i+1].text) == nxt:
                    lv = nearest_left_verb(i)
                    rv = nearest_right_verb(i+1)
                    if lv is not None and rv is not None:
                        subj = subject_of(lv)
                        if subj is not None:
                            triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_because_of", _safe_lower(rv.lemma_)))
            if w in temp_after:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_after", _safe_lower(rv.lemma_)))
            if w in temp_before:
                rv = nearest_right_verb(i)
                lv = nearest_left_verb(i)
                # Case A: 'X ... before Y ...' → X_after Y (both sides present)
                if rv is not None and lv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_after", _safe_lower(rv.lemma_)))
                # Case B: 'Before Y ..., X ...' → Y_after X (two verbs to the right)
                elif rv is not None and lv is None:
                    # Find the next verb after rv
                    next_v = None
                    j = rv.i + 1
                    while j < len(doc):
                        if doc[j].pos_ == "VERB":
                            next_v = doc[j]
                            break
                        j += 1
                    if next_v is not None:
                        subj = subject_of(rv)  # subject of earlier event
                        if subj is not None:
                            triples.append((_canon_entity_text(subj.text), f"{_safe_lower(rv.lemma_)}_after", _safe_lower(next_v.lemma_)))
            if w in temp_when:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_when", _safe_lower(rv.lemma_)))
            if w in then_markers:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_then", _safe_lower(rv.lemma_)))
            if w in contrast_markers:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_contrast", _safe_lower(rv.lemma_)))
            if w in cond_markers:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_if", _safe_lower(rv.lemma_)))
            if w in result_markers:
                lv = nearest_left_verb(i)
                rv = nearest_right_verb(i)
                if lv is not None and rv is not None:
                    subj = subject_of(lv)
                    if subj is not None:
                        triples.append((_canon_entity_text(subj.text), f"{_safe_lower(lv.lemma_)}_result", _safe_lower(rv.lemma_)))

    def _emit_appos_edges(self, doc: Doc, triples: List[Tuple[str, str, str]], lang: str) -> None:
        flag = os.getenv("YAML_APPOS_AS_EDGE", "on").strip().lower()
        if flag in {"off", "false", "0"}:
            return
        # Conservative rule: PROPN head with NOUN/PROPN appos child
        added: Set[Tuple[str, str, str]] = set()
        for tok in doc:
            if tok.dep_ == "appos" and tok.head is not None:
                head = tok.head
                if getattr(head, "pos_", "") in {"PROPN", "NOUN"} and getattr(tok, "pos_", "") in {"NOUN", "PROPN"}:
                    s = _canon_entity_text(head.text)
                    d = _canon_entity_text(tok.text)
                    if s and d and s != d:
                        tri = (s, "is", d)
                        if tri not in added:
                            triples.append(tri)
                            added.add(tri)

    def _micro_cluster_entities(self, triples: List[Tuple[str, str, str]]) -> None:
        # Simple aliasing by canonical head and singularization
        def singularize(x: str) -> str:
            if x.endswith("s") and len(x) > 3:
                return x[:-1]
            return x

        mentions: Set[str] = set()
        for s, _, d in triples:
            if s:
                mentions.add(_canon_entity_text(s))
            if d:
                mentions.add(_canon_entity_text(d))

        mapping: Dict[str, str] = {}
        by_root: Dict[str, str] = {}
        for m in mentions:
            root = singularize(m)
            if root not in by_root or len(m) < len(by_root[root]):
                by_root[root] = m
        for m in mentions:
            mapping[m] = by_root[singularize(m)]

        for i, (s, r, d) in enumerate(triples):
            s2 = mapping.get(_canon_entity_text(s), s)
            d2 = mapping.get(_canon_entity_text(d), d)
            triples[i] = (s2, r, d2)

    # ====== ALL 17 MISSING L1 PATTERN IMPLEMENTATIONS ======

    def _match_svo_passive(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract passive voice constructions: 'The book was written by John' → ('book', 'written_by', 'John')"""
        for tok in doc:
            if tok.dep_ == "nsubjpass" and tok.head.pos_ == "VERB":
                verb = tok.head
                agent = None
                for child in verb.children:
                    if child.dep_ == "agent" or (child.dep_ == "prep" and child.lemma_ == "by"):
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                agent = grandchild
                                break

                if agent:
                    s = _canon_entity_text(tok.text)
                    r = f"{_safe_lower(verb.lemma_)}_by"
                    d = _canon_entity_text(agent.text)
                    triples.append((s, r, d))
                    entities.update([s, d])

    def _match_copula_adjectival(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]]):
        """Extract copula with adjectives: 'John is happy' → ('John', 'be', 'happy')"""
        for tok in doc:
            if tok.dep_ == "acomp" and tok.pos_ == "ADJ":
                head = tok.head
                subj = None
                for ch in head.children:
                    if ch.dep_ in {"nsubj", "csubj"}:
                        subj = ch
                        break
                if subj:
                    s = _canon_entity_text(subj.text)
                    r = "be"
                    d = _canon_entity_text(tok.text)
                    triples.append((s, r, d))
                    entities.add(s)

    def _match_coord_subject(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract coordinated subjects: 'John and Mary work at Google' → multiple triples"""
        for tok in doc:
            if tok.dep_ == "nsubj" and tok.pos_ in {"NOUN", "PROPN"}:
                coords = []
                for child in tok.children:
                    if child.dep_ == "conj":
                        coords.append(child)

                if coords and tok.head.pos_ == "VERB":
                    verb = tok.head
                    obj = None
                    for ch in verb.children:
                        if ch.dep_ in {"dobj", "pobj"}:
                            obj = ch
                            break

                    s = _canon_entity_text(tok.text)
                    r = _safe_lower(verb.lemma_)
                    d = _canon_entity_text(obj.text) if obj else ""
                    if d:
                        triples.append((s, r, d))
                        entities.update([s, d])

                    for coord in coords:
                        s = _canon_entity_text(coord.text)
                        if d:
                            triples.append((s, r, d))
                            entities.add(s)

    def _match_coord_object(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract coordinated objects: 'John likes cats and dogs' → multiple triples"""
        for tok in doc:
            if tok.dep_ in {"dobj", "pobj"} and tok.pos_ in {"NOUN", "PROPN"}:
                coords = []
                for child in tok.children:
                    if child.dep_ == "conj":
                        coords.append(child)

                if coords:
                    head = tok.head
                    subj = None
                    for ch in head.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break

                    if subj:
                        s = _canon_entity_text(subj.text)
                        r = _safe_lower(head.lemma_)
                        d = _canon_entity_text(tok.text)
                        triples.append((s, r, d))
                        entities.update([s, d])

                        for coord in coords:
                            d = _canon_entity_text(coord.text)
                            triples.append((s, r, d))
                            entities.add(d)

    def _match_coord_verb(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract coordinated verbs: 'John runs and jumps' → multiple triples"""
        for tok in doc:
            if tok.pos_ == "VERB" and tok.dep_ == "ROOT":
                coords = []
                for child in tok.children:
                    if child.dep_ == "conj" and child.pos_ == "VERB":
                        coords.append(child)

                if coords:
                    subj = None
                    for ch in tok.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break

                    if subj:
                        s = _canon_entity_text(subj.text)
                        r = _safe_lower(tok.lemma_)
                        obj = None
                        for ch in tok.children:
                            if ch.dep_ in {"dobj", "pobj"}:
                                obj = ch
                                break
                        d = _canon_entity_text(obj.text) if obj else ""
                        triples.append((s, r, d))
                        entities.add(s)
                        if d:
                            entities.add(d)

                        for coord_verb in coords:
                            r = _safe_lower(coord_verb.lemma_)
                            obj = None
                            for ch in coord_verb.children:
                                if ch.dep_ in {"dobj", "pobj"}:
                                    obj = ch
                                    break
                            d = _canon_entity_text(obj.text) if obj else ""
                            triples.append((s, r, d))
                            if d:
                                entities.add(d)

    def _match_coord_mixed(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Handle mixed coordination patterns where both subject and object are coordinated.

        Example: "Alice and Bob founded and led Acme and Beta" →
        (alice, found, acme), (alice, found, beta), (bob, found, acme), ... (bob, lead, beta)
        Conservatively emits pairs when both sides have simple coordinations.
        """
        for tok in doc:
            if getattr(tok, "pos_", "") != "VERB" or tok.dep_ not in {"ROOT", "conj"}:
                continue
            # Collect coordinated subjects for this verb
            subj_tokens: List[Token] = []
            for ch in tok.children:
                if ch.dep_ in {"nsubj", "csubj"} and getattr(ch, "pos_", "") in {"NOUN", "PROPN", "PRON"}:
                    subj_tokens.append(ch)
                    for cc in ch.children:
                        if cc.dep_ == "conj" and getattr(cc, "pos_", "") in {"NOUN", "PROPN", "PRON"}:
                            subj_tokens.append(cc)
            if not subj_tokens:
                continue
            # Collect coordinated objects for this verb (direct object or pobj)
            obj_tokens: List[Token] = []
            for ch in tok.children:
                if ch.dep_ in {"obj", "dobj"} and getattr(ch, "pos_", "") in {"NOUN", "PROPN"}:
                    obj_tokens.append(ch)
                    for cc in ch.children:
                        if cc.dep_ == "conj" and getattr(cc, "pos_", "") in {"NOUN", "PROPN"}:
                            obj_tokens.append(cc)
                elif ch.dep_ == "prep":
                    for gc in ch.children:
                        if gc.dep_ in {"pobj", "obj"} and getattr(gc, "pos_", "") in {"NOUN", "PROPN"}:
                            obj_tokens.append(gc)
                            for cc in gc.children:
                                if cc.dep_ == "conj" and getattr(cc, "pos_", "") in {"NOUN", "PROPN"}:
                                    obj_tokens.append(cc)
            if not obj_tokens:
                continue
            # Emit cross-product for this verb and its coordinated sibling verbs
            verbs = [tok]
            for ch in tok.children:
                if ch.dep_ == "conj" and ch.pos_ == "VERB":
                    verbs.append(ch)
            for v in verbs:
                rel = _safe_lower(getattr(v, "lemma_", getattr(v, "text", "")))
                for s_tok in subj_tokens:
                    s = _canon_entity_text(s_tok.text)
                    entities.add(s)
                    for o_tok in obj_tokens:
                        d = _canon_entity_text(o_tok.text)
                        triples.append((s, rel, d))
                        entities.add(d)

    def _match_ditransitive_give(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract ditransitive give patterns: 'John gave Mary a book' → ('John', 'gave_to', 'Mary')"""
        give_verbs = {"give", "send", "show", "tell", "offer", "lend", "bring", "hand", "pass"}

        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ in give_verbs:
                subj = None
                iobj = None
                dobj = None

                for child in tok.children:
                    if child.dep_ in {"nsubj", "csubj"}:
                        subj = child
                    elif child.dep_ == "iobj":
                        iobj = child
                    elif child.dep_ == "dobj":
                        dobj = child
                    elif child.dep_ == "dative":
                        iobj = child

                if subj and iobj:
                    s = _canon_entity_text(subj.text)
                    r = f"{_safe_lower(tok.lemma_)}_to"
                    d = _canon_entity_text(iobj.text)
                    triples.append((s, r, d))
                    entities.update([s, d])

                    if dobj:
                        r = _safe_lower(tok.lemma_)
                        d = _canon_entity_text(dobj.text)
                        triples.append((s, r, d))
                        entities.add(d)

    def _match_ditransitive_communicate(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract communication patterns broadly.

        Handles variants:
        - 'Alice told Bob the news' → (alice, tell, bob) and optionally (alice, tell_about, news)
        - 'Alice told the news to Bob' → recognizes pobj of 'to' as recipient
        - 'Alice informed Bob about changes' → _about content edge
        """
        comm_verbs = {"tell", "inform", "notify", "warn", "remind", "ask", "advise"}

        for tok in doc:
            if getattr(tok, "pos_", "") == "VERB" and _safe_lower(getattr(tok, "lemma_", tok.text)) in comm_verbs:
                subj = None
                person = None
                content = None

                for child in tok.children:
                    dep = child.dep_
                    if dep in {"nsubj", "csubj"}:
                        subj = child
                    elif dep in {"iobj", "dative"}:
                        person = child
                    elif dep in {"dobj", "obj"} and getattr(child, "pos_", "") in {"NOUN", "PROPN", "PRON"}:
                        # Prefer this as content; keep person for iobj/dative
                        if content is None:
                            content = child
                # Prepositional recipients and content
                for child in tok.children:
                    if child.dep_ == "prep":
                        prep = _safe_lower(child.text)
                        for gc in child.children:
                            if gc.dep_ in {"pobj", "obj"}:
                                if prep == "to" and person is None:
                                    person = gc
                                elif prep in {"about", "of"} and content is None:
                                    content = gc

                if subj is not None and (person is not None or content is not None):
                    s = _canon_entity_text(subj.text)
                    rel = _safe_lower(getattr(tok, "lemma_", tok.text))
                    if person is not None:
                        d = _canon_entity_text(person.text)
                        triples.append((s, rel, d))
                        entities.update([s, d])
                    if content is not None:
                        d2 = _canon_entity_text(content.text)
                        triples.append((s, f"{rel}_about", d2))
                        entities.add(d2)

    def _match_control_verb(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract control verb patterns: 'John wants to leave' → ('John', 'wants_to', 'leave')"""
        control_verbs = {"want", "need", "try", "hope", "plan", "decide", "agree", "refuse", "promise", "expect"}

        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ in control_verbs:
                subj = None
                xcomp = None

                for child in tok.children:
                    if child.dep_ in {"nsubj", "csubj"}:
                        subj = child
                    elif child.dep_ == "xcomp":
                        xcomp = child

                if subj and xcomp:
                    s = _canon_entity_text(subj.text)
                    r = f"{_safe_lower(tok.lemma_)}_to"
                    d = _safe_lower(xcomp.lemma_)
                    triples.append((s, r, d))
                    entities.add(s)

    def _match_ccomp_embedding(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract clausal complement patterns: 'John thinks that Mary is smart' → ('John', 'thinks_that', '...')"""
        embedding_verbs = {"think", "believe", "know", "say", "claim", "argue", "suggest", "doubt"}

        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ in embedding_verbs:
                subj = None
                ccomp = None

                for child in tok.children:
                    if child.dep_ in {"nsubj", "csubj"}:
                        subj = child
                    elif child.dep_ == "ccomp":
                        ccomp = child

                if subj and ccomp:
                    s = _canon_entity_text(subj.text)
                    r = f"{_safe_lower(tok.lemma_)}_that"
                    ccomp_subj = None
                    for ch in ccomp.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            ccomp_subj = ch
                            break

                    if ccomp_subj:
                        d = f"{_canon_entity_text(ccomp_subj.text)} {_safe_lower(ccomp.lemma_)}"
                    else:
                        d = _safe_lower(ccomp.text)

                    triples.append((s, r, d))
                    entities.add(s)

    def _match_modal_verbs(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract modal verb patterns: 'John can swim' → ('John', 'can', 'swim')"""
        modals = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}

        for tok in doc:
            if tok.pos_ == "AUX" and tok.lemma_ in modals:
                subj = None
                main_verb = None

                for child in tok.children:
                    if child.dep_ in {"nsubj", "csubj"}:
                        subj = child
                    elif child.pos_ == "VERB":
                        main_verb = child

                if not main_verb and tok.head.pos_ == "VERB":
                    main_verb = tok.head
                    for ch in main_verb.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break

                if subj and main_verb:
                    s = _canon_entity_text(subj.text)
                    r = _safe_lower(tok.lemma_)
                    d = _safe_lower(main_verb.lemma_)
                    triples.append((s, r, d))
                    entities.add(s)

    def _match_temporal_adverbials(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract temporal information: 'John worked yesterday' → ('John worked', 'when', 'yesterday')"""
        temporal_markers = {"yesterday", "today", "tomorrow", "now", "then", "soon", "later", "recently"}

        for tok in doc:
            if tok.dep_ in {"advmod", "npadvmod", "tmod"} and _safe_lower(tok.text) in temporal_markers:
                head = tok.head
                if head.pos_ == "VERB":
                    subj = None
                    for ch in head.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break

                    if subj:
                        s = f"{_canon_entity_text(subj.text)} {_safe_lower(head.lemma_)}"
                        r = "when"
                        d = _safe_lower(tok.text)
                        triples.append((s, r, d))

    def _match_spatial_prepositions(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract spatial relations: 'The book is on the table' → ('book', 'on', 'table')"""
        spatial_preps = {"in", "on", "at", "under", "over", "beside", "behind", "near", "between", "above", "below"}

        for tok in doc:
            if tok.dep_ == "prep" and tok.lemma_ in spatial_preps:
                head = tok.head
                pobj = None
                for child in tok.children:
                    if child.dep_ == "pobj":
                        pobj = child
                        break

                if pobj and head.pos_ in {"NOUN", "PROPN", "VERB"}:
                    if head.pos_ == "VERB":
                        subj = None
                        for ch in head.children:
                            if ch.dep_ in {"nsubj", "csubj"}:
                                subj = ch
                                break
                        if subj:
                            s = _canon_entity_text(subj.text)
                            r = f"{_safe_lower(head.lemma_)}_{tok.lemma_}"
                            d = _canon_entity_text(pobj.text)
                            triples.append((s, r, d))
                            entities.update([s, d])
                    else:
                        s = _canon_entity_text(head.text)
                        r = tok.lemma_
                        d = _canon_entity_text(pobj.text)
                        triples.append((s, r, d))
                        entities.update([s, d])

    def _match_quantifier_scope(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract quantifier information: 'All students passed' → ('all students', 'passed', '')"""
        quantifiers = {"all", "every", "some", "many", "few", "several", "most", "any", "no"}

        for tok in doc:
            if tok.dep_ == "det" and _safe_lower(tok.text) in quantifiers:
                head = tok.head
                if head.dep_ in {"nsubj", "dobj"} and head.head.pos_ == "VERB":
                    verb = head.head
                    s = f"{_safe_lower(tok.text)} {_canon_entity_text(head.text)}"
                    r = _safe_lower(verb.lemma_)

                    obj = None
                    for ch in verb.children:
                        if ch.dep_ in {"dobj", "pobj"} and ch != head:
                            obj = ch
                            break

                    d = _canon_entity_text(obj.text) if obj else ""
                    triples.append((s, r, d))
                    if d:
                        entities.add(d)

    def _match_negation_scope(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str) -> int:
        """Extract negation information and return count of negations"""
        neg_count = 0

        for tok in doc:
            if tok.dep_ == "neg":
                neg_count += 1
                head = tok.head

                if head.pos_ == "VERB":
                    subj = None
                    for ch in head.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break

                    if subj:
                        s = _canon_entity_text(subj.text)
                        r = f"not_{_safe_lower(head.lemma_)}"
                        obj = None
                        for ch in head.children:
                            if ch.dep_ in {"dobj", "pobj"}:
                                obj = ch
                                break
                        d = _canon_entity_text(obj.text) if obj else ""
                        triples.append((s, r, d))
                        entities.add(s)
                        if d:
                            entities.add(d)

        return neg_count

    def _match_progressive_aspect(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract progressive aspect: 'John is running' → ('John', 'is_running', '')"""
        for tok in doc:
            if tok.tag_ in {"VBG"}:
                aux = None
                for child in tok.children:
                    if child.pos_ == "AUX" and child.lemma_ == "be":
                        aux = child
                        break

                if not aux and tok.head.pos_ == "AUX" and tok.head.lemma_ == "be":
                    aux = tok.head

                if aux:
                    subj = None
                    for ch in tok.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break
                    if not subj:
                        for ch in aux.children:
                            if ch.dep_ in {"nsubj", "csubj"}:
                                subj = ch
                                break

                    if subj:
                        s = _canon_entity_text(subj.text)
                        r = f"is_{_safe_lower(tok.lemma_)}"
                        obj = None
                        for ch in tok.children:
                            if ch.dep_ in {"dobj", "pobj"}:
                                obj = ch
                                break
                        d = _canon_entity_text(obj.text) if obj else ""
                        triples.append((s, r, d))
                        entities.add(s)
                        if d:
                            entities.add(d)

    def _match_perfect_aspect(self, doc: Doc, entities: Set[str], triples: List[Tuple[str, str, str]], lang: str):
        """Extract perfect aspect: 'John has eaten' → ('John', 'has_eaten', '')"""
        for tok in doc:
            if tok.tag_ in {"VBN"}:
                aux = None
                for child in tok.children:
                    if child.pos_ == "AUX" and child.lemma_ == "have":
                        aux = child
                        break

                if not aux and tok.head.pos_ == "AUX" and tok.head.lemma_ == "have":
                    aux = tok.head

                if aux:
                    subj = None
                    for ch in tok.children:
                        if ch.dep_ in {"nsubj", "csubj"}:
                            subj = ch
                            break
                    if not subj:
                        for ch in aux.children:
                            if ch.dep_ in {"nsubj", "csubj"}:
                                subj = ch
                                break

                    if subj:
                        s = _canon_entity_text(subj.text)
                        r = f"has_{_safe_lower(tok.lemma_)}"
                        obj = None
                        for ch in tok.children:
                            if ch.dep_ in {"dobj", "pobj"}:
                                obj = ch
                                break
                        d = _canon_entity_text(obj.text) if obj else ""
                        triples.append((s, r, d))
                        entities.add(s)
                        if d:
                            entities.add(d)
