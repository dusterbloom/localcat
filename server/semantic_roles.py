"""
Semantic Role Labeling (SRL) utilities for HotMem

Goals:
- Provide universal roles (agent, patient, cause, temporal, location, destination, source, beneficiary, instrument)
- Work language-agnostically by using UD dependencies to approximate SRL
- Optionally normalize relations with cross-lingual embeddings

Usage: Enable via env HOTMEM_USE_SRL=true to prefer SRL extraction in HotMem
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _canon_entity_text(text: str) -> str:
    t = _norm(text)
    # Remove leading determiners/possessives
    for det in ("the", "a", "an", "my", "your", "his", "her", "their", "our", "its"):
        if t.startswith(det + " "):
            t = t[len(det) + 1 :]
            break
    if t.endswith("'s"):
        t = t[:-2]
    # Normalize common pronouns to "you" to align with system conventions
    if t in {"i", "me", "my", "mine", "myself", "your", "yours", "yourself"}:
        return "you"
    return t


@dataclass
class Predication:
    predicate: str  # lemma
    # role -> surface string
    roles: Dict[str, str]
    # Optional light metadata
    lang: str = "en"
    sent_text: str = ""


class RelationNormalizer:
    """
    Optional cross-lingual relation normalizer using sentence-transformers.
    Falls back to heuristics if the dependency is unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._util = None
        self._prototype_texts: List[str] = []
        self._prototype_labels: List[str] = []
        self._prototype_emb = None
        self._init_prototypes()

    def _init_prototypes(self) -> None:
        # Canonical relations supported by HotMem retrieval
        prototypes = {
            "lives_in": ["live in", "reside in", "dwell in"],
            "works_at": ["work at", "work for"],
            "teach_at": ["teach at"],
            "born_in": ["be born in", "be born at"],
            "moved_from": ["move from", "relocate from"],
            "went_to": ["go to", "went to"],
            "participated_in": ["participate in", "took part in"],
            "owns": ["own"],
            "has": ["have", "has", "possess"],
            "friend_of": ["be friend of", "friend of"],
            "name": ["name is", "be named"],
            "is": ["be", "is", "are"],
            "read": ["read"],
            "favorite_color": ["favorite color"],
            "favorite_number": ["favorite number"],
        }
        for label, phrases in prototypes.items():
            for p in phrases:
                self._prototype_texts.append(p)
                self._prototype_labels.append(label)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except Exception:
            logger.info("[SRL] sentence-transformers not available; using heuristics only")
            self._model = None
            self._util = None
            return
        model_name = os.getenv(
            "HOTMEM_REL_EMBED_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        try:
            self._model = SentenceTransformer(model_name)
            self._util = util
            self._prototype_emb = self._model.encode(self._prototype_texts, normalize_embeddings=True)
            logger.info(f"[SRL] Loaded relation embed model: {model_name}")
        except Exception as e:
            logger.warning(f"[SRL] Failed to load relation embed model: {e}")
            self._model = None
            self._util = None

    def _heuristic_label(self, pred: str, prep: Optional[str]) -> Optional[str]:
        p = _norm(pred)
        pr = _norm(prep or "")
        if p in {"live", "reside", "dwell"} and pr in {"in", "at"}:
            return "lives_in"
        if p in {"work", "works"} and pr in {"at", "for"}:
            return "works_at"
        if p in {"teach", "teaches", "taught"} and pr == "at":
            return "teach_at"
        if p in {"go", "went"} and pr == "to":
            return "went_to"
        if p in {"move", "moved"} and pr == "from":
            return "moved_from"
        if p in {"participate", "participated"} and pr == "in":
            return "participated_in"
        if p in {"be", "am", "is", "are"}:
            return "is"
        if p in {"have", "has", "had", "own", "owns"}:
            return "has" if p != "own" and p != "owns" else "owns"
        if p in {"name", "named"}:
            return "name"
        if p == "read":
            return "read"
        return None

    def normalize(self, predicate: str, roles: Dict[str, str], prep_hint: Optional[str] = None) -> str:
        """
        Map a predicate (+ optional preposition hint) to a canonical relation label.
        Uses multilingual sentence embeddings when available, otherwise heuristics.
        """
        # Heuristic shortcut
        label = self._heuristic_label(predicate, prep_hint)
        if label:
            return label

        # Compose a short description of the relation for embedding match
        subj = roles.get("agent") or roles.get("subject") or "subject"
        obj = roles.get("patient") or roles.get("object") or roles.get("destination") or roles.get("location") or "object"
        phrase = f"{predicate} {prep_hint or ''}".strip()
        rel_text = f"{subj} {phrase} {obj}".strip()

        # If model is unavailable, fallback to predicate lemma itself
        self._ensure_model()
        if self._model is None or self._prototype_emb is None:
            return predicate

        try:
            q = self._model.encode(rel_text, normalize_embeddings=True)
            scores = self._util.cos_sim(q, self._prototype_emb).cpu().tolist()[0]
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            return self._prototype_labels[best_i]
        except Exception as e:
            logger.debug(f"[SRL] embed normalize failed: {e}")
            return predicate


class SRLExtractor:
    """
    Lightweight SRL on top of UD parses.
    - Identifies predicate heads (VERB) per sentence
    - Assigns roles by mapping UD labels: nsubj->agent, obj/dobj->patient, iobj->recipient
      agent (by 'agent' in passive), obl with case to roles (destination, source, location)
      temporal via DATE/TIME entities or temporal adverbs, cause via markers (because/since/due to)
    - Optionally normalizes relations with embeddings
    """

    def __init__(self, use_normalizer: bool = True):
        self.normalizer = RelationNormalizer() if use_normalizer else None

    def _is_temporal(self, tok) -> bool:
        try:
            if tok.ent_type_ in {"DATE", "TIME"}:
                return True
        except Exception:
            pass
        # Simple lexical cues
        return _norm(tok.text) in {
            "today", "tomorrow", "yesterday", "tonight", "now",
            "morning", "evening", "afternoon", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday", "week", "month", "year"
        }

    def _span_text(self, tok) -> str:
        try:
            # Prefer noun chunk if available
            if hasattr(tok, "doc") and hasattr(tok.doc, "noun_chunks"):
                for ch in tok.doc.noun_chunks:
                    if ch.root.i == tok.i:
                        return ch.text
        except Exception:
            pass
        return tok.text

    def _collect_roles_for_predicate(self, head) -> Dict[str, str]:
        roles: Dict[str, str] = {}

        # Passive voice agent via 'agent' dep
        for ch in head.children:
            if ch.dep_ == "agent":
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        roles["agent"] = _canon_entity_text(self._span_text(gc))

        # Active voice subject / passive subject as patient
        subj = None
        for ch in head.children:
            if ch.dep_ in {"nsubj", "csubj"}:
                subj = ch
                roles["agent"] = _canon_entity_text(self._span_text(ch))
            elif ch.dep_ in {"nsubjpass"}:
                subj = ch
                roles["patient"] = _canon_entity_text(self._span_text(ch))

        # Direct and indirect objects
        for ch in head.children:
            if ch.dep_ in {"obj", "dobj"}:
                roles.setdefault("patient", _canon_entity_text(self._span_text(ch)))
            elif ch.dep_ == "iobj":
                roles["recipient"] = _canon_entity_text(self._span_text(ch))

        # Prepositional modifiers to roles
        for ch in head.children:
            if ch.dep_ == "prep":
                prep = _norm(ch.text)
                pobj = None
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        pobj = gc
                        break
                if not pobj:
                    continue
                pobj_text = _canon_entity_text(self._span_text(pobj))

                # Map common prepositions to roles
                if prep in {"to", "into", "onto"}:
                    roles["destination"] = pobj_text
                elif prep in {"from", "out", "out of"}:
                    roles["source"] = pobj_text
                elif prep in {"in", "at", "on"}:
                    # Temporal vs location
                    if self._is_temporal(pobj):
                        roles["temporal"] = pobj_text
                    else:
                        roles["location"] = pobj_text
                elif prep in {"with"}:
                    roles["instrument"] = pobj_text
                elif prep in {"for"}:
                    roles["beneficiary"] = pobj_text
                elif prep in {"because", "because of", "due to", "since", "as"}:
                    roles["cause"] = pobj_text
                else:
                    # Keep the most salient if looks like time
                    if self._is_temporal(pobj):
                        roles.setdefault("temporal", pobj_text)

        # Adverbial clause cause (because/since + S)
        for ch in head.children:
            if ch.dep_ == "advcl":
                # seek marker
                marker = None
                for gc in ch.children:
                    if gc.dep_ == "mark":
                        marker = _norm(gc.text)
                        break
                if marker in {"because", "since", "as"}:
                    roles["cause"] = _canon_entity_text(ch.text)

        return roles

    def doc_to_predications(self, doc, lang: str = "en") -> List[Predication]:
        preds: List[Predication] = []
        try:
            for sent in doc.sents:
                head = sent.root
                if head.pos_ not in {"VERB"}:
                    # Handle copula: X is Y (AUX with acomp/attr)
                    cop = None
                    for ch in head.children:
                        if ch.dep_ == "cop":
                            cop = ch
                            break
                    if not cop:
                        continue
                # verb head can be aux + main verb as child; prefer main verb
                if head.pos_ == "AUX":
                    main = None
                    for ch in head.children:
                        if ch.pos_ == "VERB":
                            main = ch
                            break
                    head = main or head

                if not head:
                    continue

                roles = self._collect_roles_for_predicate(head)

                # Copula predicate mapping
                pred_lemma = head.lemma_.lower() if head.lemma_ else head.text.lower()
                if any(c.dep_ == "cop" for c in head.children) or head.pos_ == "AUX":
                    pred_lemma = "be"

                preds.append(Predication(predicate=pred_lemma, roles=roles, lang=lang, sent_text=sent.text))
        except Exception as e:
            logger.debug(f"[SRL] doc_to_predications failed: {e}")
        return preds

    def predications_to_triples(self, preds: List[Predication]) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        for p in preds:
            # Choose subject/object from roles
            s = p.roles.get("agent") or p.roles.get("subject")
            o = p.roles.get("patient") or p.roles.get("object") or p.roles.get("destination") or p.roles.get("location")
            if not s and not o and p.roles.get("beneficiary"):
                # Edge case: give/offer with only beneficiary
                s = p.roles.get("agent")
                o = p.roles.get("beneficiary")
            if not s or not o:
                # Try simple copula: X is Y via subject and attr/adjective captured as patient
                continue

            # Preposition hint helpful for normalization
            prep_hint = None
            # Roughly infer from destination/source/location roles
            if p.roles.get("destination"):
                prep_hint = "to"
            elif p.roles.get("source"):
                prep_hint = "from"
            elif p.roles.get("location"):
                prep_hint = "in"

            rel = p.predicate
            if self.normalizer:
                rel = self.normalizer.normalize(p.predicate, p.roles, prep_hint)

            # Normalize a few role-specific relations
            if rel == "is" or rel == "be":
                rel = "is"

            triples.append((s, rel, o))
            # Encode temporal/cause as auxiliary triples if available
            if p.roles.get("temporal"):
                triples.append((o, "when", p.roles["temporal"]))
            if p.roles.get("cause"):
                triples.append((p.predicate, "because_of", p.roles["cause"]))
        return triples


__all__ = ["SRLExtractor", "Predication", "RelationNormalizer"]
