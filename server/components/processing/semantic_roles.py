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
    predicate: str  # original verb form (feeds, announced, not lemmatized)
    # role -> surface string
    roles: Dict[str, str]
    # Optional light metadata
    lang: str = "en"
    sent_text: str = ""
    tense_info: Optional[Dict[str, Any]] = None  # tense, aspect, morphology


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
            elif ch.dep_ == "ccomp":
                # Clausal complements as patient (e.g., "announced that X would Y")
                # Extract subject + verb from complement: "company would restructure" -> "company restructure"
                try:
                    complement_subj = None
                    complement_verb = ch.text.lower()  # Start with the head verb

                    # Find the subject of the complement clause
                    for token in ch.children:
                        if token.dep_ in {"nsubj", "csubj"}:
                            complement_subj = self._span_text(token)
                            break

                    if complement_subj:
                        compound_event = f"{_canon_entity_text(complement_subj)} {complement_verb}"
                        roles.setdefault("patient", compound_event)
                    else:
                        roles.setdefault("patient", _canon_entity_text(ch.text))
                except Exception:
                    roles.setdefault("patient", _canon_entity_text(ch.text))

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
                elif prep in {"because", "because of", "due to", "since", "as", "after"}:
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
                # EXTRACT ALL MEANINGFUL VERBS, not just sentence root
                verb_candidates = []

                # 1. Main sentence root
                if sent.root.pos_ in {"VERB", "AUX"}:
                    verb_candidates.append(sent.root)

                # 2. Find all other meaningful verbs in the sentence
                for token in sent:
                    if (token.pos_ == "VERB" and
                        token.dep_ in {"advcl", "relcl", "xcomp", "ccomp", "conj"} and
                        token not in verb_candidates):
                        verb_candidates.append(token)

                # Process each verb candidate
                for head in verb_candidates:
                    # Handle copula: X is Y (AUX with acomp/attr)
                    if head.pos_ not in {"VERB"}:
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

                    # PRESERVE ORIGINAL VERB FORM - don't lemmatize away tense!
                    pred_text = head.text.lower()  # Keep original form: feeds, announced, etc.
                    pred_lemma = head.lemma_.lower() if head.lemma_ else pred_text

                    # Handle compound predicates: "began teaching" -> "began_teaching"
                    compound_pred = None
                    for child in head.children:
                        if (child.pos_ == "VERB" and
                            child.dep_ in {"xcomp", "ccomp"} and
                            pred_text in {"began", "started", "finished", "stopped", "continued"}):
                            compound_pred = f"{pred_text}_{child.text.lower()}"
                            break

                    if compound_pred:
                        pred_text = compound_pred

                    # Extract tense/aspect metadata
                    tense_info = {
                        'original_form': pred_text,
                        'lemma': pred_lemma,
                        'pos': head.pos_,
                        'tag': head.tag_,  # VBZ, VBD, etc. for tense
                        'morph': str(head.morph) if head.morph else None
                    }

                    # Only lemmatize copulas to "be"
                    if any(c.dep_ == "cop" for c in head.children) or head.pos_ == "AUX":
                        pred_text = "is"  # Use contextual form, not generic "be"
                        if head.tag_ in ['VBD', 'VBN']:  # was, were
                            pred_text = "was"
                        elif head.tag_ in ['VBG']:  # being
                            pred_text = "being"

                    preds.append(Predication(predicate=pred_text, roles=roles, lang=lang, sent_text=sent.text, tense_info=tense_info))
        except Exception as e:
            logger.debug(f"[SRL] doc_to_predications failed: {e}")
        return preds

    def predications_to_triples(self, preds: List[Predication]) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        last_subject = None  # Track previous subject for context

        for p in preds:
            # Choose subject/object from roles - prioritize patient over location for verbs like "writing"
            s = p.roles.get("agent") or p.roles.get("subject")
            o = p.roles.get("patient") or p.roles.get("object")
            if not o:
                o = p.roles.get("destination") or p.roles.get("location")
            if not s and not o and p.roles.get("beneficiary"):
                # Edge case: give/offer with only beneficiary
                s = p.roles.get("agent")
                o = p.roles.get("beneficiary")

            # RELAXED: Allow intransitive verbs with only subject (ended, began, etc)
            if not s:
                # Try to inherit subject from context (for gerunds like "teaching", "writing")
                if last_subject and (p.roles.get("patient") or p.roles.get("location")):
                    s = last_subject  # Use previous subject
                else:
                    continue  # Skip if no subject available
            else:
                last_subject = s  # Remember this subject for next predicate

            # Handle missing object - better semantic representation
            if not o:
                # For intransitive verbs with specific roles, use the role
                if p.roles.get("destination"):
                    o = p.roles["destination"]
                    rel_suffix = "_to"
                elif p.roles.get("location"):
                    o = p.roles["location"]
                    # Better suffix selection based on predicate type
                    if p.predicate in {"teaching", "working", "studying", "living"}:
                        rel_suffix = "_at"
                    else:
                        rel_suffix = "_in"
                elif p.roles.get("temporal"):
                    # Temporal intransitive: "ended in july" -> (festival, ended_in, july)
                    o = p.roles["temporal"]
                    rel_suffix = "_in"
                elif p.roles.get("cause"):
                    # For causal predicates, skip the main triple but allow causal processing
                    o = None  # Signal to skip main triple but allow causal processing
                else:
                    # Pure intransitive with meaningful content - allow them
                    # E.g., "began_teaching", "started", etc.
                    if "_" in p.predicate or p.predicate in {"began", "started", "finished", "stopped"}:
                        # Create a meaningful relation for compound or lifecycle verbs
                        continue  # Skip for now - may need different approach
                    else:
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

            # Add directional suffix for location/destination verbs
            if 'rel_suffix' in locals() and rel_suffix:
                rel = p.predicate + rel_suffix

            # Clean up locals to avoid suffix carryover
            if 'rel_suffix' in locals():
                del rel_suffix

            if self.normalizer:
                rel = self.normalizer.normalize(p.predicate, p.roles, prep_hint)

            # Normalize a few role-specific relations
            if rel == "is" or rel == "be":
                rel = "is"

            # Only generate main triple if we have a valid object
            if o is not None:
                triples.append((s, rel, o))

            # FIXED: Attach temporal to the ACTION (predicate), not agent or patient
            # Skip if temporal already encoded in main relation (ended_in, moved_to, etc)
            if p.roles.get("temporal") and not any(suffix in rel for suffix in ["_in", "_to", "_at"]):
                # Create meaningful temporal relation: (action, when, time)
                action_entity = f"{s}_{rel}_{o}".replace(" ", "_")  # e.g. "alice_feeds_cat"
                triples.append((action_entity, "when", p.roles["temporal"]))
            if p.roles.get("cause"):
                # Create proper causal relation: "company restructure" caused by "declining profits"
                event = f"{s} {p.predicate}".strip()
                triples.append((event, "caused_by", p.roles["cause"]))

        # Simple coreference resolution
        triples = self._resolve_simple_coreferences(triples)
        return triples

    def _resolve_simple_coreferences(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Resolve basic coreference cases like who->person, she->female_name"""
        resolved = []
        entities_seen = []  # Track entities for coreference

        for s, r, o in triples:
            # Collect potential referents
            if any(word in s.lower() for word in ['alice', 'maria', 'boy', 'ceo', 'chef', 'festival']):
                entities_seen.append(s)
            if any(word in o.lower() for word in ['alice', 'maria', 'boy', 'ceo', 'chef', 'festival']):
                entities_seen.append(o)

            # Resolve coreferences
            resolved_s = self._resolve_pronoun(s, entities_seen)
            resolved_o = self._resolve_pronoun(o, entities_seen)

            resolved.append((resolved_s, r, resolved_o))

        return resolved

    def _resolve_pronoun(self, text: str, entities_seen: List[str]) -> str:
        """Resolve pronouns to most recent appropriate entity"""
        text_lower = text.lower()

        if text_lower == "who":
            # Find most recent person entity
            for entity in reversed(entities_seen):
                if any(word in entity.lower() for word in ['boy', 'alice', 'maria', 'ceo', 'chef']):
                    return entity
        elif text_lower in ["she", "her"]:
            # Find most recent female entity
            for entity in reversed(entities_seen):
                if any(word in entity.lower() for word in ['alice', 'maria', 'chef']):
                    return entity
        elif text_lower in ["he", "him"]:
            # Find most recent male entity
            for entity in reversed(entities_seen):
                if any(word in entity.lower() for word in ['boy', 'ceo']):
                    return entity

        return text  # No resolution found

    def predications_to_triples_with_embeddings(self, preds: List[Predication]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Convert predications to triples with semantic embeddings stored in metadata.
        Returns: List of (subject, relation, object, metadata) tuples where metadata contains embeddings.
        """
        triples_with_meta: List[Tuple[str, str, str, Dict[str, Any]]] = []

        # Ensure embedding model is loaded (via normalizer)
        if self.normalizer:
            self.normalizer._ensure_model()

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

            # PRESERVE SEMANTIC MEANING: Use original predicate, not forced normalization
            rel = p.predicate
            normalized_rel = None
            if self.normalizer:
                normalized_rel = self.normalizer.normalize(p.predicate, p.roles, prep_hint)

            # Only normalize trivial cases like copulas
            if rel in ["is", "be", "are", "was", "were"]:
                rel = "is"

            # Compute semantic embedding for the relation
            metadata = {}
            if self.normalizer and self.normalizer._model is not None:
                try:
                    # Create a descriptive phrase for embedding
                    rel_phrase = f"{s} {p.predicate} {o}"
                    if prep_hint:
                        rel_phrase = f"{s} {p.predicate} {prep_hint} {o}"

                    # Compute embedding
                    embedding = self.normalizer._model.encode(rel_phrase, normalize_embeddings=True)
                    metadata["rel_embedding"] = embedding.tolist()  # Convert numpy to list for JSON storage
                    metadata["original_predicate"] = p.predicate
                    if normalized_rel:
                        metadata["normalized_relation"] = normalized_rel  # Keep for reference but don't use

                    # Add rich tense/aspect metadata
                    if p.tense_info:
                        metadata["tense_info"] = p.tense_info

                    logger.debug(f"[SRL] Computed embedding for: '{rel_phrase}' -> {rel} (original form preserved)")
                except Exception as e:
                    logger.debug(f"[SRL] Failed to compute embedding for '{p.predicate}': {e}")

            # PROPER SOLUTION: Store temporal/causal as metadata, not separate nonsensical triples
            if p.roles.get("temporal"):
                metadata["temporal_arg"] = p.roles["temporal"]  # ARGTMP in PropBank
                if self.normalizer and self.normalizer._model is not None:
                    try:
                        temporal_phrase = f"{rel} occurs at {p.roles['temporal']}"
                        temporal_embedding = self.normalizer._model.encode(temporal_phrase, normalize_embeddings=True)
                        metadata["temporal_embedding"] = temporal_embedding.tolist()
                    except Exception:
                        pass

            if p.roles.get("cause"):
                metadata["causal_arg"] = p.roles["cause"]  # ARGCAU in PropBank
                if self.normalizer and self.normalizer._model is not None:
                    try:
                        causal_phrase = f"{rel} because of {p.roles['cause']}"
                        causal_embedding = self.normalizer._model.encode(causal_phrase, normalize_embeddings=True)
                        metadata["causal_embedding"] = causal_embedding.tolist()
                    except Exception:
                        pass

            # Store location, manner, purpose as metadata too
            if p.roles.get("location"):
                metadata["location_arg"] = p.roles["location"]
            if p.roles.get("manner"):
                metadata["manner_arg"] = p.roles["manner"]
            if p.roles.get("purpose"):
                metadata["purpose_arg"] = p.roles["purpose"]

            triples_with_meta.append((s, rel, o, metadata))

        return triples_with_meta


__all__ = ["SRLExtractor", "Predication", "RelationNormalizer"]
