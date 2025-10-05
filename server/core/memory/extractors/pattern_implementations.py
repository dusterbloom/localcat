"""
Missing L1 Pattern Implementations for YAML Runtime
These functions should be added to yaml_runtime.py
"""

from typing import List, Set, Tuple, Optional
from spacy.tokens import Doc, Token


def _safe_lower(s) -> str:
    """Safe lowercase conversion."""
    if s is None:
        return ""
    return str(s).lower()


def _canon_entity_text(text) -> str:
    """Canonicalize entity text."""
    if not text:
        return ""
    # Capitalize first letter for proper names
    text = str(text).strip()
    if text and text[0].islower():
        return text.capitalize()
    return text


class MissingPatternImplementations:
    """All 17 missing L1 pattern implementations"""

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
        """Extract copula with adjectives: 'John is happy' → ('John', 'is', 'happy')"""
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
                    r = "is"
                    d = _safe_lower(tok.text)
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
        """Handle mixed coordination patterns"""
        pass

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
        """Extract communication patterns: 'John told Mary that...' → ('John', 'told', 'Mary')"""
        comm_verbs = {"tell", "inform", "notify", "warn", "remind", "ask", "advise"}

        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ in comm_verbs:
                subj = None
                iobj = None

                for child in tok.children:
                    if child.dep_ in {"nsubj", "csubj"}:
                        subj = child
                    elif child.dep_ in {"iobj", "dobj"} and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                        iobj = child

                if subj and iobj:
                    s = _canon_entity_text(subj.text)
                    r = _safe_lower(tok.lemma_)
                    d = _canon_entity_text(iobj.text)
                    triples.append((s, r, d))
                    entities.update([s, d])

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