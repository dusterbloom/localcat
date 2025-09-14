#!/usr/bin/env python3
"""
🔥 ENHANCED DEPENDENCY SEMANTIC EXTRACTOR
Combines the sophisticated patterns from your examples with semantic quality
"""

import spacy
from spacy.matcher import DependencyMatcher
from spacy.tokens import Token
from typing import List, Tuple

# --- CONFIG ---
PERSPECTIVE = "you"  # map 1st-person to "you" for assistant-facing triples

# Possessive determiner mapping to owners
DET_POSSESSOR = {
    "my": PERSPECTIVE, "your": "you", "his": "he", "her": "she",
    "our": "we", "their": "they"
}

# Simple kinship lexicon → relation (lowercase lemmas)
KIN_REL = {
    "son": "has_son", "daughter": "has_daughter",
    "child": "has_child", "kid": "has_child",
    "wife": "has_spouse", "husband": "has_spouse",
    "brother": "has_brother", "sister": "has_sister",
    "mother": "has_mother", "father": "has_father",
    "parent": "has_parent", "partner": "has_partner"
}

class EnhancedDependencyExtractor:
    """Enhanced dependency matcher with sophisticated semantic patterns"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.matcher = DependencyMatcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        """Setup comprehensive dependency patterns"""

        # Core SVO pattern
        svo = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj","csubj"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["obj","iobj","dobj"]}}}
        ]

        # Verb + oblique with case
        verb_obl_case = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj","csubj"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obl",
             "RIGHT_ATTRS": {"DEP": {"REGEX": "^obl|^nmod"}}},
            {"LEFT_ID": "obl", "REL_OP": ">", "RIGHT_ID": "case",
             "RIGHT_ATTRS": {"DEP": "case", "POS": "ADP"}}
        ]

        # Copula pattern
        copula = [
            {"RIGHT_ID": "attr", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN","PROPN","ADJ"]}}},
            {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "cop", "RIGHT_ATTRS": {"DEP": "cop"}},
            {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj"]}}}
        ]

        # Passive with agent
        passive_agent = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": "nsubj:pass"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "agent",
             "RIGHT_ATTRS": {"DEP": "obl:agent"}}
        ]

        # Ditransitive
        ditrans = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj","csubj"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "theme",
             "RIGHT_ATTRS": {"DEP": {"IN": ["obj"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "recip",
             "RIGHT_ATTRS": {"DEP": {"IN": ["iobj","obl"]}}}
        ]

        # Apposition for friends/relationships
        appos = [
            {"RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["PROPN","NOUN"]}}},
            {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "app",
             "RIGHT_ATTRS": {"DEP": "appos", "POS": {"IN": ["NOUN","PROPN"]}}}
        ]

        self.matcher.add("SVO", [svo])
        self.matcher.add("VERB_OBL_CASE", [verb_obl_case])
        self.matcher.add("COPULA", [copula])
        self.matcher.add("PASSIVE_AGENT", [passive_agent])
        self.matcher.add("DITRANS", [ditrans])
        self.matcher.add("APPOS", [appos])

    def owner_from_possessive(self, det_tok: Token) -> str:
        """Return canonical owner string for a possessive determiner/pronoun."""
        txt = det_tok.text.lower()
        return DET_POSSESSOR.get(txt, det_tok.lemma_.lower())

    def collect_name_span(self, head: Token) -> str:
        """Collect full name including flat:name / compound parts."""
        toks = [head]
        for ch in head.children:
            if ch.dep_ in ("flat", "flat:name", "compound", "fixed") or (ch.dep_ == "amod" and ch.pos_ == "ADJ"):
                toks.append(ch)

        span = sorted(set(toks + [t for t in head.subtree if t.pos_ in ("PROPN","ADJ","NOUN") and t.dep_.startswith(("flat","compound","fixed"))]), key=lambda t: t.i)
        text = " ".join(t.text for t in span) if span else head.text
        return text.replace(" - ", "-")

    def emit_name_noun_copula(self, doc):
        """Pattern: X's name is Y"""
        triples = []
        for sent in doc.sents:
            for tok in sent:
                if tok.lemma_.lower() == "name" and tok.pos_ == "NOUN":
                    # find possessor
                    poss = [c for c in tok.children if c.dep_ in ("det:poss","nmod:poss")]
                    if not poss:
                        continue
                    owner = self.owner_from_possessive(poss[0])

                    # check for copula
                    has_cop = any(c.dep_ == "cop" for c in tok.children)
                    if not has_cop:
                        continue

                    # find name complement
                    cand = None
                    appos = [c for c in tok.children if c.dep_ == "appos" and c.pos_ in ("PROPN","NOUN")]
                    if appos:
                        cand = appos[0]
                    else:
                        right_props = [t for t in sent if t.i > tok.i and t.pos_ == "PROPN"]
                        cand = right_props[0] if right_props else None

                    if cand is None:
                        continue

                    full = self.collect_name_span(cand)
                    triples.append((owner, "has_name", full))
        return triples

    def emit_is_named(self, doc):
        """Pattern: X is named Y / X was named Y"""
        triples = []
        for sent in doc.sents:
            for v in sent:
                if v.lemma_.lower() == "name" and v.pos_ in ("VERB","AUX"):
                    # must have 'be' aux
                    if not any(c.dep_.startswith("aux") and c.lemma_.lower() in ("be","become","get") for c in v.children):
                        be_heads = [h for h in [v.head] if h and h.lemma_.lower()=="be"]
                        if not be_heads:
                            continue

                    # subject X
                    subs = [c for c in v.children if c.dep_.startswith("nsubj")]
                    if not subs:
                        subs = [c for c in v.head.children if c.dep_.startswith("nsubj")] if v.head else []
                    if not subs:
                        continue

                    X = subs[0]

                    # find Y (the name)
                    Y = None
                    for rel in ("obj","xcomp","attr","obl","ccomp"):
                        cand = [c for c in v.children if c.dep_ == rel and c.pos_ in ("PROPN","NOUN","ADJ")]
                        if cand:
                            Y = cand[0]
                            break

                    if Y is None:
                        props = [t for t in sent if t.i > v.i and t.pos_=="PROPN"]
                        Y = props[0] if props else None

                    if Y is None:
                        continue

                    name_text = self.collect_name_span(Y)
                    x_text = X.text.lower()
                    if x_text in ["i", "me"]:
                        x_text = "you"
                    triples.append((x_text, "has_name", name_text))
        return triples

    def emit_possessive_kin(self, doc):
        """Pattern: possessive kinship: 'my son', 'my daughter'"""
        triples = []
        for sent in doc.sents:
            for n in sent:
                if n.pos_ == "NOUN":
                    poss = [c for c in n.children if c.dep_ in ("det:poss","nmod:poss")]
                    if not poss:
                        continue

                    rel = KIN_REL.get(n.lemma_.lower())
                    if not rel:
                        continue

                    owner = self.owner_from_possessive(poss[0])
                    triples.append((owner, rel, n.text))
        return triples

    def emit_coordination_friends(self, doc):
        """Handle 'Sarah and John are friends' pattern"""
        triples = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if "friend" not in sent_text:
                continue

            # Look for coordination pattern with "friends"
            for tok in sent:
                if tok.lemma_.lower() == "friend" and tok.pos_ == "NOUN":
                    # Find subjects connected to this predicate
                    subjects = []

                    # Look for copula construction
                    head_verb = None
                    for potential_head in sent:
                        if any(c.dep_ == "cop" and c.head == potential_head for c in sent):
                            head_verb = potential_head
                            break

                    if head_verb:
                        # Find all coordinated subjects
                        for child in head_verb.children:
                            if child.dep_ == "nsubj":
                                subjects.append(child.text.lower())
                                # Add coordinated entities
                                for conj_child in child.children:
                                    if conj_child.dep_ == "conj":
                                        subjects.append(conj_child.text.lower())

                    # Generate friend relations
                    for i, subj1 in enumerate(subjects):
                        for subj2 in subjects[i+1:]:
                            triples.append((subj1, "friend_of", subj2))
                            triples.append((subj2, "friend_of", subj1))

        return triples

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract semantic triples using enhanced patterns"""
        doc = self.nlp(text)
        triples = []

        # Add specialized patterns
        triples.extend(self.emit_name_noun_copula(doc))
        triples.extend(self.emit_is_named(doc))
        triples.extend(self.emit_possessive_kin(doc))
        triples.extend(self.emit_coordination_friends(doc))

        # Add dependency matcher patterns
        for match_id, token_ids in self.matcher(doc):
            name = self.nlp.vocab.strings[match_id]
            tokens_by_role = {doc[t].dep_: doc[t] for t in token_ids}
            tokens = [doc[t] for t in token_ids]

            if name == "SVO":
                verb = next((t for t in tokens if t.pos_ == "VERB"), None)
                subj = tokens_by_role.get("nsubj") or tokens_by_role.get("csubj")
                obj = tokens_by_role.get("obj") or tokens_by_role.get("iobj") or tokens_by_role.get("dobj")

                if verb and subj and obj:
                    subj_text = self._normalize_pronoun(subj.text)
                    obj_text = obj.text.lower()
                    triples.append((subj_text, verb.lemma_, obj_text))

            elif name == "VERB_OBL_CASE":
                verb = next((t for t in tokens if t.pos_ == "VERB"), None)
                subj = tokens_by_role.get("nsubj") or tokens_by_role.get("csubj")
                obl = next((t for t in tokens if t.dep_.startswith(("obl", "nmod"))), None)
                prep = next((t for t in tokens if t.dep_ == "case"), None)

                if verb and subj and obl and prep:
                    subj_text = self._normalize_pronoun(subj.text)
                    relation = f"{verb.lemma_}_{prep.lemma_}"
                    triples.append((subj_text, relation, obl.text.lower()))

            elif name == "COPULA":
                attr = next((t for t in tokens if t.pos_ in ("NOUN", "PROPN", "ADJ")), None)
                subj = next((t for t in tokens if t.dep_ == "nsubj"), None)

                if attr and subj:
                    subj_text = self._normalize_pronoun(subj.text)
                    attr_text = self._get_compound_text(attr)

                    # Special handling for colors and attributes
                    sent_text = doc.text.lower()
                    if "color" in sent_text:
                        triples.append((subj_text, "has_favorite_color", attr_text))
                    else:
                        triples.append((subj_text, "has_attribute", attr_text))

        return list(set(triples))  # Remove duplicates

    def _normalize_pronoun(self, text: str) -> str:
        """Convert pronouns to consistent forms"""
        text_lower = text.lower()
        if text_lower in ["i", "me"]:
            return "you"
        return text_lower

    def _get_compound_text(self, token) -> str:
        """Get full text including compound modifiers"""
        parts = []

        for child in token.children:
            if child.dep_ == "compound" and child.i < token.i:
                parts.append(child.text)

        parts.append(token.text)

        for child in token.children:
            if child.dep_ == "compound" and child.i > token.i:
                parts.append(child.text)

        return " ".join(parts)

def test_enhanced_approach():
    """Test the enhanced dependency approach"""
    extractor = EnhancedDependencyExtractor()

    # Test cases that failed with pure SRL
    failed_cases = [
        "My name is Alex Thompson",
        "My dog's name is Potola",
        "Sarah and John are friends",
        "My favorite color is blue",
        "I was born in 1995",
        "My son is named Jake",
    ]

    # Test cases that worked with SRL
    working_cases = [
        "Alice feeds the cat in the morning",
        "I live in Seattle",
        "I work at Microsoft",
    ]

    print("🔥 ENHANCED DEPENDENCY SEMANTIC EXTRACTION TEST")
    print("=" * 70)

    print("\n🚨 PREVIOUSLY FAILED CASES:")
    for i, text in enumerate(failed_cases, 1):
        print(f"\n{i}. '{text}'")
        print("-" * 50)

        triples = extractor.extract_triples(text)

        if triples:
            print("✅ SEMANTIC TRIPLES:")
            for triple in triples:
                print(f"   {triple}")
        else:
            print("❌ No extractions")

    print("\n\n✅ PREVIOUSLY WORKING CASES:")
    for i, text in enumerate(working_cases, 1):
        print(f"\n{i}. '{text}'")
        print("-" * 50)

        triples = extractor.extract_triples(text)

        if triples:
            print("✅ SEMANTIC TRIPLES:")
            for triple in triples:
                print(f"   {triple}")
        else:
            print("❌ No extractions")

if __name__ == "__main__":
    test_enhanced_approach()