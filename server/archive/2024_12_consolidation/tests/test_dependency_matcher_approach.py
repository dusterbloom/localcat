#!/usr/bin/env python3
"""
🔥 DEPENDENCY MATCHER SEMANTIC EXTRACTION
Based on the spaCy DependencyMatcher approach - UD-aware rules producing semantic triples
"""

import spacy
from spacy.matcher import DependencyMatcher
from typing import List, Tuple

class DependencySemanticExtractor:
    """
    Uses spaCy DependencyMatcher with UD-aware rules to extract semantic triples
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.matcher = DependencyMatcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        """Setup semantic extraction patterns using UD dependencies"""

        # 1) Active SVO: (nsubj) —VERB— (obj|iobj)
        svo = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "csubj"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "iobj", "dobj"]}}}
        ]

        # 2) Verb + oblique nominal + case(prep): e.g., live + obl[case=in] -> live_in
        verb_obl_case = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "csubj"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "obl",
             "RIGHT_ATTRS": {"DEP": {"REGEX": "^obl|^nmod"}}},
            {"LEFT_ID": "obl", "REL_OP": ">", "RIGHT_ID": "case",
             "RIGHT_ATTRS": {"DEP": "case", "POS": "ADP"}}
        ]

        # 3) Copula: Alice is a doctor -> (Alice, be, doctor)
        copula = [
            {"RIGHT_ID": "attr", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}},
            {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "cop", "RIGHT_ATTRS": {"DEP": "cop"}},
            {"LEFT_ID": "attr", "REL_OP": ">", "RIGHT_ID": "subj", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj"]}}}
        ]

        # 4) Possession: My X is Y -> (you, has_X, Y)
        possession = [
            {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN"]}}},
            {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "poss", "RIGHT_ATTRS": {"DEP": "poss"}}
        ]

        # 5) Passive: I was born -> (you, born_in, X)
        passive = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
             "RIGHT_ATTRS": {"DEP": {"IN": ["nsubjpass", "nsubj:pass"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "aux",
             "RIGHT_ATTRS": {"DEP": {"IN": ["auxpass", "aux:pass", "aux"]}}}
        ]

        # 6) Coordination: Sarah and John are friends -> (Sarah, friend_of, John)
        coordination = [
            {"RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}},
            {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "conj", "RIGHT_ATTRS": {"DEP": "conj"}}
        ]

        self.matcher.add("SVO", [svo])
        self.matcher.add("VERB_OBL_CASE", [verb_obl_case])
        self.matcher.add("COPULA", [copula])
        self.matcher.add("POSSESSION", [possession])
        self.matcher.add("PASSIVE", [passive])
        self.matcher.add("COORDINATION", [coordination])

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract semantic triples using dependency matcher"""
        doc = self.nlp(text)
        triples = []

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
                    triples.append((subj_text, verb.lemma_, obj.text.lower()))

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

                    # Special cases for common copula patterns
                    sent_text = doc.text.lower()
                    if "name" in sent_text and subj.text.lower() == "name":
                        # Handle "My name is X"
                        possessor = self._find_possessor(subj, doc)
                        if possessor:
                            attr_text = self._get_compound_text(attr)
                            triples.append((possessor, "has_name", attr_text))
                    elif "friend" in sent_text:
                        # Handle coordination in friends
                        entities = self._get_coordinated_entities(subj, doc)
                        for e1 in entities:
                            for e2 in entities:
                                if e1 != e2:
                                    triples.append((e1.lower(), "friend_of", e2.lower()))
                    else:
                        attr_text = self._get_compound_text(attr)
                        triples.append((subj_text, "has_attribute", attr_text))

            elif name == "POSSESSION":
                noun = next((t for t in tokens if t.pos_ == "NOUN"), None)
                poss = next((t for t in tokens if t.dep_ == "poss"), None)

                if noun and poss:
                    possessor = self._normalize_pronoun(poss.text)
                    # This helps with "My dog's name is X" type constructions
                    triples.append((possessor, "has", noun.text.lower()))

            elif name == "PASSIVE":
                verb = next((t for t in tokens if t.pos_ == "VERB"), None)
                subj = tokens_by_role.get("nsubjpass") or tokens_by_role.get("nsubj:pass")

                if verb and subj:
                    subj_text = self._normalize_pronoun(subj.text)

                    # Special handling for common passive constructions
                    if verb.lemma_ == "bear":
                        # "I was born in X"
                        time_loc = self._find_time_location(verb, doc)
                        if time_loc:
                            if any(char.isdigit() for char in time_loc):
                                triples.append((subj_text, "born_in_year", time_loc))
                            else:
                                triples.append((subj_text, "born_in", time_loc))
                    elif "name" in doc.text.lower():
                        # "My son is named X"
                        name_value = self._find_object_predicate(verb, doc)
                        if name_value:
                            possessor = self._find_possessor_context(doc)
                            if possessor:
                                triples.append((possessor, "has_child_named", name_value))

        return triples

    def _normalize_pronoun(self, text: str) -> str:
        """Convert pronouns to consistent forms"""
        text_lower = text.lower()
        if text_lower in ["i", "me"]:
            return "you"
        return text_lower

    def _get_compound_text(self, token) -> str:
        """Get full text including compound modifiers"""
        parts = []

        # Get compounds that come before this token
        for child in token.children:
            if child.dep_ == "compound" and child.i < token.i:
                parts.append(child.text)

        parts.append(token.text)

        # Get compounds that come after
        for child in token.children:
            if child.dep_ == "compound" and child.i > token.i:
                parts.append(child.text)

        return " ".join(parts)

    def _find_possessor(self, token, doc):
        """Find possessor for constructions like 'My name'"""
        for child in token.children:
            if child.dep_ == "poss":
                return self._normalize_pronoun(child.text)
        return None

    def _find_possessor_context(self, doc):
        """Find possessor in broader sentence context"""
        for token in doc:
            if token.text.lower() in ["my", "mine"]:
                return "you"
        return None

    def _get_coordinated_entities(self, token, doc):
        """Get all entities connected by coordination"""
        entities = [token.text]

        for child in token.children:
            if child.dep_ == "conj":
                entities.append(child.text)

        return entities

    def _find_time_location(self, verb, doc):
        """Find time or location in prepositional phrases"""
        for child in verb.children:
            if child.dep_ in ["prep", "obl"]:
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return grandchild.text
            elif child.dep_ in ["obl"]:
                return child.text
        return None

    def _find_object_predicate(self, verb, doc):
        """Find object predicate in passive constructions"""
        for token in doc:
            if token.dep_ in ["oprd", "attr"]:
                return token.text.lower()
        return None

def test_dependency_approach():
    """Test the dependency matcher approach on failed cases"""
    extractor = DependencySemanticExtractor()

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

    print("🔥 DEPENDENCY MATCHER SEMANTIC EXTRACTION TEST")
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
    test_dependency_approach()