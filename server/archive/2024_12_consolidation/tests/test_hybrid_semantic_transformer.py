#!/usr/bin/env python3
"""
🧠 HYBRID SEMANTIC TRANSFORMER
Proof of concept: UD structure detection + semantic transformation rules
"""

import spacy
from typing import List, Tuple

class HybridSemanticTransformer:
    """
    Uses UD to detect syntactic structures, applies semantic transformation rules
    to produce PropBank-style semantic relations
    """

    def __init__(self, nlp):
        self.nlp = nlp

    def extract_semantic_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract semantic triples using hybrid UD + transformation approach"""
        doc = self.nlp(text)
        triples = []

        for sent in doc.sents:
            # Transform copula constructions
            triples.extend(self._transform_copula(sent))

            # Transform passive constructions
            triples.extend(self._transform_passive(sent))

            # Transform coordination
            triples.extend(self._transform_coordination(sent))

        return triples

    def _transform_copula(self, sent) -> List[Tuple[str, str, str]]:
        """Transform copula constructions into semantic relations"""
        triples = []
        root = sent.root

        # Detect copula: "X is Y" structure
        copula_child = None
        for child in root.children:
            if child.dep_ == "cop":
                copula_child = child
                break

        if not copula_child:
            return triples

        # Get subject and predicate
        subject = None
        predicate = root  # In copula, root is usually the predicate

        for child in root.children:
            if child.dep_ in ["nsubj", "nsubj:pass"]:
                subject = child
                break

        if not subject:
            return triples

        # Apply semantic transformation rules
        sent_text = sent.text.lower()

        if "name" in sent_text and subject.text.lower() == "name":
            # "My name is X" -> (you, has_name, X)
            possessor = self._resolve_possessor(subject, sent)
            name_value = self._get_name_value(sent)
            if possessor and name_value:
                triples.append((possessor, "has_name", name_value))

        elif "friend" in sent_text and predicate.text.lower() in ["friends", "friend"]:
            # "X and Y are friends" -> (X, friend_of, Y)
            subjects = self._get_coordinated_entities(subject, sent)
            for s1 in subjects:
                for s2 in subjects:
                    if s1 != s2:
                        triples.append((s1.lower(), "friend_of", s2.lower()))

        elif "color" in sent_text:
            # "My favorite color is X" -> (you, has_favorite_color, X)
            possessor = self._resolve_possessor(subject, sent)
            color = predicate.text.lower()
            if possessor and color:
                triples.append((possessor, "has_favorite_color", color))

        elif "son" in sent_text or "daughter" in sent_text:
            # "My son is named X" -> (you, has_child_named, X)
            possessor = self._resolve_possessor(subject, sent)
            child_name = self._get_name_from_passive(sent)
            if possessor and child_name:
                triples.append((possessor, "has_child_named", child_name))

        else:
            # Generic attribute relation
            subj_text = self._get_entity_text(subject)
            pred_text = self._get_entity_text(predicate)
            if subj_text and pred_text:
                triples.append((subj_text, "has_attribute", pred_text))

        return triples

    def _transform_passive(self, sent) -> List[Tuple[str, str, str]]:
        """Transform passive constructions into semantic relations"""
        triples = []

        for token in sent:
            if token.dep_ in ["nsubjpass", "nsubj:pass"]:
                # Found passive subject
                verb = token.head
                subject = self._get_entity_text(token)

                sent_text = sent.text.lower()

                if "born" in sent_text:
                    # "I was born in X" -> (you, born_in, X)
                    location_time = self._get_location_or_time(verb, sent)
                    if subject and location_time:
                        if any(char.isdigit() for char in location_time):
                            triples.append((subject, "born_in_year", location_time))
                        else:
                            triples.append((subject, "born_in", location_time))

        return triples

    def _transform_coordination(self, sent) -> List[Tuple[str, str, str]]:
        """Transform coordination into multiple relations"""
        triples = []

        # This is handled within copula transformation for "friends" case
        # Could extend for other coordinated relations

        return triples

    def _resolve_possessor(self, noun, sent):
        """Resolve possessive pronouns to semantic entities"""
        for child in noun.children:
            if child.dep_ == "poss":
                if child.text.lower() in ["my", "mine"]:
                    return "you"
                else:
                    return child.text.lower()

        # Look for possessive in broader context
        for token in sent:
            if token.text.lower() in ["my", "mine"] and token.head == noun:
                return "you"

        return None

    def _get_coordinated_entities(self, subject, sent):
        """Get all coordinated entities (X and Y)"""
        entities = [subject.text]

        # Look for conjunctions
        for child in subject.children:
            if child.dep_ == "conj":
                entities.append(child.text)

        return entities

    def _get_name_value(self, sent):
        """Extract name value from copula sentence"""
        for token in sent:
            if token.dep_ in ["attr", "oprd"] or (token.ent_type_ == "PERSON" and token.dep_ != "nsubj"):
                # Handle compound names
                name_parts = [token.text]
                for child in token.children:
                    if child.dep_ == "compound":
                        name_parts.insert(0, child.text)
                return " ".join(name_parts)
        return None

    def _get_name_from_passive(self, sent):
        """Extract name from passive construction like 'is named X'"""
        for token in sent:
            if token.dep_ in ["oprd", "attr"]:
                return token.text.lower()
        return None

    def _get_location_or_time(self, verb, sent):
        """Extract location or time from prepositional phrase"""
        for child in verb.children:
            if child.dep_ in ["prep", "agent"]:
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return grandchild.text
            elif child.dep_ in ["obl"]:
                return child.text
        return None

    def _get_entity_text(self, token):
        """Get clean entity text, handling pronouns"""
        if token.text.lower() in ["i", "me"]:
            return "you"
        return token.text.lower()

def test_hybrid_transformer():
    """Test the hybrid approach on failed cases"""
    nlp = spacy.load("en_core_web_trf")
    transformer = HybridSemanticTransformer(nlp)

    # Test cases that failed with pure SRL
    failed_cases = [
        "My name is Alex Thompson",
        "My dog's name is Potola",
        "Sarah and John are friends",
        "My favorite color is blue",
        "I was born in 1995",
        "My son is named Jake",
    ]

    print("🧠 HYBRID SEMANTIC TRANSFORMER TEST")
    print("=" * 60)

    for i, text in enumerate(failed_cases, 1):
        print(f"\n{i}. '{text}'")
        print("-" * 40)

        triples = transformer.extract_semantic_triples(text)

        if triples:
            print("✅ SEMANTIC TRIPLES:")
            for triple in triples:
                print(f"   {triple}")
        else:
            print("❌ No extractions")

if __name__ == "__main__":
    test_hybrid_transformer()