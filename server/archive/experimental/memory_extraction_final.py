#!/usr/bin/env python3
"""
USGS-inspired systematic extraction using all dependency patterns
Clean, non-hacky approach based on Grammar-to-Graph methodology
"""

import spacy
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass

@dataclass
class DependencyTriple:
    subject: str
    predicate: str  
    object: str
    dep_type: str

class SystematicExtractor:
    """
    Extract triples systematically using dependency grammar
    Based on USGS Grammar-to-Graph approach
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        self.user_pronouns = {"i", "me", "my", "mine", "myself"}
        
        # Map dependency relations to extraction handlers
        self.dep_handlers = {
            # Core arguments
            "nsubj": self._handle_subject,
            "nsubjpass": self._handle_subject,
            "dobj": self._handle_object,
            "obj": self._handle_object,
            "iobj": self._handle_indirect_object,
            "attr": self._handle_attribute,
            
            # Modifiers
            "amod": self._handle_adjectival_mod,
            "advmod": self._handle_adverbial_mod,
            "nummod": self._handle_numeric_mod,
            "nmod": self._handle_nominal_mod,
            
            # Relations
            "poss": self._handle_possessive,
            "compound": self._handle_compound,
            "prep": self._handle_preposition,
            "pobj": self._handle_prep_object,
            "conj": self._handle_conjunction,
            "appos": self._handle_apposition,
            
            # Clauses
            "acl": self._handle_adnominal_clause,
            "advcl": self._handle_adverbial_clause,
            "ccomp": self._handle_clausal_comp,
            "xcomp": self._handle_open_clause,
            
            # Function words (usually don't generate triples directly)
            "aux": None,
            "auxpass": None,
            "cop": None,
            "det": None,
            "case": None,
            "mark": None,
            "cc": None,
            "neg": None,
        }
    
    def extract(self, text: str) -> Tuple[List[DependencyTriple], Set[str]]:
        """Extract all triples from text"""
        doc = self.nlp(text)
        triples = []
        entities = set()
        
        # Stage 1: Collect all entities
        entity_map = self._extract_entities(doc)
        entities.update(entity_map.values())
        
        # Stage 2: Process each token's dependencies
        for token in doc:
            if token.dep_ in self.dep_handlers:
                handler = self.dep_handlers[token.dep_]
                if handler:
                    token_triples = handler(token, entity_map, doc)
                    triples.extend(token_triples)
        
        # Convert to simple format
        simple_triples = []
        for t in triples:
            simple_triples.append((t.subject, t.predicate, t.object))
        
        return simple_triples, entities
    
    def _extract_entities(self, doc) -> Dict[int, str]:
        """Extract all entities from document"""
        entities = {}
        
        # Named entities
        for ent in doc.ents:
            normalized = self._normalize(ent.text)
            for token in ent:
                entities[token.i] = normalized
        
        # Noun chunks  
        for chunk in doc.noun_chunks:
            normalized = self._normalize(chunk.text)
            entities[chunk.root.i] = normalized
        
        # Individual tokens
        for token in doc:
            if token.i not in entities:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    normalized = self._normalize(token.text)
                    # Canonicalize pronouns
                    if normalized in self.user_pronouns:
                        normalized = "you"
                    entities[token.i] = normalized
        
        return entities
    
    def _normalize(self, text: str) -> str:
        """Normalize text"""
        return text.lower().strip()
    
    def _get_entity(self, token, entity_map) -> str:
        """Get entity for token"""
        return entity_map.get(token.i, self._normalize(token.text))
    
    # === Handler methods for each dependency type ===
    
    def _handle_subject(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle nominal subject (nsubj, nsubjpass)"""
        triples = []
        subj = self._get_entity(token, entity_map)
        head = token.head
        
        # Copula: "X is Y"
        if head.pos_ == "AUX" or any(c.dep_ == "cop" for c in head.children):
            # Find attribute
            for child in head.children:
                if child.dep_ == "attr":
                    obj = self._get_entity(child, entity_map)
                    # Special case: "My name is X"
                    if token.text.lower() == "name":
                        for grandchild in token.children:
                            if grandchild.dep_ == "poss" and grandchild.text.lower() in {"my", "mine"}:
                                triples.append(DependencyTriple("you", "name", obj, "nsubj-name"))
                                return triples
                    triples.append(DependencyTriple(subj, "is", obj, "nsubj-copula"))
        
        # Active verb: "X does Y"
        elif head.pos_ == "VERB":
            verb = head.lemma_.lower()
            
            # Look for direct object
            for child in head.children:
                if child.dep_ in {"dobj", "obj"}:
                    obj = self._get_entity(child, entity_map)
                    # Canonicalize common verbs
                    if verb in {"have", "has", "had"}:
                        pred = "has"
                    else:
                        pred = verb
                    triples.append(DependencyTriple(subj, pred, obj, "nsubj-verb"))
            
            # Look for prepositional complements
            for child in head.children:
                if child.dep_ == "prep":
                    prep = child.text.lower()
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            obj = self._get_entity(grandchild, entity_map)
                            # Common patterns
                            if verb == "live" and prep == "in":
                                triples.append(DependencyTriple(subj, "lives_in", obj, "nsubj-prep"))
                            elif verb == "work" and prep in {"at", "for"}:
                                triples.append(DependencyTriple(subj, "works_at", obj, "nsubj-prep"))
                            elif verb in {"born", "bear"} and prep == "in":
                                triples.append(DependencyTriple(subj, "born_in", obj, "nsubj-prep"))
                            else:
                                triples.append(DependencyTriple(subj, f"{verb}_{prep}", obj, "nsubj-prep"))
        
        return triples
    
    def _handle_object(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle direct object"""
        triples = []
        obj = self._get_entity(token, entity_map)
        head = token.head
        
        if head.pos_ == "VERB":
            # Find subject
            for child in head.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = self._get_entity(child, entity_map)
                    verb = head.lemma_.lower()
                    if verb in {"have", "has", "had"}:
                        pred = "has"
                    else:
                        pred = verb
                    triples.append(DependencyTriple(subj, pred, obj, "dobj"))
                    break
        
        return triples
    
    def _handle_indirect_object(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle indirect object"""
        # Similar to direct object but marks recipient
        return self._handle_object(token, entity_map, doc)
    
    def _handle_attribute(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle attribute (copula complement)"""
        triples = []
        attr = self._get_entity(token, entity_map)
        
        # Find subject
        for child in token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = self._get_entity(child, entity_map)
                triples.append(DependencyTriple(subj, "is", attr, "attr"))
                break
        
        return triples
    
    def _handle_adjectival_mod(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle adjectival modifier"""
        triples = []
        modifier = self._get_entity(token, entity_map)
        head_entity = self._get_entity(token.head, entity_map)
        triples.append(DependencyTriple(head_entity, "has_quality", modifier, "amod"))
        return triples
    
    def _handle_adverbial_mod(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle adverbial modifier"""
        # Usually modifies actions, not entities
        return []
    
    def _handle_numeric_mod(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle numeric modifier"""
        triples = []
        number = token.text
        head_entity = self._get_entity(token.head, entity_map)
        triples.append(DependencyTriple(head_entity, "has_quantity", number, "nummod"))
        return triples
    
    def _handle_nominal_mod(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle nominal modifier"""
        triples = []
        modifier = self._get_entity(token, entity_map)
        head_entity = self._get_entity(token.head, entity_map)
        triples.append(DependencyTriple(head_entity, "modified_by", modifier, "nmod"))
        return triples
    
    def _handle_possessive(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle possessive relation"""
        triples = []
        possessor = self._get_entity(token, entity_map)
        possessed = self._get_entity(token.head, entity_map)
        
        # Canonicalize "my" to "you"
        if possessor in {"my", "mine"}:
            possessor = "you"
        
        triples.append(DependencyTriple(possessor, "has", possessed, "poss"))
        return triples
    
    def _handle_compound(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle compound (multiword expression)"""
        triples = []
        part = self._get_entity(token, entity_map)
        whole = self._get_entity(token.head, entity_map)
        triples.append(DependencyTriple(whole, "includes", part, "compound"))
        return triples
    
    def _handle_preposition(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle preposition"""
        # Prepositions create relations between head and pobj
        return []
    
    def _handle_prep_object(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle prepositional object"""
        # Handled by subject/verb handlers
        return []
    
    def _handle_conjunction(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle conjunction"""
        triples = []
        item1 = self._get_entity(token.head, entity_map)
        item2 = self._get_entity(token, entity_map)
        triples.append(DependencyTriple(item1, "and", item2, "conj"))
        return triples
    
    def _handle_apposition(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle apposition"""
        triples = []
        entity1 = self._get_entity(token.head, entity_map)
        entity2 = self._get_entity(token, entity_map)
        triples.append(DependencyTriple(entity1, "also_known_as", entity2, "appos"))
        return triples
    
    def _handle_adnominal_clause(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle adnominal clause"""
        # Complex - usually requires full clause analysis
        return []
    
    def _handle_adverbial_clause(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle adverbial clause"""
        # Complex - usually requires full clause analysis
        return []
    
    def _handle_clausal_comp(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle clausal complement"""
        # Complex - usually requires full clause analysis
        return []
    
    def _handle_open_clause(self, token, entity_map, doc) -> List[DependencyTriple]:
        """Handle open clausal complement"""
        # Complex - usually requires full clause analysis
        return []


# Test the systematic extractor
if __name__ == "__main__":
    extractor = SystematicExtractor()
    
    test_sentences = [
        "My name is Alex Thompson",
        "I live in Seattle",
        "I work at Microsoft",
        "My dog's name is Potola",
        "I have three pets",
        "My favorite color is blue",
        "Sarah and John are friends",
        "The old red car belongs to me"
    ]
    
    print("Systematic USGS-Style Extraction")
    print("="*60)
    
    for sentence in test_sentences:
        triples, entities = extractor.extract(sentence)
        
        print(f"\nSentence: '{sentence}'")
        print(f"Entities: {sorted(entities)}")
        if triples:
            print("Triples:")
            for t in triples:
                print(f"  {t}")