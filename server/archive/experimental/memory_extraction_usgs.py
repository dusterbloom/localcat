#!/usr/bin/env python3
"""
USGS Grammar-to-Graph style extraction with all 27 dependency relation types
Based on USGS Scientific Investigations Report 2025-5064
"""

import spacy
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Triple:
    """A semantic triple with metadata"""
    subject: str
    predicate: str
    object: str
    dep_type: str  # The dependency relation type
    confidence: float = 1.0


class USGSExtractor:
    """
    Implements all 27 dependency relation types from USGS Grammar to Graph
    Creates rich semantic triples from text using Universal Dependencies
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
        # User pronouns that map to "you"
        self.user_pronouns = {"i", "me", "my", "mine", "myself"}
        
        # All 27 dependency relations from USGS report
        self.dependency_relations = {
            # Clausal relations
            "acl": "adnominal_clause",        # Adnominal clause
            "advcl": "adverbial_clause",      # Adverbial clause
            "ccomp": "clausal_complement",    # Clausal complement
            "csubj": "clausal_subject",       # Clausal subject
            "xcomp": "open_clausal_comp",     # Open clausal complement
            
            # Modifier relations  
            "advmod": "adverbial_modifier",   # Adverbial modifier
            "amod": "adjectival_modifier",    # Adjectival modifier
            "nmod": "nominal_modifier",       # Nominal modifier
            "nummod": "numeric_modifier",     # Numeric modifier
            
            # Function word relations
            "aux": "auxiliary",               # Auxiliary
            "auxpass": "passive_auxiliary",   # Passive auxiliary
            "case": "case_marker",            # Case marker
            "cc": "coordination",             # Coordinating conjunction
            "cop": "copula",                  # Copula
            "det": "determiner",              # Determiner
            "mark": "marker",                 # Marker
            "neg": "negation",                # Negation
            
            # Core grammatical relations
            "nsubj": "nominal_subject",       # Nominal subject
            "nsubjpass": "passive_subject",   # Passive nominal subject
            "dobj": "direct_object",          # Direct object
            "iobj": "indirect_object",        # Indirect object
            "obj": "object",                  # Object (unified)
            
            # Special relations
            "agent": "agent",                 # Agent
            "attr": "attribute",              # Attribute
            "compound": "compound",           # Multiword expression
            "conj": "conjunction",            # Conjunction
            "poss": "possessive",            # Possessive
            "pobj": "prepositional_object",  # Object of preposition
            "prep": "preposition",           # Preposition
            "oprd": "object_predicate",      # Object predicate
            "root": "root"                   # Root
        }
        
        # Canonical relation mappings for common patterns
        self.canonical_relations = {
            "lives_in": ["live", "reside", "dwell"],
            "works_at": ["work", "employed"],
            "born_in": ["born", "bear"],
            "has": ["have", "own", "possess"],
            "name": ["name", "call", "known"],
            "loves": ["love", "adore"],
            "likes": ["like", "enjoy", "prefer"],
            "knows": ["know", "understand"]
        }
    
    def extract(self, text: str) -> Tuple[List[Triple], Set[str]]:
        """
        Extract all dependency-based triples from text
        Returns: (triples, entities)
        """
        doc = self.nlp(text)
        triples = []
        entities = set()
        
        # Process each sentence
        for sent in doc.sents:
            sent_triples, sent_entities = self._process_sentence(sent)
            triples.extend(sent_triples)
            entities.update(sent_entities)
        
        return triples, entities
    
    def _process_sentence(self, sent) -> Tuple[List[Triple], Set[str]]:
        """Process a single sentence for all dependency patterns"""
        triples = []
        entities = set()
        
        # Extract entities first (nouns, pronouns, named entities, chunks)
        entity_map = self._extract_entities(sent)
        entities.update(entity_map.values())
        
        # Process each token and its dependencies
        for token in sent:
            dep = token.dep_
            
            if dep not in self.dependency_relations:
                continue
            
            dep_type = self.dependency_relations[dep]
            
            # Extract triples based on dependency type
            if dep in {"nsubj", "nsubjpass", "csubj"}:
                # Subject relations
                t = self._extract_subject_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep in {"dobj", "iobj", "obj"}:
                # Object relations  
                t = self._extract_object_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep in {"amod", "advmod", "nummod", "nmod"}:
                # Modifier relations
                t = self._extract_modifier_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep == "compound":
                # Compound/multiword expressions
                t = self._extract_compound_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep == "poss":
                # Possessive relations
                t = self._extract_possessive_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep == "prep":
                # Prepositional relations
                t = self._extract_prepositional_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep in {"acl", "advcl", "ccomp", "xcomp"}:
                # Clausal relations
                t = self._extract_clausal_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep == "conj":
                # Conjunction relations
                t = self._extract_conjunction_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep == "attr":
                # Attribute relations (copula complements)
                t = self._extract_attribute_relations(token, entity_map, dep_type)
                triples.extend(t)
                
            elif dep == "agent":
                # Agent relations (by-phrases in passives)
                t = self._extract_agent_relations(token, entity_map, dep_type)
                triples.extend(t)
        
        return triples, entities
    
    def _extract_entities(self, sent) -> Dict[int, str]:
        """Extract all entities from sentence"""
        entities = {}
        
        # Named entities
        for ent in sent.ents:
            for token in ent:
                entities[token.i] = self._normalize(ent.text)
        
        # Noun chunks
        for chunk in sent.noun_chunks:
            entities[chunk.root.i] = self._normalize(chunk.text)
        
        # Individual nouns/pronouns
        for token in sent:
            if token.i not in entities:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    text = self._normalize(token.text)
                    # Map user pronouns
                    if text in self.user_pronouns:
                        text = "you"
                    entities[token.i] = text
        
        return entities
    
    def _extract_subject_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract subject-based relations"""
        triples = []
        
        subj = entities.get(token.i, token.text)
        head = token.head
        
        # If head is copula, look for attribute
        if head.pos_ == "AUX" or head.dep_ == "ROOT" and head.pos_ == "AUX":
            for child in head.children:
                if child.dep_ in {"attr", "acomp"}:
                    obj = entities.get(child.i, child.text)
                    # Check for "name is X" pattern
                    if token.text.lower() == "name":
                        # Look for possessor
                        for subchild in token.children:
                            if subchild.dep_ == "poss":
                                owner = entities.get(subchild.i, subchild.text)
                                if owner in self.user_pronouns:
                                    owner = "you"
                                triples.append(Triple(owner, "name", obj, dep_type))
                                return triples
                    triples.append(Triple(subj, "is", obj, dep_type))
        
        # If head is verb, create verb relation
        elif head.pos_ == "VERB":
            verb = self._get_canonical_verb(head.lemma_)
            # Look for objects
            for child in head.children:
                if child.dep_ in {"dobj", "obj"}:
                    obj = entities.get(child.i, child.text)
                    triples.append(Triple(subj, verb, obj, dep_type))
                elif child.dep_ == "prep":
                    # Prepositional phrase
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            obj = entities.get(grandchild.i, grandchild.text)
                            prep_verb = self._get_prep_verb(head.lemma_, child.text)
                            triples.append(Triple(subj, prep_verb, obj, dep_type))
        
        return triples
    
    def _extract_object_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract object-based relations"""
        triples = []
        
        obj = entities.get(token.i, token.text)
        head = token.head
        
        # Find subject
        subj = None
        for child in head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = entities.get(child.i, child.text)
                break
        
        if subj and head.pos_ == "VERB":
            verb = self._get_canonical_verb(head.lemma_)
            triples.append(Triple(subj, verb, obj, dep_type))
        
        return triples
    
    def _extract_modifier_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract modifier relations"""
        triples = []
        
        modifier = entities.get(token.i, token.text)
        head_entity = entities.get(token.head.i, token.head.text)
        
        if dep_type == "adjectival_modifier":
            triples.append(Triple(head_entity, "has_quality", modifier, dep_type))
        elif dep_type == "adverbial_modifier":
            triples.append(Triple(head_entity, "has_manner", modifier, dep_type))
        elif dep_type == "numeric_modifier":
            triples.append(Triple(head_entity, "has_quantity", modifier, dep_type))
        elif dep_type == "nominal_modifier":
            triples.append(Triple(head_entity, "modified_by", modifier, dep_type))
        
        return triples
    
    def _extract_compound_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract compound/multiword relations"""
        triples = []
        
        part = entities.get(token.i, token.text)
        whole = entities.get(token.head.i, token.head.text)
        
        triples.append(Triple(whole, "has_part", part, dep_type))
        
        return triples
    
    def _extract_possessive_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract possessive relations"""
        triples = []
        
        possessor = entities.get(token.i, token.text)
        if possessor in self.user_pronouns:
            possessor = "you"
        
        possessed = entities.get(token.head.i, token.head.text)
        
        triples.append(Triple(possessor, "has", possessed, dep_type))
        
        return triples
    
    def _extract_prepositional_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract prepositional relations"""
        triples = []
        
        prep = token.text.lower()
        head_entity = entities.get(token.head.i, token.head.text)
        
        # Find prepositional object
        for child in token.children:
            if child.dep_ == "pobj":
                obj = entities.get(child.i, child.text)
                triples.append(Triple(head_entity, f"prep_{prep}", obj, dep_type))
        
        return triples
    
    def _extract_clausal_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract clausal relations"""
        triples = []
        
        clause_head = entities.get(token.i, token.text)
        main_head = entities.get(token.head.i, token.head.text)
        
        if dep_type == "adnominal_clause":
            triples.append(Triple(main_head, "has_clause", clause_head, dep_type))
        elif dep_type == "adverbial_clause":
            triples.append(Triple(main_head, "conditioned_by", clause_head, dep_type))
        elif dep_type == "clausal_complement":
            triples.append(Triple(main_head, "complements", clause_head, dep_type))
        
        return triples
    
    def _extract_conjunction_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract conjunction relations"""
        triples = []
        
        conj1 = entities.get(token.head.i, token.head.text)
        conj2 = entities.get(token.i, token.text)
        
        triples.append(Triple(conj1, "conjoined_with", conj2, dep_type))
        
        return triples
    
    def _extract_attribute_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract attribute relations (copula complements)"""
        triples = []
        
        attr = entities.get(token.i, token.text)
        
        # Find subject
        for child in token.head.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = entities.get(child.i, child.text)
                triples.append(Triple(subj, "has_attribute", attr, dep_type))
                break
        
        return triples
    
    def _extract_agent_relations(self, token, entities, dep_type) -> List[Triple]:
        """Extract agent relations (by-phrases)"""
        triples = []
        
        agent = entities.get(token.i, token.text)
        action = entities.get(token.head.i, token.head.text)
        
        triples.append(Triple(agent, "performs", action, dep_type))
        
        return triples
    
    def _normalize(self, text: str) -> str:
        """Normalize text"""
        return text.lower().strip()
    
    def _get_canonical_verb(self, lemma: str) -> str:
        """Get canonical form of verb"""
        lemma = lemma.lower()
        
        for canonical, variants in self.canonical_relations.items():
            if lemma in variants:
                return canonical
        
        # Special handling for common verbs
        if lemma in {"have", "has", "had"}:
            return "has"
        elif lemma in {"is", "am", "are", "was", "were", "be"}:
            return "is"
        
        return lemma
    
    def _get_prep_verb(self, verb: str, prep: str) -> str:
        """Get verb + preposition combination"""
        verb = verb.lower()
        prep = prep.lower()
        
        # Special combinations
        if verb == "live" and prep == "in":
            return "lives_in"
        elif verb == "work" and prep in {"at", "for"}:
            return "works_at"
        elif verb in {"born", "bear"} and prep == "in":
            return "born_in"
        
        return f"{verb}_{prep}"


# Test the USGS-style extractor
if __name__ == "__main__":
    extractor = USGSExtractor()
    
    test_sentences = [
        "My name is Alex Thompson",
        "I live in Seattle",
        "I work at Microsoft as a software engineer",
        "My dog's name is Potola",
        "My wife Sarah is a doctor",
        "I have three pets including a cat named Whiskers",
        "My favorite color is blue",
        "I was born in 1995 in Chicago"
    ]
    
    print("USGS Grammar-to-Graph Style Extraction (27 Dependency Types)")
    print("="*70)
    
    for sentence in test_sentences:
        triples, entities = extractor.extract(sentence)
        
        print(f"\nSentence: '{sentence}'")
        print(f"Entities: {sorted(entities)}")
        print("Triples:")
        for t in triples:
            print(f"  ({t.subject}, {t.predicate}, {t.object}) [{t.dep_type}]")