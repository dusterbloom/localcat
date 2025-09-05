#!/usr/bin/env python3
"""
Two-stage extraction: Entity Recognition + Relation Extraction
Based on research from USGS Grammar-to-Graph and GLiNER/GLiREL approaches
"""

import spacy
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass
class Entity:
    """Extracted entity with metadata"""
    text: str
    lemma: str
    pos: str
    dep: str
    ner_type: Optional[str] = None
    is_chunk: bool = False
    
@dataclass  
class Relation:
    """Extracted relation between entities"""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float = 1.0


class RobustExtractor:
    """
    Two-stage extraction system:
    1. Entity extraction (NER + noun chunks + pronouns)
    2. Relation extraction (dependency patterns)
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
        # Pronouns that map to user
        self.user_pronouns = {"i", "me", "my", "mine", "myself"}
        
        # Copula verbs
        self.copula_verbs = {"be", "is", "am", "are", "was", "were", "being", "been"}
        
        # Possessive relations
        self.possessive_deps = {"poss", "nmod:poss"}
        
        # Common relation verbs and their canonical forms
        self.relation_verbs = {
            "have": "has",
            "has": "has", 
            "had": "has",
            "own": "has",
            "owns": "has",
            "possess": "has",
            "live": "lives_in",
            "lives": "lives_in",
            "lived": "lives_in",
            "reside": "lives_in",
            "work": "works_at",
            "works": "works_at",
            "worked": "works_at",
            "born": "born_in",
            "bear": "born_in",
            "name": "name",
            "named": "name",
            "call": "name",
            "called": "name",
            "know": "knows",
            "love": "loves",
            "like": "likes",
            "hate": "hates",
            "want": "wants"
        }
    
    def extract(self, text: str) -> Tuple[List[Relation], Set[str]]:
        """
        Extract relations and entities from text
        Returns: (relations, entities)
        """
        doc = self.nlp(text)
        
        # Stage 1: Extract all entities
        entities = self._extract_entities(doc)
        
        # Stage 2: Extract relations between entities
        relations = self._extract_relations(doc, entities)
        
        # Convert to simple format
        simple_relations = []
        entity_texts = set()
        
        for rel in relations:
            # Map pronouns to "you"
            subj_text = "you" if rel.subject.text.lower() in self.user_pronouns else rel.subject.text.lower()
            obj_text = "you" if rel.object.text.lower() in self.user_pronouns else rel.object.text.lower()
            
            simple_relations.append((subj_text, rel.predicate, obj_text))
            entity_texts.add(subj_text)
            entity_texts.add(obj_text)
        
        return simple_relations, entity_texts
    
    def _extract_entities(self, doc) -> Dict[int, Entity]:
        """
        Stage 1: Extract all potential entities
        """
        entities = {}
        
        # 1. Named entities from NER
        for ent in doc.ents:
            for token in ent:
                entities[token.i] = Entity(
                    text=ent.text,
                    lemma=ent.lemma_,
                    pos="PROPN",
                    dep=token.dep_,
                    ner_type=ent.label_
                )
        
        # 2. Noun chunks (compound nouns)
        for chunk in doc.noun_chunks:
            # Store the chunk's head
            entities[chunk.root.i] = Entity(
                text=chunk.text,
                lemma=chunk.lemma_,
                pos=chunk.root.pos_,
                dep=chunk.root.dep_,
                is_chunk=True
            )
        
        # 3. Individual nouns and pronouns not captured above
        for token in doc:
            if token.i not in entities:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    entities[token.i] = Entity(
                        text=token.text,
                        lemma=token.lemma_,
                        pos=token.pos_,
                        dep=token.dep_
                    )
        
        return entities
    
    def _extract_relations(self, doc, entities: Dict[int, Entity]) -> List[Relation]:
        """
        Stage 2: Extract relations using dependency patterns
        """
        relations = []
        
        for sent in doc.sents:
            root = sent.root
            
            # Pattern 1: Copula constructions (X is Y)
            # Handle both "cop" dependency and ROOT AUX
            if self._is_copula_construction(root):
                rels = self._extract_copula_relations(root, entities)
                relations.extend(rels)
            
            # Pattern 2: Active voice SVO (I have a dog)
            elif root.pos_ == "VERB":
                rels = self._extract_verb_relations(root, entities)
                relations.extend(rels)
            
            # Pattern 3: Possessive constructions anywhere
            for token in sent:
                if token.dep_ in self.possessive_deps:
                    rels = self._extract_possessive_relations(token, entities)
                    relations.extend(rels)
        
        return relations
    
    def _is_copula_construction(self, root) -> bool:
        """Check if this is a copula construction"""
        # Case 1: ROOT is AUX (spaCy quirk)
        if root.pos_ == "AUX" and root.lemma_ in self.copula_verbs:
            return True
        
        # Case 2: Has cop dependency
        for child in root.children:
            if child.dep_ == "cop":
                return True
        
        return False
    
    def _extract_copula_relations(self, root, entities: Dict[int, Entity]) -> List[Relation]:
        """Extract relations from copula constructions"""
        relations = []
        
        # Find subject and attribute
        subj = None
        attr = None
        
        for child in root.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                if child.i in entities:
                    subj = entities[child.i]
            elif child.dep_ in {"attr", "acomp", "amod"}:
                if child.i in entities:
                    attr = entities[child.i]
        
        if subj and attr:
            # Check if subject is "name" with possessive
            if subj.text.lower() in {"name", "nombre"}:
                # Look for possessive modifier
                for child in root.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        for grandchild in child.children:
                            if grandchild.dep_ in self.possessive_deps:
                                owner = entities.get(grandchild.i)
                                if owner:
                                    relations.append(Relation(owner, "name", attr))
                                    return relations
            
            # Default copula relation
            predicate = "is"
            relations.append(Relation(subj, predicate, attr))
        
        return relations
    
    def _extract_verb_relations(self, root, entities: Dict[int, Entity]) -> List[Relation]:
        """Extract relations from verb constructions"""
        relations = []
        
        # Find subject
        subj = None
        for child in root.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                if child.i in entities:
                    subj = entities[child.i]
                    break
        
        if not subj:
            return relations
        
        # Pattern 1: Direct object (I have a dog)
        for child in root.children:
            if child.dep_ in {"dobj", "obj"}:
                if child.i in entities:
                    obj = entities[child.i]
                    verb_lemma = root.lemma_.lower()
                    predicate = self.relation_verbs.get(verb_lemma, f"v:{verb_lemma}")
                    relations.append(Relation(subj, predicate, obj))
        
        # Pattern 2: Prepositional object (I live in Seattle)
        for child in root.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        if grandchild.i in entities:
                            obj = entities[grandchild.i]
                            verb_lemma = root.lemma_.lower()
                            prep = child.text.lower()
                            
                            # Special patterns
                            if verb_lemma == "live" and prep == "in":
                                predicate = "lives_in"
                            elif verb_lemma == "work" and prep in {"at", "for"}:
                                predicate = "works_at"
                            elif verb_lemma in {"born", "bear"} and prep == "in":
                                predicate = "born_in"
                            else:
                                predicate = f"{verb_lemma}_{prep}"
                            
                            relations.append(Relation(subj, predicate, obj))
        
        # Pattern 3: Passive voice (My son is named Jake)
        if any(c.dep_ in {"auxpass", "aux:pass"} for c in root.children):
            if root.lemma_.lower() in {"name", "call"}:
                for child in root.children:
                    if child.dep_ in {"oprd", "xcomp", "attr"}:
                        if child.i in entities:
                            obj = entities[child.i]
                            relations.append(Relation(subj, "name", obj))
        
        return relations
    
    def _extract_possessive_relations(self, poss_token, entities: Dict[int, Entity]) -> List[Relation]:
        """Extract possessive relations (My dog)"""
        relations = []
        
        # The possessed entity is the head of the possessive
        if poss_token.head.i in entities:
            possessed = entities[poss_token.head.i]
            
            # The possessor
            if poss_token.i in entities:
                possessor = entities[poss_token.i]
                relations.append(Relation(possessor, "has", possessed))
        
        return relations


# Test the improved extractor
if __name__ == "__main__":
    extractor = RobustExtractor()
    
    test_sentences = [
        "My name is Alex Thompson",
        "I live in Seattle", 
        "I work at Microsoft",
        "My favorite color is blue",
        "I was born in 1995",
        "My son is named Jake",
        "My dog's name is Potola",
        "I have three pets"
    ]
    
    print("Testing Robust Two-Stage Extraction")
    print("="*60)
    
    for sentence in test_sentences:
        relations, entities = extractor.extract(sentence)
        print(f"\nSentence: '{sentence}'")
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")