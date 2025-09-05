#!/usr/bin/env python3
"""
Debug extraction directly
"""

import spacy
from typing import Set, List, Tuple

nlp = spacy.load("en_core_web_sm")

def extract_debug(text: str):
    doc = nlp(text)
    entities = set()
    triples = []
    entity_map = {}
    
    # Extract entities
    for ent in doc.ents:
        entities.add(ent.text.lower())
        for token in ent:
            entity_map[token.i] = ent.text.lower()
    
    for chunk in doc.noun_chunks:
        entities.add(chunk.text.lower())
        entity_map[chunk.root.i] = chunk.text.lower()
    
    for token in doc:
        if token.i not in entity_map:
            if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                text = token.text.lower()
                if text in {"i", "me", "my"}:
                    text = "you"
                entities.add(text)
                entity_map[token.i] = text
    
    print(f"Entity map: {entity_map}")
    
    # Extract relations
    for sent in doc.sents:
        root = sent.root
        print(f"Root: {root.text} (pos={root.pos_})")
        
        # Check copula
        is_copula = root.pos_ == "AUX" or any(c.dep_ == "cop" for c in root.children)
        
        if is_copula or root.pos_ == "AUX":
            print("  Processing as copula")
            
            # Find subject and attr
            subj = None
            attr = None
            for child in root.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = child
                    print(f"  Found subject: {child.text}")
                if child.dep_ in {"attr", "acomp"}:
                    attr = child
                    print(f"  Found attr: {child.text}")
            
            if subj and attr:
                print(f"  Have both subj and attr")
                
                # Check for "My name is X"
                if subj.text.lower() == "name":
                    print("  Subject is 'name'")
                    # Find possessive
                    poss = None
                    for child in subj.children:
                        if child.dep_ == "poss":
                            poss = child
                            print(f"  Found possessive: {child.text}")
                            break
                    
                    if poss and poss.text.lower() in {"my", "mine"}:
                        attr_text = entity_map.get(attr.i, attr.text.lower())
                        triples.append(("you", "name", attr_text))
                        print(f"  Added triple: ('you', 'name', '{attr_text}')")
    
    return entities, triples

# Test
texts = [
    "My name is Alex",
    "I live in Seattle"
]

for text in texts:
    print(f"\nTesting: '{text}'")
    print("-" * 40)
    entities, triples = extract_debug(text)
    print(f"Entities: {entities}")
    print(f"Triples: {triples}")