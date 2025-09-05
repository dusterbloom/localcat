#!/usr/bin/env python3
"""Debug parsing to see why extraction fails"""

import spacy
nlp = spacy.load("en_core_web_sm")

texts = [
    "My name is Alex",
    "I live in Seattle", 
    "My dog's name is Potola"
]

for text in texts:
    print(f"\nText: '{text}'")
    doc = nlp(text)
    
    # Show parse
    for token in doc:
        print(f"  {token.text:10} pos={token.pos_:5} dep={token.dep_:8} head={token.head.text}")
    
    # Check root
    for sent in doc.sents:
        root = sent.root
        print(f"  ROOT: {root.text} (pos={root.pos_})")
        
        # Check if copula
        is_copula = root.pos_ == "AUX" or any(c.dep_ == "cop" for c in root.children)
        print(f"  Is copula? {is_copula}")
        
        # Find subject/attr
        subj = None
        attr = None
        for child in root.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                subj = child
            elif child.dep_ in {"attr", "acomp", "amod"}:
                attr = child
        
        print(f"  Subject: {subj}")
        print(f"  Attribute: {attr}")