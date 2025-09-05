#!/usr/bin/env python3
import spacy

nlp = spacy.load("en_core_web_sm")

texts = [
    "Caroline went to the LGBTQ support group",
    "Melanie painted a sunrise",
    "My son is named Jake"
]

for text in texts:
    doc = nlp(text)
    print(f"\n'{text}'")
    
    for token in doc:
        print(f"  {token.text:10} pos={token.pos_:5} dep={token.dep_:8} lemma={token.lemma_}")
    
    # Check subject
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            print(f"  Subject: {token.text}, Head: {token.head.text} (pos={token.head.pos_}, lemma={token.head.lemma_})")