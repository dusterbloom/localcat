#!/usr/bin/env python3
"""
Debug why UD extraction is failing on basic sentences
"""

import spacy
from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

# Load spacy
nlp = spacy.load("en_core_web_sm")

# Test sentences that are failing
test_sentences = [
    "My name is Alex Thompson",
    "I live in Seattle",
    "I work at Microsoft", 
    "My favorite color is blue",
    "I was born in 1995",
    "My son is named Jake",
    "My cat is called Whiskers"
]

print("Debugging UD Extraction Failures")
print("="*60)

for sentence in test_sentences:
    print(f"\nSentence: '{sentence}'")
    print("-"*40)
    
    doc = nlp(sentence)
    
    # Show dependency parse
    print("Tokens:")
    for token in doc:
        print(f"  {token.text:12} {token.pos_:6} {token.dep_:10} -> {token.head.text}")
    
    # Look for copula patterns
    copula_found = False
    for token in doc:
        if token.pos_ == "AUX" and token.dep_ == "cop":
            copula_found = True
            print(f"\nCopula found: {token.text}")
            print(f"  Subject: {[t for t in token.head.children if t.dep_ == 'nsubj']}")
            print(f"  Attribute: {token.head}")
    
    # Look for ROOT verbs
    for token in doc:
        if token.dep_ == "ROOT":
            print(f"\nRoot: {token.text} ({token.pos_})")
            subj = None
            obj = None
            for child in token.children:
                if child.dep_ == "nsubj":
                    subj = child
                if child.dep_ in ["dobj", "attr", "prep"]:
                    obj = child
            if subj and obj:
                print(f"  Potential relation: {subj} -> {token.text} -> {obj}")
    
    # Actually run the extraction
    print("\nTrying extraction with HotMemory...")
    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Extract
    bullets, triples = hot.process_turn(sentence, "test", 1)
    print(f"Extracted triples: {triples}")
    
print("\n" + "="*60)
print("Analysis complete")