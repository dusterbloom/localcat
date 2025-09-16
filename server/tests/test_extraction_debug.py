#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import spacy
nlp = spacy.load("en_core_web_sm")

from components.extraction.enhanced_level3_extractor import QualityExtractor
extractor = QualityExtractor()

test_text = "I live in Sardinia."
doc = nlp(test_text)

# Manually walk through the extraction logic
print(f"Testing: '{test_text}'")
print("\nWalking through extraction logic:")

for sent in doc.sents:
    print(f"\nSentence: '{sent}'")
    
    for token in sent:
        print(f"\nToken: {token.text} (lemma={token.lemma_}, pos={token.pos_})")
        
        # Check if it's a core verb
        if token.lemma_.lower() in extractor.core_verbs and token.pos_ == 'VERB':
            print(f"  ✅ Found core verb: {token.lemma_}")
            
            # Find subjects
            subjects = [child for child in token.children if child.dep_ == 'nsubj']
            print(f"  Subjects: {[s.text for s in subjects]}")
            
            # Find objects
            objects = [child for child in token.children if child.dep_ in ['dobj', 'attr']]
            print(f"  Direct objects: {[o.text for o in objects]}")
            
            # Find prepositions
            prep_phrases = [child for child in token.children if child.dep_ == 'prep']
            print(f"  Prepositions: {[p.text for p in prep_phrases]}")
            
            for prep in prep_phrases:
                print(f"    Prep '{prep.text}' in allowed list: {prep.text.lower() in ['at', 'in', 'on', 'to', 'from', 'with', 'under', 'across', 'during', 'since']}")
                pobj = [child for child in prep.children if child.dep_ == 'pobj']
                print(f"    Prep objects: {[p.text for p in pobj]}")
