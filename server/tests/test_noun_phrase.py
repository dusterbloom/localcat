#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import spacy
nlp = spacy.load("en_core_web_sm")

from components.extraction.enhanced_level3_extractor import QualityExtractor
extractor = QualityExtractor()

doc = nlp("I live in Sardinia.")

for token in doc:
    if token.text == "I":
        result = extractor._get_clean_noun_phrase(token, role='subject')
        print(f"_get_clean_noun_phrase('I', role='subject') = '{result}'")
        print(f"  Token subtree: {list(token.subtree)}")
        print(f"  Token children: {list(token.children)}")
        print(f"  Token POS: {token.pos_}")
