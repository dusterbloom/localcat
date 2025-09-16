#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import spacy
nlp = spacy.load("en_core_web_sm")

from components.extraction.enhanced_level3_extractor import QualityExtractor
from components.extraction.enhanced_level3_extractor import QualityRelation

# Patch to see ALL relations created before filtering
original_extract = QualityExtractor._extract_candidate_relations

created_relations = []

def debug_extract(self, doc):
    relations = []
    for sent in doc.sents:
        for token in sent:
            if token.lemma_.lower() in self.core_verbs and token.pos_ == 'VERB':
                subjects = [child for child in token.children if child.dep_ == 'nsubj']
                for subj in subjects:
                    subj_text = self._get_clean_noun_phrase(subj, role='subject')
                    print(f"[DEBUG] Subject: '{subj_text}' from token '{subj.text}'")
                    
                    # Check prepositions
                    prep_phrases = [child for child in token.children if child.dep_ == 'prep']
                    for prep in prep_phrases:
                        print(f"[DEBUG] Found prep: '{prep.text}'")
                        if prep.text.lower() in ['at', 'in', 'on', 'to', 'from', 'with', 'under', 'across', 'during', 'since']:
                            pobj = [child for child in prep.children if child.dep_ == 'pobj']
                            if pobj:
                                pobj_text = self._get_clean_noun_phrase(pobj[0])
                                print(f"[DEBUG] Creating relation: {subj_text} | {token.lemma_.lower()}_{prep.text.lower()} | {pobj_text}")
                                
                                rel = QualityRelation(
                                    id=f"relation_test",
                                    subject=subj_text,
                                    predicate=f"{token.lemma_.lower()}_{prep.text.lower()}",
                                    object=pobj_text,
                                    relation_type="spatial_temporal",
                                    confidence=0.88,
                                    source_sentence=0,
                                    semantic_roles={}
                                )
                                relations.append(rel)
                                created_relations.append(rel)
    
    result = original_extract(self, doc)
    print(f"[DEBUG] Original method returned {len(result)} relations")
    print(f"[DEBUG] Our debug method created {len(relations)} relations")
    return result

QualityExtractor._extract_candidate_relations = debug_extract

extractor = QualityExtractor()
doc = nlp("I live in Sardinia.")

result = extractor.extract_quality_kg(doc)

print(f"\n✅ Debug relations created: {len(created_relations)}")
for r in created_relations:
    print(f"  {r.subject} | {r.predicate} | {r.object} (conf={r.confidence})")
