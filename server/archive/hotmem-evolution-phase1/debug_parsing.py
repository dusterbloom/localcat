#!/usr/bin/env python3
"""
Debug spaCy parsing of "My favorite number is 77" to understand the dependency structure
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import spacy
from memory_hotpath import HotMemory
from memory_store import MemoryStore, Paths

def debug_parsing():
    """Debug the dependency parsing of the problematic sentence"""
    
    print("=== Debugging spaCy Parsing ===")
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    
    # Test sentences
    test_sentences = [
        "My favorite number is 77.",
        "My dog's name is Potola.",
        "I have a favorite number.",
        "The favorite number is 77."
    ]
    
    for sentence in test_sentences:
        print(f"\n--- Analyzing: '{sentence}' ---")
        doc = nlp(sentence)
        
        print("Token dependencies:")
        for token in doc:
            print(f"  {token.text:<12} | {token.dep_:<12} | {token.pos_:<8} | head: {token.head.text} | children: {[child.text for child in token.children]}")
        
        print("\nDependency tree:")
        for token in doc:
            if token.dep_ != "ROOT":
                print(f"  {token.text} --{token.dep_}--> {token.head.text}")
            else:
                print(f"  {token.text} [ROOT]")
    
    # Test with HotMemory extraction
    print(f"\n=== Testing HotMemory Extraction ===")
    
    paths = Paths(sqlite_path="test_parse.db", lmdb_dir="test_parse.lmdb")
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    for sentence in test_sentences:
        print(f"\n--- HotMemory extraction for: '{sentence}' ---")
        
        try:
            entities, triples, neg_count, doc = hot._extract(sentence, "en")
            print(f"  Entities: {entities}")
            print(f"  Triples: {triples}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n=== Parsing Debug Complete ===")

if __name__ == "__main__":
    debug_parsing()