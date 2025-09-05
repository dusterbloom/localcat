#!/usr/bin/env python3
"""
Debug UD parsing for "My dog's name is Potola"
"""

import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Parse the sentence
text = "My dog's name is Potola"
doc = nlp(text)

print(f"Parsing: '{text}'")
print("="*50)

# Show token analysis
print("\nToken Analysis:")
print(f"{'Token':<10} {'POS':<6} {'Dep':<10} {'Head':<10} {'Children'}")
print("-"*50)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<10} {token.pos_:<6} {token.dep_:<10} {token.head.text:<10} {children}")

print("\nDependency Tree:")
for sent in doc.sents:
    root = sent.root
    print(f"Root: {root.text} ({root.pos_})")
    
    # Show tree structure
    def print_subtree(token, indent=0):
        print("  " * indent + f"├─ {token.text} ({token.pos_}, {token.dep_})")
        for child in token.children:
            print_subtree(child, indent + 1)
    
    for child in root.children:
        print_subtree(child, 1)

# Analyze the possessive structure
print("\n" + "="*50)
print("Possessive Analysis:")

for token in doc:
    if token.dep_ == "poss":
        print(f"  Possessor: '{token.text}' -> Possessed: '{token.head.text}'")
        # Check what the possessed item is
        if token.head.dep_ == "nmod":
            print(f"    '{token.head.text}' modifies '{token.head.head.text}'")
            
    if token.dep_ == "nsubj":
        print(f"  Subject: '{token.text}' of verb/copula '{token.head.text}'")
        # Check for possessive modifier
        for child in token.children:
            if child.dep_ == "poss":
                print(f"    Has possessor: '{child.text}'")
            if child.dep_ == "nmod":
                print(f"    Has nominal modifier: '{child.text}'")
                
    if token.dep_ == "attr":
        print(f"  Attribute: '{token.text}' (the predicate)")

# The correct extraction
print("\n" + "="*50)
print("Correct Extraction:")

# Find the copula construction
for token in doc:
    if token.pos_ == "AUX" and token.dep_ == "cop":
        # This is a copula construction
        head = token.head  # The root (usually the attribute)
        
        # Find subject
        subj = None
        for child in head.children:
            if child.dep_ == "nsubj":
                subj = child
                break
        
        if subj and head:
            print(f"  Copula relation: {subj.text} --is--> {head.text}")
            
            # Check if subject has possessive structure
            poss_chain = []
            current = subj
            
            # Traverse possessive chain
            for child in subj.children:
                if child.dep_ == "poss":
                    poss_chain.append(child.text)
                if child.dep_ == "nmod" and child.text.endswith("'s"):
                    # Handle 's possessive
                    poss_chain.append(child.text.replace("'s", ""))
            
            if poss_chain:
                print(f"  Possessive chain: {poss_chain} -> {subj.text}")
                
            # Build the relations
            if poss_chain and subj.text == "name":
                owner = poss_chain[0] if poss_chain else None
                if owner == "My":
                    owner = "you"  # Convert to canonical form
                    
                # We have: owner's dog's name is Potola
                # Extract: (you, has, dog) and (dog, name, Potola)
                print(f"\n  Extracted relations:")
                print(f"    1. ({owner}, has, dog)")
                print(f"    2. (dog, name, {head.text})")

print("\n" + "="*50)
print("Using spacy-experimental displacy for visualization:")
print("python -m spacy download en_core_web_trf")
print("Then: python -m spacy parse 'My dog's name is Potola' --displacy")