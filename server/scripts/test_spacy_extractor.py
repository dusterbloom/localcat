#!/usr/bin/env python3
"""
Test script for a spaCy-based entity and relation extractor using a transformer model.

This script loads a multilingual spaCy pipeline with a transformer backbone,
runs it on sample texts, and validates the extracted entities and relations.
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
import spacy
from spacy.tokens import Doc

# --- Setup Project Environment ---
base = os.path.dirname(__file__)
server_path = os.path.abspath(os.path.join(base, '..', 'server'))
if server_path not in sys.path:
    sys.path.append(server_path)

# --- Core Extraction Logic ---

def extract_entities(doc: Doc) -> list[tuple[str, str]]:
    """Extracts named entities from a spaCy Doc."""
    return sorted([(ent.text, ent.label_) for ent in doc.ents])

def extract_relations(doc: Doc) -> list[tuple[str, str, str]]:
    """
    Extracts subject-verb-object triples using dependency parsing and noun chunks.
    """
    relations = []
    # Use noun chunks to get more meaningful subjects and objects
    for chunk in doc.noun_chunks:
        # The root of the chunk is often the main noun
        # The head of that root token is often the verb
        verb = chunk.root.head
        if verb.pos_ == "VERB":
            # Find the subject and object connected to this verb
            subjects = [c for c in verb.children if c.dep_ == "nsubj"]
            objects = [c for c in verb.children if c.dep_ in ("dobj", "pobj")]
            
            if subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        # Use the full text of the noun chunks for subject and object
                        subj_text = " ".join(t.text for t in subj.subtree)
                        obj_text = " ".join(t.text for t in obj.subtree)
                        relation_text = verb.lemma_
                        relations.append((subj_text, relation_text, obj_text))
    return sorted(list(set(relations))) # Use set to remove duplicates


# --- Test Cases (Updated) ---

TEST_CASES = [
    {
        "text": "Apple is looking at buying U.K. startup for $1 billion.",
        "lang": "en",
        "expected_entities": [('$1 billion', 'MONEY'), ('Apple', 'ORG'), ('U.K.', 'GPE')],
        "expected_relations": [('Apple', 'look', 'at buying U.K. startup')]
    },
    {
        "text": "Elon Musk fundó SpaceX en 2002.",
        "lang": "es",
        "expected_entities": [('2002', 'DATE'), ('Elon Musk', 'PERSON'), ('SpaceX', 'ORG')],
        "expected_relations": [('Elon Musk', 'fundar', 'SpaceX')]
    },
    {
        "text": "The company's CEO, John Smith, announced a new product.",
        "lang": "en",
        "expected_entities": [('John Smith', 'PERSON')],
        "expected_relations": [("The company's CEO, John Smith", "announce", "a new product")]
    }
]

async def main():
    """Main async function to run the test."""
    print("--- Starting spaCy Transformer Extractor Test ---")

    env_path = os.path.join(server_path, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        print(f"Loaded environment from: {env_path}")

    model = "en_core_web_trf"
    print(f"Loading spaCy transformer model: {model}...")
    try:
        import spacy_transformers
    except ImportError:
        print("\n[ERROR] `spacy-transformers` is not installed.")
        print("Please run: pip install spacy-transformers\n")
        return

    try:
        nlp = spacy.load(model)
        print("Model loaded successfully.")
    except OSError:
        print(f"\n[ERROR] Model '{model}' not found.")
        print(f"Please run: python -m spacy download {model}\n")
        return
    except Exception as e:
        print(f"Failed to load spaCy model: {e}")
        return

    passed_count = 0
    for i, case in enumerate(TEST_CASES):
        print(f"\n--- Test Case {i+1}: lang={case['lang']} ---")
        print(f"Text: '{case['text']}'")

        doc = nlp(case['text'])
        actual_entities = extract_entities(doc)
        actual_relations = extract_relations(doc)

        print(f"  Actual Entities  : {actual_entities}")
        print(f"  Expected Entities: {case['expected_entities']}")
        entities_match = (actual_entities == case['expected_entities'])
        if entities_match:
            print("  [PASS] Entities match expected.")
        else:
            print("  [FAIL] Entities do not match expected.")

        print(f"  Actual Relations  : {actual_relations}")
        print(f"  Expected Relations: {case['expected_relations']}")
        relations_match = (actual_relations == case['expected_relations'])
        if relations_match:
            print("  [PASS] Relations match expected.")
        else:
            print("  [FAIL] Relations do not match expected.")

        if entities_match and relations_match:
            passed_count += 1

    print("\n--- Test Summary ---")
    print(f"{passed_count}/{len(TEST_CASES)} test cases passed.")
    if passed_count == len(TEST_CASES):
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed.")

if __name__ == '__main__':
    asyncio.run(main())
