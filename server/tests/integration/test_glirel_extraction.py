#!/usr/bin/env python3
"""
Test GLiREL integration for zero-shot relation extraction
"""

import time
from glirel import GLiREL

def test_glirel_basic():
    """Test basic GLiREL functionality"""

    print("🔬 Testing GLiREL Basic Integration")
    print("=" * 50)

    try:
        # Load GLiREL model
        print("📚 Loading GLiREL model...")
        start = time.perf_counter()
        model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
        load_time = (time.perf_counter() - start) * 1000
        print(f"✅ Model loaded in {load_time:.1f}ms")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Test text and entities
    text = "Steve Jobs founded Apple Inc. in Cupertino, California."

    # Simple entities (simulating GLiNER output)
    entities = [
        {"text": "Steve Jobs", "start": 0, "end": 10, "label": "PERSON"},
        {"text": "Apple Inc.", "start": 19, "end": 29, "label": "ORG"},
        {"text": "Cupertino", "start": 33, "end": 42, "label": "LOC"},
        {"text": "California", "start": 44, "end": 54, "label": "LOC"}
    ]

    print(f"📝 Text: {text}")
    print(f"📍 Entities: {[(e['text'], e['label']) for e in entities]}")

    # Define relations we want to extract (zero-shot!)
    relations = {
        'founded': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        'headquartered_in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC"]},
        'located_in': {"allowed_head": ["LOC"], "allowed_tail": ["LOC"]},
        'works_at': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        'born_in': {"allowed_head": ["PERSON"], "allowed_tail": ["LOC"]}
    }

    print(f"🎯 Relations: {list(relations.keys())}")

    # Test prediction
    print("\n🧪 Testing relation extraction...")
    start = time.perf_counter()

    try:
        # Get tokens (GLiREL needs tokenized input)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("jackboyla/glirel-large-v0")
        tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

        # Extract relations
        results = model.predict_relations(
            tokens=tokens,
            labels=relations,
            ner=entities,
            threshold=0.5
        )

        extract_time = (time.perf_counter() - start) * 1000

        print(f"✅ Success! Time: {extract_time:.1f}ms")
        print(f"📊 Found {len(results)} relations:")

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['head']} --{result['relation']}--> {result['tail']} (score: {result['score']:.3f})")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

def test_glirel_simplified():
    """Test GLiREL with simpler approach"""

    print("\n🔬 Testing GLiREL Simplified Approach")
    print("=" * 50)

    try:
        model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
        print("✅ Model loaded")

        # Test with spaCy integration if available
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm')
            print("✅ spaCy loaded")

            # Add GLiREL pipeline
            if "glirel" not in nlp.pipe_names:
                nlp.add_pipe("glirel", after="ner", config={
                    "model_name": "jackboyla/glirel-large-v0",
                    "threshold": 0.5
                })
                print("✅ GLiREL pipeline added")

            # Define relations
            labels = {
                'founded': {},
                'works_at': {},
                'located_in': {}
            }

            text = "Marie Curie discovered radium in Paris."
            print(f"📝 Text: {text}")

            # Process with spaCy + GLiREL
            docs = list(nlp.pipe([(text, labels)]))
            relations = docs[0][0]._.relations

            print(f"📊 Found {len(relations)} relations:")
            for i, rel in enumerate(relations, 1):
                print(f"  {i}. {rel}")

        except Exception as e:
            print(f"⚠️ spaCy integration failed: {e}")

    except Exception as e:
        print(f"❌ Model load failed: {e}")

if __name__ == "__main__":
    test_glirel_basic()
    test_glirel_simplified()