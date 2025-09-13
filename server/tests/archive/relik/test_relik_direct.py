#!/usr/bin/env python3
"""
Test ReLiK Reader directly without retriever
This shows how to properly use ReLiK for relation extraction
"""

import time
import torch
from typing import List, Dict, Any
from loguru import logger

def test_relik_reader_direct():
    """Test ReLiK reader component directly"""

    print("🔬 Testing ReLiK Reader Direct Mode")
    print("=" * 60)

    try:
        from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
        print("✅ ReLiK reader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ReLiK reader: {e}")
        return

    # Detect best device
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"🎯 Using MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"🎯 Using CUDA acceleration")
    else:
        device = "cpu"
        print(f"🎯 Using CPU")

    # Load ONLY the reader component
    print("\n📚 Loading ReLiK reader model...")
    start = time.perf_counter()

    try:
        reader = RelikReaderForTripletExtraction.from_pretrained(
            "relik-ie/relik-reader-deberta-v3-small-re-wikipedia",
            device=device
        )
        load_time = (time.perf_counter() - start) * 1000
        print(f"✅ Reader loaded in {load_time:.1f}ms")
    except Exception as e:
        print(f"❌ Failed to load reader: {e}")
        return

    # Test sentences
    test_cases = [
        {
            "text": "Steve Jobs founded Apple Inc. in Cupertino, California.",
            "entities": [
                {"text": "Steve Jobs", "start": 0, "end": 10},
                {"text": "Apple Inc.", "start": 19, "end": 29},
                {"text": "Cupertino", "start": 33, "end": 42},
                {"text": "California", "start": 44, "end": 54}
            ]
        },
        {
            "text": "Marie Curie discovered radium and polonium while working in Paris.",
            "entities": [
                {"text": "Marie Curie", "start": 0, "end": 11},
                {"text": "radium", "start": 23, "end": 29},
                {"text": "polonium", "start": 34, "end": 42},
                {"text": "Paris", "start": 60, "end": 65}
            ]
        }
    ]

    print("\n🧪 Testing extraction:")
    print("-" * 40)

    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        entities = test_case["entities"]

        print(f"\nTest {i}: {text[:50]}...")

        # Convert entities to ReLiK format
        spans = []
        for ent in entities:
            spans.append({
                "start": ent["start"],
                "end": ent["end"],
                "text": ent["text"],
                "label": "--NME--"  # ReLiK's entity marker
            })

        print(f"  Entities: {[s['text'] for s in spans]}")

        # Create batch for reader
        batch = {
            "text": [text],
            "spans": [spans],
            "candidates": [[]]  # Empty candidates since no retriever
        }

        # Run extraction
        start = time.perf_counter()

        try:
            # Use the reader directly
            with torch.no_grad():
                # The reader expects batched input
                result = reader.read(
                    text=[text],
                    samples=[{
                        "text": text,
                        "spans": spans,
                        "candidates": []
                    }],
                    max_length=256
                )

            extract_time = (time.perf_counter() - start) * 1000

            # Parse results
            if result and len(result) > 0:
                sample_result = result[0]

                # Check for triplets
                if hasattr(sample_result, 'triplets') and sample_result.triplets:
                    print(f"  ✅ Extracted {len(sample_result.triplets)} relations in {extract_time:.1f}ms:")
                    for triplet in sample_result.triplets[:5]:
                        # Triplet has subject, object, and label attributes
                        subj = triplet.subject.text if hasattr(triplet.subject, 'text') else str(triplet.subject)
                        obj = triplet.object.text if hasattr(triplet.object, 'text') else str(triplet.object)
                        rel = triplet.label if hasattr(triplet, 'label') else triplet.relation
                        print(f"    • {subj} --{rel}--> {obj}")
                else:
                    print(f"  ⚠️  No triplets found (took {extract_time:.1f}ms)")
            else:
                print(f"  ❌ No result returned (took {extract_time:.1f}ms)")

        except Exception as e:
            print(f"  ❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Test completed!")

def test_with_gliner_entities():
    """Test using GLiNER for entity extraction then ReLiK for relations"""

    print("\n🔬 Testing ReLiK with GLiNER Entity Extraction")
    print("=" * 60)

    # First, test if GLiNER is available
    try:
        from components.extraction.gliner_extractor import GLiNERExtractor
        gliner = GLiNERExtractor()
        print("✅ Using GLiNER for entity extraction")
        use_gliner = True
    except:
        print("⚠️  GLiNER not available, using fallback entities")
        use_gliner = False

    # Load ReLiK reader
    try:
        from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        reader = RelikReaderForTripletExtraction.from_pretrained(
            "relik-ie/relik-reader-deberta-v3-small-re-wikipedia",
            device=device
        )
        print(f"✅ ReLiK reader loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load ReLiK: {e}")
        return

    # Test text
    text = "Dr. Sarah Chen is the AI research director at OpenAI. She previously worked at Google Brain."
    print(f"\nText: {text}")

    # Get entities
    if use_gliner:
        result = gliner.extract(text)
        entities = result.entities
        print(f"GLiNER entities: {entities}")

        # Convert to spans with positions
        spans = []
        for entity in entities:
            # Find entity position in text
            pos = text.lower().find(entity.lower())
            if pos != -1:
                spans.append({
                    "start": pos,
                    "end": pos + len(entity),
                    "text": entity,
                    "label": "--NME--"
                })
    else:
        # Manual fallback entities
        spans = [
            {"start": 4, "end": 14, "text": "Sarah Chen", "label": "--NME--"},
            {"start": 47, "end": 53, "text": "OpenAI", "label": "--NME--"},
            {"start": 80, "end": 92, "text": "Google Brain", "label": "--NME--"}
        ]

    print(f"Entity spans: {[s['text'] for s in spans]}")

    # Extract relations with ReLiK
    start = time.perf_counter()

    try:
        with torch.no_grad():
            result = reader.read(
                text=[text],
                samples=[{
                    "text": text,
                    "spans": spans,
                    "candidates": []
                }],
                max_length=256
            )

        extract_time = (time.perf_counter() - start) * 1000

        if result and len(result) > 0 and hasattr(result[0], 'triplets'):
            triplets = result[0].triplets
            print(f"\n✅ Extracted {len(triplets)} relations in {extract_time:.1f}ms:")
            for triplet in triplets:
                subj = triplet.subject.text if hasattr(triplet.subject, 'text') else str(triplet.subject)
                obj = triplet.object.text if hasattr(triplet.object, 'text') else str(triplet.object)
                rel = triplet.label
                print(f"  • {subj} --{rel}--> {obj}")
        else:
            print(f"⚠️  No relations found (took {extract_time:.1f}ms)")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    # Test direct reader usage
    test_relik_reader_direct()

    # Test with GLiNER integration
    test_with_gliner_entities()