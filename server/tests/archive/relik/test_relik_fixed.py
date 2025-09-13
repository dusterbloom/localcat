#!/usr/bin/env python3
"""
Test ReLiK Reader with fixed API calls
"""

import time
import torch
from loguru import logger

def test_relik_reader_fixed():
    """Test ReLiK reader with correct API"""

    print("🔬 Testing ReLiK Reader with Fixed API")
    print("=" * 60)

    try:
        from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
        print("✅ ReLiK reader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ReLiK reader: {e}")
        return

    # Load reader
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🎯 Using {device} device")

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

    # Test text
    text = "Steve Jobs founded Apple Inc. in Cupertino, California."
    print(f"\n📝 Test text: {text}")

    # Create entity spans (as ReLiK expects)
    entities = ["Steve Jobs", "Apple Inc.", "Cupertino", "California"]
    spans = []
    for entity in entities:
        pos = text.find(entity)
        if pos != -1:
            spans.append({
                "start": pos,
                "end": pos + len(entity),
                "text": entity,
                "label": "--NME--"
            })

    print(f"📍 Entity spans: {[s['text'] for s in spans]}")

    # Test different API calls
    print("\n🧪 Testing different API approaches:")

    # Approach 1: With text, spans, and candidates
    print("\n1️⃣ Approach 1: text + spans + candidates")
    try:
        start = time.perf_counter()
        with torch.no_grad():
            result = reader.read(
                text=[text],
                spans=[spans],
                candidates=[[]],  # Empty candidates
                max_length=256
            )

        extract_time = (time.perf_counter() - start) * 1000

        if result and len(result) > 0:
            sample_result = result[0]
            print(f"   ✅ Got result type: {type(sample_result)}")

            if hasattr(sample_result, 'triplets'):
                print(f"   ✅ Found {len(sample_result.triplets)} triplets in {extract_time:.1f}ms")
                for triplet in sample_result.triplets[:3]:
                    subj = triplet.subject.text if hasattr(triplet.subject, 'text') else str(triplet.subject)
                    obj = triplet.object.text if hasattr(triplet.object, 'text') else str(triplet.object)
                    rel = triplet.label if hasattr(triplet, 'label') else "unknown"
                    print(f"      • {subj} --{rel}--> {obj}")
            else:
                print(f"   ⚠️ No triplets attribute found")
        else:
            print(f"   ⚠️ No result returned")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Approach 2: Just text and candidates (no spans)
    print("\n2️⃣ Approach 2: text + candidates only")
    try:
        start = time.perf_counter()
        with torch.no_grad():
            result = reader.read(
                text=[text],
                candidates=[[]],  # Empty candidates
                max_length=256
            )

        extract_time = (time.perf_counter() - start) * 1000

        if result and len(result) > 0:
            sample_result = result[0]
            print(f"   ✅ Got result in {extract_time:.1f}ms")
            if hasattr(sample_result, 'triplets'):
                print(f"   ✅ Found {len(sample_result.triplets)} triplets")
            else:
                print(f"   ⚠️ No triplets found")
        else:
            print(f"   ⚠️ No result returned")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print("✅ Test completed!")

if __name__ == "__main__":
    test_relik_reader_fixed()