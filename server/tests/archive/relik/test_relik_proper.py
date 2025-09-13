#!/usr/bin/env python3
"""
Test proper ReLiK sample format
"""

import time
import torch
from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
from relik.reader.pytorch_modules.hf.modeling_relik import RelikReaderSample

def test_proper_sample_format():
    """Test ReLiK with proper RelikReaderSample"""

    print("🔬 Testing ReLiK with Proper Sample Format")
    print("=" * 60)

    # Load reader
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    reader = RelikReaderForTripletExtraction.from_pretrained(
        "relik-ie/relik-reader-deberta-v3-small-re-wikipedia",
        device=device
    )
    print(f"✅ Reader loaded on {device}")

    # Test text
    text = "Steve Jobs founded Apple Inc. in Cupertino."
    print(f"📝 Text: {text}")

    # Create entities as spans
    entities = ["Steve Jobs", "Apple Inc.", "Cupertino"]
    spans = []
    for entity in entities:
        pos = text.find(entity)
        if pos != -1:
            spans.append((pos, pos + len(entity)))

    print(f"📍 Spans: {spans}")

    # Create proper RelikReaderSample
    sample = RelikReaderSample(
        text=text,
        spans=spans,
        candidates=[],
        offset=0,
        _mixin_prediction_position=None
    )

    # Test extraction
    start = time.perf_counter()
    try:
        with torch.no_grad():
            result = reader.read(
                text=[text],
                samples=[sample],
                max_length=256
            )

        extract_time = (time.perf_counter() - start) * 1000
        print(f"✅ Success! Time: {extract_time:.1f}ms")

        if result and len(result) > 0:
            sample_result = result[0]
            if hasattr(sample_result, 'triplets'):
                print(f"📊 Found {len(sample_result.triplets)} triplets:")
                for i, triplet in enumerate(sample_result.triplets, 1):
                    subj = triplet.subject.text if hasattr(triplet.subject, 'text') else str(triplet.subject)
                    obj = triplet.object.text if hasattr(triplet.object, 'text') else str(triplet.object)
                    rel = triplet.label
                    print(f"  {i}. {subj} --{rel}--> {obj}")
            else:
                print("⚠️ No triplets found")
        else:
            print("⚠️ No result returned")

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_proper_sample_format()