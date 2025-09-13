#!/usr/bin/env python3
"""
Test ReLiK reader-only approach for fast relation extraction
"""

import time
import torch
from relik.reader import RelikReaderForTripletExtraction

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print()

test_text = "Dr. Sarah Chen works at OpenAI as AI research director. She founded the company in 2015."

print("--- Testing Reader-Only Approach ---")
start = time.time()
try:
    # Load ONLY the reader component
    reader = RelikReaderForTripletExtraction.from_pretrained(
        "relik-ie/relik-reader-deberta-v3-small-re-wikipedia",
        device="cpu"
    )
    load_time = time.time() - start
    print(f"Reader load time: {load_time:.2f}s")
    
    # Test inference - need to figure out the right input format
    start = time.time()
    
    # Try different input formats
    try:
        # Format 1: Direct text
        result = reader(test_text)
        print(f"Format 1 (direct text): SUCCESS")
    except Exception as e1:
        print(f"Format 1 failed: {e1}")
        
        try:
            # Format 2: With entities
            from relik.reader.data import RelikReaderSample
            sample = RelikReaderSample(text=test_text)
            result = reader(sample)
            print(f"Format 2 (RelikReaderSample): SUCCESS")
        except Exception as e2:
            print(f"Format 2 failed: {e2}")
            
            try:
                # Format 3: Batch format
                result = reader([test_text])
                print(f"Format 3 (batch): SUCCESS")
            except Exception as e3:
                print(f"Format 3 failed: {e3}")
                raise e3
    
    inference_time = time.time() - start
    print(f"Inference time: {inference_time:.3f}s")
    
    # Check result
    print(f"Result type: {type(result)}")
    if hasattr(result, 'triplets'):
        print(f"Relations found: {len(result.triplets)}")
        for i, triplet in enumerate(result.triplets[:3]):
            print(f"  {i+1}. {triplet}")
    elif isinstance(result, list) and len(result) > 0:
        print(f"Got list with {len(result)} items")
        if hasattr(result[0], 'triplets'):
            print(f"First item has {len(result[0].triplets)} triplets")
    
except Exception as e:
    print(f"Reader approach failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Alternative: Use transformers pipeline ---")
try:
    from transformers import pipeline
    
    # Try a simple relation extraction pipeline
    start = time.time()
    classifier = pipeline(
        "text-classification",
        model="yseop/distilbert-base-financial-relation-extraction", 
        device=-1  # CPU
    )
    load_time = time.time() - start
    print(f"Transformers pipeline load time: {load_time:.2f}s")
    
    start = time.time()
    result = classifier(test_text)
    inference_time = time.time() - start
    print(f"Transformers inference time: {inference_time:.3f}s")
    print(f"Result: {result}")
    
except Exception as e:
    print(f"Transformers approach failed: {e}")

print("\n--- Testing Custom ReLiK Config ---")
try:
    from relik import Relik
    
    # Try custom config without index
    start = time.time()
    relik = Relik.from_pretrained(
        "relik-ie/relik-relation-extraction-small",
        retriever=None,
        device="cpu"
    )
    load_time = time.time() - start
    print(f"Custom config load time: {load_time:.2f}s")
    
    # Try a simpler text
    simple_text = "Apple Inc. is based in Cupertino."
    start = time.time()
    result = relik(simple_text)
    inference_time = time.time() - start
    print(f"Simple text inference: {inference_time:.3f}s")
    
    if hasattr(result, 'triplets'):
        print(f"Simple text relations: {len(result.triplets)}")
    
except Exception as e:
    print(f"Custom config failed: {e}")
    import traceback
    traceback.print_exc()