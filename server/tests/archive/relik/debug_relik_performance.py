#!/usr/bin/env python3
"""
Debug ReLiK performance issues
Testing the user's suggested fixes
"""

import time
import torch
from relik import Relik

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print()

test_text = "Dr. Sarah Chen works at OpenAI as AI research director. She founded the company in 2015."

# Test 1: Small model WITH retriever (current broken approach)
print("--- Test 1: Small WITH Retriever (BROKEN) ---")
start = time.time()
try:
    relik_with_retriever = Relik.from_pretrained(
        "relik-ie/relik-relation-extraction-small",
        device="cpu"
    )
    load_time = time.time() - start
    
    start = time.time()
    result = relik_with_retriever(test_text)
    inference_time = time.time() - start
    
    print(f"Load time: {load_time:.2f}s")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Total: {load_time + inference_time:.2f}s")
    
    if hasattr(result, 'triplets'):
        print(f"Relations found: {len(result.triplets)}")
    
except Exception as e:
    print(f"Failed: {e}")

print()

# Test 2: Small model WITHOUT retriever (FIXED)
print("--- Test 2: Small WITHOUT Retriever (FIXED) ---")
start = time.time()
try:
    relik_no_retriever = Relik.from_pretrained(
        "relik-ie/relik-relation-extraction-small",
        retriever=None,  # CRITICAL FIX!
        device="cpu"
    )
    load_time = time.time() - start
    
    start = time.time()
    result = relik_no_retriever(test_text)
    inference_time = time.time() - start
    
    print(f"Load time: {load_time:.2f}s")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Total: {load_time + inference_time:.2f}s")
    
    if hasattr(result, 'triplets'):
        print(f"Relations found: {len(result.triplets)}")
        for i, triplet in enumerate(result.triplets[:3]):
            print(f"  {i+1}. {triplet}")
    
except Exception as e:
    print(f"Failed: {e}")

print()

# Test 3: Test different model sizes
print("--- Test 3: Model Size Comparison ---")
models_to_test = [
    "relik-ie/relik-relation-extraction-tiny",
    "relik-ie/relik-relation-extraction-small"
]

for model_name in models_to_test:
    print(f"\nTesting {model_name}:")
    start = time.time()
    try:
        relik = Relik.from_pretrained(
            model_name,
            retriever=None,  # No retriever
            device="cpu"
        )
        load_time = time.time() - start
        
        start = time.time()
        result = relik(test_text)
        inference_time = time.time() - start
        
        print(f"  Load: {load_time:.2f}s, Inference: {inference_time:.2f}s")
        
        if hasattr(result, 'triplets'):
            print(f"  Relations: {len(result.triplets)}")
            
    except Exception as e:
        print(f"  Failed: {e}")

print("\n=== SUMMARY ===")
print("Key finding: The retriever loads millions of Wikipedia documents")
print("Solution: Always use retriever=None for pure relation extraction")
print("Target: <1s load time, <200ms inference time")