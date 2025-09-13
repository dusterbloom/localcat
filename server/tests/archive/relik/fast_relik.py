#!/usr/bin/env python3
"""
Fast ReLiK implementation that bypasses the slow Wikipedia retriever
Uses only the reader component for fast relation extraction
"""

import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Span:
    start: int
    end: int
    text: str
    label: str = "--NME--"

@dataclass
class Triplet:
    subject: Span
    object: Span
    label: str
    confidence: float = 1.0

class FastReLiK:
    """
    Ultra-fast ReLiK wrapper that bypasses the Wikipedia retriever
    """
    
    def __init__(self, model_name: str = "relik-ie/relik-relation-extraction-small", device: str = "cpu"):
        """Initialize fast ReLiK without retriever"""
        self.model_name = model_name
        self.device = device
        self.reader = None
        self.tokenizer = None
        self._load_components()
    
    def _load_components(self):
        """Load only the reader component"""
        print(f"Loading fast ReLiK: {self.model_name}")
        start = time.time()
        
        try:
            from relik import Relik
            
            # Load the full model temporarily
            temp_model = Relik.from_pretrained(
                self.model_name,
                retriever=None,  # Don't load retriever
                device=self.device
            )
            
            # Extract the reader
            self.reader = temp_model.reader
            
            # Try to get tokenizer
            if hasattr(temp_model, 'tokenizer'):
                self.tokenizer = temp_model.tokenizer
            elif hasattr(self.reader, 'tokenizer'):
                self.tokenizer = self.reader.tokenizer
            
            # Clean up
            del temp_model
            
            load_time = time.time() - start
            print(f"✅ Fast ReLiK loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to load ReLiK: {e}")
            self.reader = None
            self.tokenizer = None
    
    def extract_triplets(self, text: str, entities: Optional[List[Dict]] = None) -> List[Triplet]:
        """
        Extract relation triplets from text using entities
        
        Args:
            text: Input text
            entities: List of entity dicts with 'text', 'start', 'end' keys
                     If None, requires entities to be provided separately
        
        Returns:
            List of Triplet objects
        """
        if self.reader is None:
            return []
        
        if entities is None:
            raise ValueError("Entities must be provided for fast extraction")
        
        try:
            # Convert entities to spans
            spans = []
            for ent in entities:
                span = Span(
                    start=ent.get('start', 0),
                    end=ent.get('end', len(ent['text'])),
                    text=ent.get('text', ''),
                    label=ent.get('label', '--NME--')
                )
                spans.append(span)
            
            # Create input format for reader
            # This is the key - bypassing the retriever completely
            input_data = {
                'text': text,
                'spans': spans,
                'candidates': []  # No candidates from retriever
            }
            
            # Process with reader
            start = time.time()
            result = self.reader(input_data)
            inference_time = time.time() - start
            
            # Extract triplets from result
            triplets = []
            if hasattr(result, 'triplets'):
                for triplet in result.triplets:
                    triplets.append(triplet)
            elif isinstance(result, list):
                triplets = result
            
            print(f"🚀 Fast extraction: {len(triplets)} relations in {inference_time*1000:.1f}ms")
            return triplets
            
        except Exception as e:
            print(f"❌ Fast extraction failed: {e}")
            return []
    
    def __call__(self, text: str, entities: List[Dict]) -> Any:
        """Make the object callable like original ReLiK"""
        triplets = self.extract_triplets(text, entities)
        
        # Create a result object similar to original ReLiK
        class FastResult:
            def __init__(self, triplets):
                self.triplets = triplets
        
        return FastResult(triplets)

def test_fast_relik():
    """Test the fast ReLiK implementation"""
    print("🧪 TESTING FAST RELIK")
    print("=" * 50)
    
    # Test text
    text = "Dr. Sarah Chen works at OpenAI as AI research director. She founded the company in 2015."
    
    # Mock entities (normally from GLiNER)
    entities = [
        {"text": "Dr. Sarah Chen", "start": 0, "end": 14, "label": "PERSON"},
        {"text": "OpenAI", "start": 24, "end": 30, "label": "ORG"},
        {"text": "AI research director", "start": 34, "end": 54, "label": "TITLE"},
        {"text": "2015", "start": 77, "end": 81, "label": "DATE"}
    ]
    
    # Test fast version
    print("\n--- Testing Fast ReLiK ---")
    start = time.time()
    
    try:
        fast_relik = FastReLiK(device="cpu")
        load_time = time.time() - start
        
        start = time.time()
        result = fast_relik(text, entities)
        inference_time = time.time() - start
        
        print(f"✅ Load time: {load_time:.2f}s")
        print(f"✅ Inference time: {inference_time:.3f}s")
        print(f"✅ Total time: {load_time + inference_time:.2f}s")
        
        if hasattr(result, 'triplets'):
            print(f"✅ Relations found: {len(result.triplets)}")
            for i, triplet in enumerate(result.triplets[:5]):
                print(f"   {i+1}. {triplet}")
        
    except Exception as e:
        print(f"❌ Fast ReLiK test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare with original (if we have time)
    print("\n--- Testing Original ReLiK (for comparison) ---")
    start = time.time()
    
    try:
        from relik import Relik
        original_relik = Relik.from_pretrained(
            "relik-ie/relik-relation-extraction-small", 
            device="cpu"
        )
        original_load_time = time.time() - start
        
        start = time.time()
        original_result = original_relik(text)
        original_inference_time = time.time() - start
        
        print(f"⚠️  Original load time: {original_load_time:.2f}s")
        print(f"⚠️  Original inference: {original_inference_time:.2f}s")
        print(f"⚠️  Speedup: {(original_load_time + original_inference_time) / (load_time + inference_time):.1f}x")
        
    except Exception as e:
        print(f"❌ Original ReLiK test failed: {e}")

if __name__ == "__main__":
    test_fast_relik()