#!/usr/bin/env python3
"""
Test CogComp SRL Model from HuggingFace
======================================

Testing the Yuqian/Celine_SRL model which should be a working SRL system.
"""

import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def test_cogcomp_srl():
    """Test CogComp SRL model from HuggingFace"""

    model_name = "Yuqian/Celine_SRL"

    test_sentences = [
        "John gave Mary a book yesterday.",
        "My name is Alex Thompson.",
        "The company announced profits.",
        "Maria bought a car in Madrid last week.",
        "The president holds a lot of power."
    ]

    print(f"🚀 Testing CogComp SRL: {model_name}")
    print("=" * 60)

    try:
        start_time = time.time()

        # Load tokenizer and model
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Create pipeline
        srl_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")

        # Test sentences
        for sentence in test_sentences:
            print(f"\n📝 Sentence: {sentence}")

            start_time = time.time()
            results = srl_pipeline(sentence)
            inference_time = time.time() - start_time

            print(f"⏱️  Inference time: {inference_time*1000:.1f}ms")
            print("📊 SRL Results:")

            if results:
                for result in results[:10]:  # Show first 10 results
                    print(f"   • {result['word']}: {result['entity_group']} (score: {result['score']:.3f})")
                if len(results) > 10:
                    print(f"   ... and {len(results) - 10} more results")
            else:
                print("   • No results")

            print("-" * 40)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cogcomp_srl()