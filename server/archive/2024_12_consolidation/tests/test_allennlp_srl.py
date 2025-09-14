#!/usr/bin/env python3
"""
Test AllenNLP SRL Predictor (Production Ready)
==============================================

AllenNLP has a built-in SRL predictor that should work out of the box.
This is likely the most production-ready English SRL solution.
"""

try:
    from allennlp.predictors.predictor import Predictor
    import time

    def test_allennlp_srl():
        """Test AllenNLP's built-in SRL predictor"""

        test_sentences = [
            "John gave Mary a book yesterday.",
            "My name is Alex Thompson.",
            "The company announced profits.",
            "Maria bought a car in Madrid last week.",
            "The president holds a lot of power."
        ]

        print("🚀 Testing AllenNLP SRL Predictor")
        print("=" * 50)

        try:
            start_time = time.time()

            # Load the pre-trained SRL model
            print("Loading AllenNLP SRL predictor...")
            predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
            )

            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")

            # Test sentences
            for sentence in test_sentences:
                print(f"\n📝 Sentence: {sentence}")

                start_time = time.time()
                result = predictor.predict(sentence=sentence)
                inference_time = time.time() - start_time

                print(f"⏱️  Inference time: {inference_time*1000:.1f}ms")
                print("📊 SRL Results:")

                # Parse the results
                words = result["words"]
                verbs = result["verbs"]

                if verbs:
                    for i, verb_info in enumerate(verbs):
                        verb = verb_info["verb"]
                        tags = verb_info["tags"]

                        print(f"   🎯 Predicate {i+1}: '{verb}'")

                        # Extract arguments
                        current_arg = None
                        current_words = []

                        for word, tag in zip(words, tags):
                            if tag.startswith('B-'):
                                # Start of new argument
                                if current_arg and current_words:
                                    print(f"      • {current_arg}: {' '.join(current_words)}")

                                current_arg = tag[2:]  # Remove 'B-'
                                current_words = [word]

                            elif tag.startswith('I-') and current_arg:
                                # Continuation of argument
                                current_words.append(word)

                            else:
                                # End of argument or O tag
                                if current_arg and current_words:
                                    print(f"      • {current_arg}: {' '.join(current_words)}")
                                    current_arg = None
                                    current_words = []

                        # Handle last argument
                        if current_arg and current_words:
                            print(f"      • {current_arg}: {' '.join(current_words)}")

                        print()
                else:
                    print("   • No predicates found")

                print("-" * 40)

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        test_allennlp_srl()

except ImportError:
    print("❌ AllenNLP not available. Install with: pip install allennlp")
    print("\nLet me try alternative approach...")

    # Alternative: try to use the model directly
    def test_alternative():
        print("Testing alternative approach...")

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            from transformers import pipeline

            # Try a known working model
            model_name = "microsoft/DialoGPT-medium"  # Just a test
            print("Testing transformers pipeline...")

        except Exception as e:
            print(f"Alternative also failed: {e}")

    test_alternative()