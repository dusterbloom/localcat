"""
Fast Intent Classifier using lightweight pre-trained models
Target: <50ms inference time
"""

from transformers import pipeline
import time
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class FastIntentClassifier:
    """
    Ultra-fast intent classifier using lightweight pre-trained models
    """

    def __init__(self, model_name: str = "Falconsai/intent_classification"):
        self.model_name = model_name
        self.classifier = None
        self.inference_times = []

        # Map model outputs to our intent system
        self.intent_mapping = self._get_intent_mapping()

    def _get_intent_mapping(self) -> Dict[str, str]:
        """Map model-specific labels to our standardized intents"""
        if "Falconsai" in self.model_name:
            return {
                "speak to person": "general_chat",
                "greeting": "greeting",
                "goodbye": "goodbye",
                "affirmation": "affirmation",
                "negative": "negation",
                "book_flight": "general_chat",
                "book_hotel": "general_chat",
                "get_weather": "recall_query",
                "play_music": "general_chat",
                "translate": "general_chat",
                "search_news": "recall_query",
                "find_restaurant": "general_chat",
                "timer": "general_chat",
                "alarm": "general_chat",
                "email": "general_chat"
            }
        elif "minilm" in self.model_name:
            return {
                "get_inference": "general_chat",
                "greeting": "greeting",
                "goodbye": "goodbye",
                "affirmation": "affirmation",
                "negation": "negation",
                "question": "clarification",
                "request": "general_chat",
                "complaint": "correction",
                "compliment": "affirmation"
            }
        else:
            # Default mapping for unknown models
            return {}

    def initialize(self):
        """Initialize the classification pipeline"""
        logger.info(f"Initializing fast classifier with {self.model_name}")
        start_time = time.time()

        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if self._has_gpu() else -1
        )

        init_time = (time.time() - start_time) * 1000
        logger.info(f"Fast classifier initialized in {init_time:.2f}ms")

    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except:
            return False

    def predict(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Predict intent for given text
        Returns: {"intent": str, "confidence": float, "inference_time_ms": float}
        """
        if self.classifier is None:
            self.initialize()

        start_time = time.time()

        # Get prediction from model
        result = self.classifier(text)

        # Handle single result or list
        if isinstance(result, list):
            prediction = result[0]
        else:
            prediction = result

        model_label = prediction['label']
        confidence = prediction['score']

        # Map to our intent system
        intent = self.intent_mapping.get(model_label, self._guess_intent_from_text(text))

        inference_time_ms = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time_ms)

        # Log performance every 100 predictions
        if len(self.inference_times) % 100 == 0:
            avg_time = np.mean(self.inference_times[-100:])
            logger.info(f"Average inference time (last 100): {avg_time:.2f}ms")

        return {
            "intent": intent,
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
            "model_label": model_label,
            "low_confidence": confidence < confidence_threshold
        }

    def _guess_intent_from_text(self, text: str) -> str:
        """Simple rule-based fallback for unmapped labels"""
        text_lower = text.lower()

        # Memory-related keywords
        if any(word in text_lower for word in ["remember", "save", "store"]):
            return "remember_fact"
        elif any(word in text_lower for word in ["recall", "what did", "tell me about", "remind"]):
            return "recall_query"
        elif any(word in text_lower for word in ["forget", "delete", "remove"]):
            return "forget_request"
        elif any(word in text_lower for word in ["do you remember", "do you know"]):
            return "memory_check"

        # Conversational
        elif any(word in text_lower for word in ["hello", "hi", "hey", "good morning"]):
            return "greeting"
        elif any(word in text_lower for word in ["bye", "goodbye", "see you"]):
            return "goodbye"
        elif any(word in text_lower for word in ["yes", "yeah", "correct", "right"]):
            return "affirmation"
        elif any(word in text_lower for word in ["no", "nope", "wrong", "incorrect"]):
            return "negation"
        elif any(word in text_lower for word in ["what can you", "what do you", "help me"]):
            return "capability_query"
        elif any(word in text_lower for word in ["what", "why", "how", "explain"]):
            return "clarification"
        elif any(word in text_lower for word in ["actually", "correction", "fix that"]):
            return "correction"

        return "general_chat"

    def is_memory_intent(self, intent: str) -> bool:
        """Check if intent requires memory processing"""
        memory_intents = {
            "remember_fact", "recall_query", "forget_request", "memory_check"
        }
        return intent in memory_intents

    def get_memory_processing_strategy(self, intent: str) -> str:
        """Get memory processing strategy based on intent"""
        strategies = {
            "remember_fact": "store_focused",
            "recall_query": "retrieval_focused",
            "forget_request": "deletion_focused",
            "memory_check": "lookup_focused",
            "general_chat": "minimal",
            "clarification": "contextual",
            "correction": "recent_context",
            "continuation": "contextual",
            "greeting": "skip",
            "goodbye": "skip",
            "affirmation": "minimal",
            "negation": "minimal",
            "capability_query": "standard"
        }
        return strategies.get(intent, "standard")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {"status": "no_data"}

        times = np.array(self.inference_times)
        return {
            "total_predictions": len(times),
            "avg_inference_ms": np.mean(times),
            "p95_inference_ms": np.percentile(times, 95),
            "p99_inference_ms": np.percentile(times, 99),
            "min_inference_ms": np.min(times),
            "max_inference_ms": np.max(times),
            "model": self.model_name
        }


def benchmark_models():
    """Benchmark different lightweight models"""
    models_to_test = [
        "Falconsai/intent_classification",
        "kousik-2310/intent-classifier-minilm"
    ]

    test_texts = [
        "Hello how are you",
        "Remember that I like coffee",
        "What did I tell you about work?",
        "Goodbye",
        "Yes that's correct",
        "No that's wrong"
    ]

    results = {}

    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        classifier = FastIntentClassifier(model_name)
        classifier.initialize()

        model_times = []
        for text in test_texts:
            result = classifier.predict(text)
            model_times.append(result['inference_time_ms'])
            print(f"  '{text}' -> {result['intent']} ({result['confidence']:.3f}) - {result['inference_time_ms']:.2f}ms")

        avg_time = np.mean(model_times)
        results[model_name] = avg_time
        print(f"  Average: {avg_time:.2f}ms")

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS:")
    for model, avg_time in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model}: {avg_time:.2f}ms average")

    return results


if __name__ == "__main__":
    benchmark_models()