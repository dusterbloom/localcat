"""
Simple Intent Classifier for LocalCat Voice Agent
Using Hugging Face zero-shot classification - no training required!
Target: <50ms inference time
"""

from transformers import pipeline
import time
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class SimpleIntentClassifier:
    """
    Zero-shot intent classifier using pre-trained BART model
    No training required - works out of the box!
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None

        # Define intent categories based on the research doc
        self.intent_labels = [
            "remember information",
            "recall information",
            "forget information",
            "check memory",
            "general conversation",
            "ask for clarification",
            "make correction",
            "continue conversation",
            "ask about capabilities",
            "greeting",
            "goodbye",
            "agree or affirm",
            "disagree or deny"
        ]

        # Map detailed labels back to simple intent names
        self.label_to_intent = {
            "remember information": "remember_fact",
            "recall information": "recall_query",
            "forget information": "forget_request",
            "check memory": "memory_check",
            "general conversation": "general_chat",
            "ask for clarification": "clarification",
            "make correction": "correction",
            "continue conversation": "continuation",
            "ask about capabilities": "capability_query",
            "greeting": "greeting",
            "goodbye": "goodbye",
            "agree or affirm": "affirmation",
            "disagree or deny": "negation"
        }

        # Performance tracking
        self.inference_times = []

    def initialize(self):
        """Initialize the zero-shot classification pipeline"""
        logger.info(f"Initializing zero-shot classifier with {self.model_name}")
        start_time = time.time()

        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=0 if self._has_gpu() else -1  # Use GPU if available
        )

        init_time = (time.time() - start_time) * 1000
        logger.info(f"Classifier initialized in {init_time:.2f}ms")

    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except:
            return False

    def predict(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Predict intent for given text using zero-shot classification
        Returns: {"intent": str, "confidence": float, "inference_time_ms": float}
        """
        if self.classifier is None:
            self.initialize()

        start_time = time.time()

        # Perform zero-shot classification
        result = self.classifier(text, self.intent_labels)

        # Get top prediction
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        # Map to our intent system
        intent_name = self.label_to_intent.get(top_label, "general_chat")

        inference_time_ms = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time_ms)

        # Log performance every 50 predictions
        if len(self.inference_times) % 50 == 0:
            avg_time = np.mean(self.inference_times[-50:])
            logger.info(f"Average inference time (last 50): {avg_time:.2f}ms")

        prediction = {
            "intent": intent_name,
            "confidence": top_score,
            "inference_time_ms": inference_time_ms,
            "raw_result": result,  # Include full results for debugging
            "low_confidence": top_score < confidence_threshold
        }

        if top_score < confidence_threshold:
            logger.debug(f"Low confidence prediction: {intent_name} ({top_score:.3f})")

        return prediction

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
            "negation": "minimal"
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

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Predict intents for multiple texts"""
        return [self.predict(text) for text in texts]


# Convenience function for easy integration
def create_intent_classifier(model_name: Optional[str] = None) -> SimpleIntentClassifier:
    """Create and initialize intent classifier"""
    classifier = SimpleIntentClassifier(model_name or "facebook/bart-large-mnli")
    classifier.initialize()
    return classifier


if __name__ == "__main__":
    # Quick test
    print("Testing Simple Intent Classifier...")
    print("-" * 50)

    classifier = create_intent_classifier()

    test_texts = [
        "Remember that I like coffee",
        "What did I tell you about my job?",
        "Hello how are you today?",
        "What can you do for me?",
        "That's not right, let me correct you",
        "Yes, that's correct",
        "No, that's wrong",
        "Tell me more about that",
        "Goodbye",
        "Forget what I said about pizza"
    ]

    for text in test_texts:
        result = classifier.predict(text)
        print(f"Text: {text}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print(f"Strategy: {classifier.get_memory_processing_strategy(result['intent'])}")
        print(f"Inference time: {result['inference_time_ms']:.2f}ms")
        print()

    print("Performance Stats:")
    stats = classifier.get_performance_stats()
    print(f"Average inference: {stats['avg_inference_ms']:.2f}ms")
    print(f"P95 inference: {stats['p95_inference_ms']:.2f}ms")