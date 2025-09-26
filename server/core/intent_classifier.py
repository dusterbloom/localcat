"""
DIET Intent Classifier for LocalCat Voice Agent
Based on WeiNyn/DIETClassifier-pytorch implementation
Optimized for <20ms inference latency
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Union
import time
from loguru import logger


class DIETClassifierConfig:
    """Configuration for DIET Classifier"""

    def __init__(self, config_dict: Dict):
        self.model_name = config_dict.get("model_name", "distilbert-base-uncased")
        self.num_intent_labels = config_dict.get("num_intent_labels", 10)
        self.intent_labels = config_dict.get("intent_labels", [])
        self.dropout = config_dict.get("dropout", 0.1)
        self.intent_loss_weight = config_dict.get("intent_loss_weight", 1.0)
        self.device = config_dict.get("device", "cpu")


class DIETClassifier(nn.Module):
    """
    DIET (Dual Intent and Entity Transformer) Classifier
    Optimized for intent classification only (no entities for now)
    """

    def __init__(self, config: DIETClassifierConfig):
        super().__init__()
        self.config = config
        self.num_intent_labels = config.num_intent_labels

        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)

        # Intent classification head
        self.intent_classifier = nn.Linear(
            self.transformer.config.hidden_size,
            self.num_intent_labels
        )

        # Loss function
        self.intent_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, intent_labels=None):
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)

        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)

        result = {"intent_logits": intent_logits}

        # Calculate loss if labels provided
        if intent_labels is not None:
            intent_loss = self.intent_loss_fct(
                intent_logits.view(-1, self.num_intent_labels),
                intent_labels.view(-1)
            )
            result["loss"] = intent_loss * self.config.intent_loss_weight

        return result


class LocalCatIntentClassifier:
    """
    High-level wrapper for DIET classifier optimized for LocalCat voice agent
    Target: <20ms inference time
    """

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_dict is None:
            # Default config for LocalCat intents
            config_dict = self._get_default_config()

        self.config = DIETClassifierConfig(config_dict)
        self.model = None
        self.tokenizer = None
        self.intent_to_id = {}
        self.id_to_intent = {}

        # Performance tracking
        self.inference_times = []

    def _get_default_config(self) -> Dict:
        """Default configuration for LocalCat voice agent intents"""
        return {
            "model_name": "distilbert-base-uncased",  # Lightweight model
            "intent_labels": [
                "remember_fact",
                "recall_query",
                "forget_request",
                "memory_check",
                "general_chat",
                "clarification",
                "correction",
                "continuation",
                "capability_query",
                "greeting",
                "goodbye",
                "affirmation",
                "negation"
            ],
            "num_intent_labels": 13,
            "dropout": 0.1,
            "intent_loss_weight": 1.0,
            "device": "cpu"  # Start with CPU for compatibility
        }

    def initialize_model(self):
        """Initialize tokenizer and model"""
        logger.info(f"Initializing DIET classifier with {self.config.model_name}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Create intent mappings
        self.intent_to_id = {intent: i for i, intent in enumerate(self.config.intent_labels)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}

        # Initialize model
        self.model = DIETClassifier(self.config)
        self.model.eval()

        logger.info(f"Model initialized with {len(self.config.intent_labels)} intents")

    def predict(self, text: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Predict intent for given text
        Returns: {"intent": str, "confidence": float, "inference_time_ms": float}
        """
        if self.model is None:
            self.initialize_model()

        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128  # Keep short for speed
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        # Get predictions
        intent_logits = outputs["intent_logits"]
        intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)
        intent_confidence, intent_id = torch.max(intent_probs, dim=-1)

        intent_name = self.id_to_intent[intent_id.item()]
        confidence = intent_confidence.item()

        inference_time_ms = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time_ms)

        # Log performance every 100 predictions
        if len(self.inference_times) % 100 == 0:
            avg_time = np.mean(self.inference_times[-100:])
            logger.info(f"Average inference time (last 100): {avg_time:.2f}ms")

        result = {
            "intent": intent_name,
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
            "all_probabilities": {
                self.id_to_intent[i]: prob.item()
                for i, prob in enumerate(intent_probs[0])
            }
        }

        # Mark low confidence predictions
        if confidence < confidence_threshold:
            result["low_confidence"] = True
            logger.debug(f"Low confidence prediction: {intent_name} ({confidence:.3f})")

        return result

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
            "continuation": "contextual"
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
            "max_inference_ms": np.max(times)
        }


# Convenience function for easy integration
def create_intent_classifier(config_path: Optional[str] = None) -> LocalCatIntentClassifier:
    """Create and initialize intent classifier"""
    classifier = LocalCatIntentClassifier(config_path=config_path)
    classifier.initialize_model()
    return classifier


if __name__ == "__main__":
    # Quick test
    classifier = create_intent_classifier()

    test_texts = [
        "Remember that I like coffee",
        "What did you say about my job?",
        "Hello how are you today?",
        "What can you do for me?",
        "That's not right, let me correct you"
    ]

    print("Testing DIET Intent Classifier:")
    print("-" * 50)

    for text in test_texts:
        result = classifier.predict(text)
        print(f"Text: {text}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print(f"Inference time: {result['inference_time_ms']:.2f}ms")
        print()

    print("Performance Stats:")
    print(classifier.get_performance_stats())