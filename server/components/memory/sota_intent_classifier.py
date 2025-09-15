"""
SOTA 2025 Intent Classification System
Using transformer-based models for high-accuracy intent classification
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Enhanced intent types for 2025 SOTA classification"""
    # Information Seeking
    QUESTION_PURE = "question_pure"  # "What is X?"
    QUESTION_WITH_CONTEXT = "question_with_context"  # "Did I tell you about X?"
    CLARIFICATION = "clarification"  # "What do you mean by X?"

    # Information Providing
    FACT_STATEMENT = "fact_statement"  # "My name is X"
    CORRECTION = "correction"  # "No, actually it's X"
    TEMPORAL_FACT = "temporal_fact"  # "Yesterday I did X"

    # Task-Oriented
    COMMAND = "command"  # "Show me X", "Calculate Y"
    REQUEST = "request"  # "Can you help with X?"

    # Conversational
    GREETING = "greeting"  # "Hello", "Good morning"
    FAREWELL = "farewell"  # "Goodbye", "See you later"
    ACKNOWLEDGMENT = "acknowledgment"  # "OK", "Got it"
    REACTION = "reaction"  # "Wow", "That's interesting"

    # Complex
    HYPOTHETICAL = "hypothetical"  # "What if X?"
    MULTIPLE_INTENTS = "multiple_intents"  # Contains multiple intent types
    UNKNOWN = "unknown"  # Cannot classify

@dataclass
class IntentClassification:
    """Result of intent classification"""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[Tuple[IntentType, float]]
    requires_memory: bool
    requires_retrieval: bool
    metadata: Dict

from components.memory.memory_interfaces import IIntentClassifier

class SOTAIntentClassifier(IIntentClassifier):
    """State-of-the-art intent classifier using transformer models"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize SOTA intent classifier

        Args:
            model_name: HuggingFace model name or path to local model
        """
        self.model_name = model_name or os.getenv(
            "INTENT_CLASSIFIER_MODEL",
            "typeform/distilbert-base-uncased-mnli"  # DistilBERT - much faster
        )

        # Intent to retrieval mapping
        self.retrieval_required = {
            IntentType.QUESTION_PURE: True,
            IntentType.QUESTION_WITH_CONTEXT: True,
            IntentType.CLARIFICATION: True,
            IntentType.CORRECTION: False,  # Store but don't retrieve
            IntentType.FACT_STATEMENT: False,
            IntentType.TEMPORAL_FACT: False,
            IntentType.COMMAND: True,  # May need context
            IntentType.REQUEST: True,
            IntentType.GREETING: False,
            IntentType.FAREWELL: False,
            IntentType.ACKNOWLEDGMENT: False,
            IntentType.REACTION: False,
            IntentType.HYPOTHETICAL: True,  # Needs context for reasoning
            IntentType.MULTIPLE_INTENTS: True,
            IntentType.UNKNOWN: True,  # Safe default
        }

        # Intent to storage mapping
        self.storage_required = {
            IntentType.FACT_STATEMENT: True,
            IntentType.TEMPORAL_FACT: True,
            IntentType.CORRECTION: True,
            IntentType.QUESTION_WITH_CONTEXT: True,  # May contain facts
            IntentType.MULTIPLE_INTENTS: True,
            # All others default to False
        }

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the transformer model for classification with optional quantization"""
        try:
            # Check if INT8 quantization is requested
            use_int8 = os.getenv("INTENT_CLASSIFIER_USE_INT8", "true").lower() in ("true", "1", "yes")

            # Determine best available device
            if torch.cuda.is_available():
                device = 0  # CUDA GPU
                device_name = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                device_name = "mps"
            else:
                device = -1  # CPU
                device_name = "cpu"

            logger.info(f"Initializing SOTA classifier on device: {device_name}")

            if use_int8 and device_name in ["cpu", "mps"]:
                logger.info("Loading model with INT8 quantization for faster inference...")

                # Load model and tokenizer separately for quantization
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

                # Apply dynamic quantization for CPU/MPS
                import torch.quantization as quantization
                model = quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},  # Quantize Linear layers
                    dtype=torch.qint8
                )

                # Create pipeline with quantized model
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # Quantized models run on CPU
                )
                logger.info("✅ Model quantized to INT8 for ~2-4x speedup")
            else:
                # Standard loading without quantization
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=device
                )

            # Define candidate labels for zero-shot classification
            self.candidate_labels = [
                "question",
                "statement of fact",
                "correction",
                "command or request",
                "greeting",
                "reaction or acknowledgment",
                "hypothetical scenario",
                "temporal information",
                "clarification request"
            ]

            logger.info(f"Initialized SOTA intent classifier: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize SOTA classifier: {e}")
            logger.info("Falling back to rule-based classification")
            self.classifier = None

    def classify(self, text: str, context: Optional[List[str]] = None) -> IntentClassification:
        """
        Classify the intent of the given text

        Args:
            text: The text to classify
            context: Optional conversation context

        Returns:
            IntentClassification with detailed results
        """
        start_time = time.time()

        # Use transformer model if available
        if self.classifier:
            result = self._classify_with_transformer(text, context)
        else:
            # Fallback to enhanced rule-based
            result = self._classify_with_rules(text)

        # Determine memory operations needed
        result.requires_retrieval = self.retrieval_required.get(
            result.primary_intent, True
        )
        result.requires_memory = self.storage_required.get(
            result.primary_intent, False
        )

        # Add performance metrics
        result.metadata['classification_time_ms'] = (time.time() - start_time) * 1000

        return result

    def _classify_with_transformer(
        self,
        text: str,
        context: Optional[List[str]] = None
    ) -> IntentClassification:
        """Classify using transformer model with zero-shot learning"""

        # Add context if provided
        input_text = text
        if context:
            # Include last 2 messages for context
            context_str = " [CONTEXT] ".join(context[-2:])
            input_text = f"{context_str} [CURRENT] {text}"

        # Run zero-shot classification
        result = self.classifier(
            input_text,
            candidate_labels=self.candidate_labels,
            multi_label=True  # Allow multiple intents
        )

        # Map results to intent types
        intent_map = {
            "question": IntentType.QUESTION_PURE,
            "statement of fact": IntentType.FACT_STATEMENT,
            "correction": IntentType.CORRECTION,
            "command or request": IntentType.COMMAND,
            "greeting": IntentType.GREETING,
            "reaction or acknowledgment": IntentType.REACTION,
            "hypothetical scenario": IntentType.HYPOTHETICAL,
            "temporal information": IntentType.TEMPORAL_FACT,
            "clarification request": IntentType.CLARIFICATION,
        }

        # Get primary intent
        primary_label = result['labels'][0]
        primary_intent = intent_map.get(primary_label, IntentType.UNKNOWN)
        primary_confidence = result['scores'][0]

        # Get secondary intents (if confidence > 0.2)
        secondary_intents = []
        for label, score in zip(result['labels'][1:], result['scores'][1:]):
            if score > 0.2:
                intent = intent_map.get(label, IntentType.UNKNOWN)
                secondary_intents.append((intent, score))

        # Check for multiple intents
        if len(secondary_intents) > 0 and primary_confidence < 0.6:
            primary_intent = IntentType.MULTIPLE_INTENTS

        # Special handling for questions with context
        if primary_intent == IntentType.QUESTION_PURE:
            if any(phrase in text.lower() for phrase in [
                "did i", "have i", "was i", "remember when", "told you"
            ]):
                primary_intent = IntentType.QUESTION_WITH_CONTEXT

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            secondary_intents=secondary_intents,
            requires_memory=False,  # Set later
            requires_retrieval=False,  # Set later
            metadata={
                'model': self.model_name,
                'method': 'transformer',
                'raw_scores': dict(zip(result['labels'], result['scores']))
            }
        )

    def _classify_with_rules(self, text: str) -> IntentClassification:
        """Enhanced rule-based classification as fallback"""
        text_lower = text.lower().strip()

        # Check for greetings
        if any(g in text_lower for g in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return IntentClassification(
                primary_intent=IntentType.GREETING,
                confidence=0.9,
                secondary_intents=[],
                requires_memory=False,
                requires_retrieval=False,
                metadata={'method': 'rules'}
            )

        # Check for questions
        if text_lower.endswith('?') or text_lower.startswith(('what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should')):
            # Check if it's a question about past information
            if any(phrase in text_lower for phrase in ['did i', 'have i', 'was i', 'told you', 'remember']):
                intent = IntentType.QUESTION_WITH_CONTEXT
            else:
                intent = IntentType.QUESTION_PURE

            return IntentClassification(
                primary_intent=intent,
                confidence=0.8,
                secondary_intents=[],
                requires_memory=False,
                requires_retrieval=True,
                metadata={'method': 'rules'}
            )

        # Check for corrections
        if any(c in text_lower for c in ['no,', 'actually', 'correction:', 'i meant', 'not ']):
            return IntentClassification(
                primary_intent=IntentType.CORRECTION,
                confidence=0.85,
                secondary_intents=[],
                requires_memory=True,
                requires_retrieval=False,
                metadata={'method': 'rules'}
            )

        # Check for temporal facts
        if any(t in text_lower for t in ['yesterday', 'today', 'tomorrow', 'last week', 'next', 'ago']):
            return IntentClassification(
                primary_intent=IntentType.TEMPORAL_FACT,
                confidence=0.75,
                secondary_intents=[(IntentType.FACT_STATEMENT, 0.5)],
                requires_memory=True,
                requires_retrieval=False,
                metadata={'method': 'rules'}
            )

        # Default to fact statement for declarative sentences
        if '.' in text or len(text.split()) > 3:
            return IntentClassification(
                primary_intent=IntentType.FACT_STATEMENT,
                confidence=0.6,
                secondary_intents=[],
                requires_memory=True,
                requires_retrieval=False,
                metadata={'method': 'rules'}
            )

        # Short responses are likely reactions
        if len(text.split()) <= 3:
            return IntentClassification(
                primary_intent=IntentType.REACTION,
                confidence=0.7,
                secondary_intents=[],
                requires_memory=False,
                requires_retrieval=False,
                metadata={'method': 'rules'}
            )

        # Unknown
        return IntentClassification(
            primary_intent=IntentType.UNKNOWN,
            confidence=0.3,
            secondary_intents=[],
            requires_memory=False,
            requires_retrieval=True,  # Safe default
            metadata={'method': 'rules'}
        )

    def should_retrieve_memory(self, intent: IntentClassification) -> bool:
        """Determine if memory retrieval is needed for this intent"""
        return intent.requires_retrieval

    def should_store_memory(self, intent: IntentClassification) -> bool:
        """Determine if this intent contains information to store"""
        return intent.requires_memory

    def get_intent_metrics(self) -> Dict:
        """Get performance metrics for the classifier"""
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        return {
            'model': self.model_name,
            'device': device,
            'retrieval_intents': [k.value for k, v in self.retrieval_required.items() if v],
            'storage_intents': [k.value for k, v in self.storage_required.items() if v],
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = SOTAIntentClassifier()

    # Test cases
    test_cases = [
        "What is the capital of France?",
        "My dog's name is Potola",
        "Did I tell you about my vacation?",
        "No, actually her name is Sarah",
        "Hello, how are you?",
        "Calculate the square root of 144",
        "Wow, that's amazing!",
        "Yesterday I went to the park",
        "What do you mean by that?",
        "Remember when we discussed the project?"
    ]

    print("SOTA Intent Classification Results:")
    print("-" * 50)

    for text in test_cases:
        result = classifier.classify(text)
        print(f"Text: {text}")
        print(f"  Intent: {result.primary_intent.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Retrieve: {result.requires_retrieval}")
        print(f"  Store: {result.requires_memory}")
        print(f"  Time: {result.metadata.get('classification_time_ms', 0):.1f}ms")
        print()