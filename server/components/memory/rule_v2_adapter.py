"""
Adapter to integrate Enhanced Rule Classifier V2 with the memory system
"""

import os
from typing import Optional, List
from components.memory.enhanced_rule_classifier_v2 import (
    EnhancedRuleClassifierV2,
    IntentClassification as V2Classification,
    IntentType as V2IntentType
)
from components.memory.memory_intent import IntentClassifier, IntentAnalysis, IntentType


class RuleV2Adapter(IntentClassifier):
    """Adapter to use Enhanced Rule Classifier V2 in the memory system"""

    def __init__(self):
        super().__init__()
        self.classifier = EnhancedRuleClassifierV2()

        # Map V2 intent types to memory system intent types
        self.intent_mapping = {
            V2IntentType.QUESTION: IntentType.PURE_QUESTION,  # Questions need retrieval
            V2IntentType.FACT: IntentType.FACT_STATEMENT,
            V2IntentType.GREETING: IntentType.REACTION,
            V2IntentType.ACKNOWLEDGMENT: IntentType.REACTION,
            V2IntentType.REACTION: IntentType.REACTION,
            V2IntentType.CORRECTION: IntentType.CORRECTION,
            V2IntentType.COMMAND: IntentType.PURE_QUESTION,  # Commands often need retrieval
            V2IntentType.REQUEST: IntentType.PURE_QUESTION,  # Requests need retrieval
            V2IntentType.FAREWELL: IntentType.REACTION,
            V2IntentType.UNKNOWN: IntentType.REACTION,  # Safe default
        }

    def classify(self, text: str, context: Optional[List[str]] = None) -> IntentAnalysis:
        """Classify intent using Enhanced Rule V2"""
        # Get V2 classification
        v2_result = self.classifier.classify(text, context)

        # Map to memory system intent type, default to REACTION if not found
        mapped_intent = self.intent_mapping.get(
            v2_result.primary_intent,
            IntentType.REACTION  # Safe fallback
        )

        # Create IntentAnalysis result
        return IntentAnalysis(
            intent=mapped_intent,  # Fixed: field is 'intent' not 'primary_intent'
            confidence=v2_result.confidence,
            should_extract_facts=v2_result.requires_memory,
            embedded_facts_likely=v2_result.requires_memory,
            temporal_markers=[],
            correction_signals=[],
            requires_memory=v2_result.requires_memory,
            requires_retrieval=v2_result.requires_retrieval
        )

    def should_retrieve_memory(self, intent: IntentAnalysis) -> bool:
        """Determine if memory retrieval is needed"""
        return intent.requires_retrieval

    def should_store_memory(self, intent: IntentAnalysis) -> bool:
        """Determine if information should be stored"""
        return intent.requires_memory


def get_rule_v2_classifier() -> IntentClassifier:
    """Factory function to get Rule V2 classifier"""
    return RuleV2Adapter()


# Enable via environment variable
def should_use_rule_v2() -> bool:
    """Check if Rule V2 should be used"""
    return os.getenv("USE_RULE_V2_CLASSIFIER", "false").lower() in ("true", "1", "yes")