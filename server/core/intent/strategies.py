"""
Intent Strategies Configuration
Centralized strategy mappings to eliminate DRY violations and provide single source of truth
"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass


class MemoryProcessingStrategy(Enum):
    """Memory processing strategies for different intent types"""
    STORAGE_FOCUSED = "storage_focused"
    RETRIEVAL_FOCUSED = "retrieval_focused"
    DELETION_FOCUSED = "deletion_focused"
    LOOKUP_FOCUSED = "lookup_focused"
    MINIMAL = "minimal"
    CONTEXTUAL = "contextual"
    RECENT_CONTEXT = "recent_context"
    SKIP = "skip"
    STANDARD = "standard"


class IntentCategory(Enum):
    """Intent categories for grouping and routing"""
    MEMORY_OPERATIONS = "memory_operations"
    CONVERSATIONAL = "conversational"
    CLARIFICATION = "clarification"
    CAPABILITY_QUERIES = "capability_queries"
    SKIP_MEMORY = "skip_memory"


@dataclass(frozen=True)
class IntentDefinition:
    """Complete definition of an intent with its properties"""
    name: str
    category: IntentCategory
    strategy: MemoryProcessingStrategy
    skip_memory: bool
    description: str


class IntentStrategies:
    """
    Centralized intent strategy configuration
    Single source of truth for all intent routing logic
    """

    # Core intent definitions
    INTENT_DEFINITIONS = [
        # Memory operation intents
        IntentDefinition(
            name="remember_fact",
            category=IntentCategory.MEMORY_OPERATIONS,
            strategy=MemoryProcessingStrategy.STORAGE_FOCUSED,
            skip_memory=False,
            description="Store new information or facts"
        ),
        IntentDefinition(
            name="recall_query",
            category=IntentCategory.MEMORY_OPERATIONS,
            strategy=MemoryProcessingStrategy.RETRIEVAL_FOCUSED,
            skip_memory=False,
            description="Retrieve stored information"
        ),
        IntentDefinition(
            name="forget_request",
            category=IntentCategory.MEMORY_OPERATIONS,
            strategy=MemoryProcessingStrategy.DELETION_FOCUSED,
            skip_memory=False,
            description="Delete or forget information"
        ),
        IntentDefinition(
            name="memory_check",
            category=IntentCategory.MEMORY_OPERATIONS,
            strategy=MemoryProcessingStrategy.LOOKUP_FOCUSED,
            skip_memory=False,
            description="Check if information is remembered"
        ),

        # Conversational intents
        IntentDefinition(
            name="general_chat",
            category=IntentCategory.CONVERSATIONAL,
            strategy=MemoryProcessingStrategy.MINIMAL,
            skip_memory=True,
            description="General conversation and casual chat"
        ),
        IntentDefinition(
            name="greeting",
            category=IntentCategory.CONVERSATIONAL,
            strategy=MemoryProcessingStrategy.SKIP,
            skip_memory=True,
            description="Greetings and conversation starters"
        ),
        IntentDefinition(
            name="goodbye",
            category=IntentCategory.CONVERSATIONAL,
            strategy=MemoryProcessingStrategy.SKIP,
            skip_memory=True,
            description="Farewell and conversation endings"
        ),
        IntentDefinition(
            name="affirmation",
            category=IntentCategory.CONVERSATIONAL,
            strategy=MemoryProcessingStrategy.MINIMAL,
            skip_memory=True,
            description="Agreement and confirmation"
        ),
        IntentDefinition(
            name="negation",
            category=IntentCategory.CONVERSATIONAL,
            strategy=MemoryProcessingStrategy.MINIMAL,
            skip_memory=True,
            description="Disagreement and denial"
        ),

        # Clarification intents
        IntentDefinition(
            name="clarification",
            category=IntentCategory.CLARIFICATION,
            strategy=MemoryProcessingStrategy.CONTEXTUAL,
            skip_memory=False,
            description="Requests for clarification or explanation"
        ),
        IntentDefinition(
            name="correction",
            category=IntentCategory.CLARIFICATION,
            strategy=MemoryProcessingStrategy.RECENT_CONTEXT,
            skip_memory=False,
            description="Corrections to previous statements"
        ),
        IntentDefinition(
            name="continuation",
            category=IntentCategory.CLARIFICATION,
            strategy=MemoryProcessingStrategy.CONTEXTUAL,
            skip_memory=False,
            description="Requests to continue conversation"
        ),

        # Capability queries
        IntentDefinition(
            name="capability_query",
            category=IntentCategory.CAPABILITY_QUERIES,
            strategy=MemoryProcessingStrategy.STANDARD,
            skip_memory=False,
            description="Questions about system capabilities"
        ),
    ]

    # Model-specific label mappings
    MODEL_LABEL_MAPPINGS = {
        "Falconsai/intent_classification": {
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
        },
        "kousik-2310/intent-classifier-minilm": {
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
    }

    def __init__(self):
        """Initialize strategy configuration with computed mappings"""
        # Create lookup dictionaries for fast access
        self._intent_to_definition = {defn.name: defn for defn in self.INTENT_DEFINITIONS}
        self._category_to_intents = self._build_category_mapping()

    def _build_category_mapping(self) -> Dict[IntentCategory, List[str]]:
        """Build mapping from categories to intent lists"""
        mapping = {}
        for category in IntentCategory:
            mapping[category] = [
                defn.name for defn in self.INTENT_DEFINITIONS
                if defn.category == category
            ]
        return mapping

    def get_strategy(self, intent: str) -> MemoryProcessingStrategy:
        """Get memory processing strategy for an intent"""
        definition = self._intent_to_definition.get(intent)
        if definition:
            return definition.strategy
        return MemoryProcessingStrategy.STANDARD

    def should_skip_memory(self, intent: str) -> bool:
        """Check if intent should skip memory processing"""
        definition = self._intent_to_definition.get(intent)
        if definition:
            return definition.skip_memory
        return False

    def get_category(self, intent: str) -> IntentCategory:
        """Get category for an intent"""
        definition = self._intent_to_definition.get(intent)
        if definition:
            return definition.category
        return IntentCategory.CONVERSATIONAL  # Default fallback

    def get_intents_by_category(self, category: IntentCategory) -> List[str]:
        """Get all intents in a specific category"""
        return self._category_to_intents.get(category, [])

    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all categories as string-keyed dictionary for backward compatibility"""
        return {
            category.value: intents
            for category, intents in self._category_to_intents.items()
        }

    def get_skip_memory_intents(self) -> Set[str]:
        """Get set of intents that should skip memory processing"""
        return {
            defn.name for defn in self.INTENT_DEFINITIONS
            if defn.skip_memory
        }

    def get_model_label_mapping(self, model_name: str) -> Dict[str, str]:
        """Get label mapping for a specific model"""
        # Find mapping by checking if model_name contains any key
        for model_key, mapping in self.MODEL_LABEL_MAPPINGS.items():
            if model_key in model_name or any(part in model_name for part in model_key.split('/')):
                return mapping
        return {}

    def validate_intent(self, intent: str) -> bool:
        """Validate if an intent is known to the system"""
        return intent in self._intent_to_definition

    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of an intent"""
        definition = self._intent_to_definition.get(intent)
        if definition:
            return definition.description
        return f"Unknown intent: {intent}"

    def add_custom_intent(self, definition: IntentDefinition) -> None:
        """Add a custom intent definition (for extensibility)"""
        self._intent_to_definition[definition.name] = definition
        self._category_to_intents = self._build_category_mapping()

    def get_strategy_summary(self) -> Dict[str, str]:
        """Get summary of all intents and their strategies"""
        return {
            intent: strategy.value
            for intent, strategy in [(defn.name, defn.strategy) for defn in self.INTENT_DEFINITIONS]
        }


# Singleton instance for global access
_strategies_instance = None

def get_intent_strategies() -> IntentStrategies:
    """Get or create the global intent strategies instance"""
    global _strategies_instance
    if _strategies_instance is None:
        _strategies_instance = IntentStrategies()
    return _strategies_instance


# Convenience functions for backward compatibility
def get_memory_processing_strategy(intent: str) -> str:
    """Get memory processing strategy as string (backward compatible)"""
    return get_intent_strategies().get_strategy(intent).value

def should_skip_memory_processing(intent: str) -> bool:
    """Check if intent should skip memory processing (backward compatible)"""
    return get_intent_strategies().should_skip_memory(intent)

def get_intent_categories() -> Dict[str, List[str]]:
    """Get intent categories (backward compatible)"""
    return get_intent_strategies().get_all_categories()


if __name__ == "__main__":
    # Quick test of the strategies configuration
    strategies = get_intent_strategies()

    print("Intent Strategies Test")
    print("=" * 40)

    test_intents = ["remember_fact", "greeting", "general_chat", "recall_query"]

    for intent in test_intents:
        strategy = strategies.get_strategy(intent)
        skip = strategies.should_skip_memory(intent)
        category = strategies.get_category(intent)

        print(f"{intent:15} → {strategy.value:15} (skip: {skip}, category: {category.value})")

    print("\nCategories:")
    for category, intents in strategies.get_all_categories().items():
        print(f"{category}: {intents}")