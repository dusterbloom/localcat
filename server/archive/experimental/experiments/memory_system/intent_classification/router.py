"""
Intent Router Component
Focused routing decisions based on intent classification results
"""

import os
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from loguru import logger

# Handle both package and direct execution imports
try:
    from .strategies import get_intent_strategies, MemoryProcessingStrategy, IntentCategory
    from .exceptions import RoutingDecisionError, InvalidIntentError
except ImportError:
    from strategies import get_intent_strategies, MemoryProcessingStrategy, IntentCategory
    from exceptions import RoutingDecisionError, InvalidIntentError


@dataclass
class RoutingDecision:
    """Complete routing decision with context"""
    intent: str
    strategy: MemoryProcessingStrategy
    skip_memory: bool
    category: IntentCategory
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class IntentRouter:
    """
    Focused component for making routing decisions based on intent classification
    Single responsibility: determine how to process an intent
    """

    def __init__(self, enable_logging: bool = None):
        """
        Initialize intent router

        Args:
            enable_logging: Whether to log routing decisions (defaults to env var)
        """
        self.strategies = get_intent_strategies()
        self.enable_logging = (
            enable_logging if enable_logging is not None
            else os.getenv("INTENT_LOG_ROUTING_DECISIONS", "false").lower() == "true"
        )

        # Configuration from environment
        self._load_routing_config()

        logger.debug(f"Intent router initialized with logging={'enabled' if self.enable_logging else 'disabled'}")

    def _load_routing_config(self) -> None:
        """Load routing configuration from environment variables"""
        # Custom skip intents from environment
        env_skip = os.getenv("INTENT_SKIP_MEMORY_FOR", "")
        if env_skip:
            self.custom_skip_intents = set(intent.strip() for intent in env_skip.split(',') if intent.strip())
        else:
            self.custom_skip_intents = set()

        # Enhanced retrieval intents
        env_enhanced_retrieval = os.getenv("INTENT_ENHANCED_RETRIEVAL_FOR", "")
        if env_enhanced_retrieval:
            self.enhanced_retrieval_intents = set(intent.strip() for intent in env_enhanced_retrieval.split(',') if intent.strip())
        else:
            self.enhanced_retrieval_intents = set()

        # Enhanced storage intents
        env_enhanced_storage = os.getenv("INTENT_ENHANCED_STORAGE_FOR", "")
        if env_enhanced_storage:
            self.enhanced_storage_intents = set(intent.strip() for intent in env_enhanced_storage.split(',') if intent.strip())
        else:
            self.enhanced_storage_intents = set()

    def make_routing_decision(self,
                            intent: str,
                            confidence: float,
                            text_context: Optional[str] = None,
                            fallback: bool = False) -> RoutingDecision:
        """
        Make a complete routing decision for an intent

        Args:
            intent: Classified intent
            confidence: Classification confidence
            text_context: Original text for context (optional)
            fallback: Whether this is a fallback classification

        Returns:
            Complete routing decision

        Raises:
            RoutingDecisionError: If routing cannot be determined
            InvalidIntentError: If intent is not recognized
        """
        try:
            # Validate intent
            if not self.strategies.validate_intent(intent):
                raise InvalidIntentError(intent, list(self.strategies._intent_to_definition.keys()))

            # Get base strategy from configuration
            base_strategy = self.strategies.get_strategy(intent)
            category = self.strategies.get_category(intent)
            base_skip_memory = self.strategies.should_skip_memory(intent)

            # Apply custom overrides
            final_strategy, final_skip_memory, reasoning = self._apply_routing_overrides(
                intent, base_strategy, base_skip_memory, confidence, fallback
            )

            # Create routing decision
            decision = RoutingDecision(
                intent=intent,
                strategy=final_strategy,
                skip_memory=final_skip_memory,
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'base_strategy': base_strategy.value,
                    'base_skip_memory': base_skip_memory,
                    'overridden': final_strategy != base_strategy or final_skip_memory != base_skip_memory,
                    'fallback': fallback,
                    'text_length': len(text_context) if text_context else 0,
                    'has_custom_config': bool(self.custom_skip_intents or self.enhanced_retrieval_intents or self.enhanced_storage_intents)
                }
            )

            # Log routing decision
            if self.enable_logging:
                self._log_routing_decision(decision, text_context)

            return decision

        except (InvalidIntentError, RoutingDecisionError):
            raise
        except Exception as e:
            raise RoutingDecisionError(intent, f"Unexpected error in routing: {e}")

    def _apply_routing_overrides(self,
                                intent: str,
                                base_strategy: MemoryProcessingStrategy,
                                base_skip_memory: bool,
                                confidence: float,
                                fallback: bool) -> tuple[MemoryProcessingStrategy, bool, str]:
        """
        Apply custom routing overrides based on configuration and context

        Returns:
            (final_strategy, final_skip_memory, reasoning)
        """
        reasoning_parts = []

        # Start with base configuration
        final_strategy = base_strategy
        final_skip_memory = base_skip_memory
        reasoning_parts.append(f"base strategy: {base_strategy.value}")

        # Apply custom skip override
        if intent in self.custom_skip_intents:
            final_skip_memory = True
            reasoning_parts.append("custom skip override")

        # Apply enhanced retrieval override
        if intent in self.enhanced_retrieval_intents and not final_skip_memory:
            final_strategy = MemoryProcessingStrategy.RETRIEVAL_FOCUSED
            reasoning_parts.append("enhanced retrieval override")

        # Apply enhanced storage override
        if intent in self.enhanced_storage_intents and not final_skip_memory:
            final_strategy = MemoryProcessingStrategy.STORAGE_FOCUSED
            reasoning_parts.append("enhanced storage override")

        # Apply confidence-based adjustments
        if confidence < 0.5 and not fallback:
            # Low confidence - use safer, more conservative processing
            if not final_skip_memory:
                final_strategy = MemoryProcessingStrategy.MINIMAL
                reasoning_parts.append("low confidence adjustment")

        # Apply fallback adjustments
        if fallback:
            final_skip_memory = True
            final_strategy = MemoryProcessingStrategy.SKIP
            reasoning_parts.append("fallback mode")

        return final_strategy, final_skip_memory, " + ".join(reasoning_parts)

    def _log_routing_decision(self, decision: RoutingDecision, text_context: Optional[str]) -> None:
        """Log routing decision with appropriate level"""
        context_preview = ""
        if text_context:
            context_preview = f" for '{text_context[:30]}...'" if len(text_context) > 30 else f" for '{text_context}'"

        if decision.skip_memory:
            logger.info(f"[Intent Routing] Skipping memory processing for intent: {decision.intent}{context_preview}")
        else:
            logger.debug(f"[Intent Routing] Strategy for '{decision.intent}': {decision.strategy.value}{context_preview}")

        # Log additional details for complex decisions
        if decision.metadata.get('overridden'):
            logger.debug(f"[Intent Routing] Decision overridden: {decision.reasoning}")

    def should_skip_memory_processing(self, intent: str) -> bool:
        """
        Simple check if intent should skip memory processing
        Convenience method for backward compatibility

        Args:
            intent: Intent to check

        Returns:
            True if memory processing should be skipped
        """
        try:
            # Use base strategy configuration
            base_skip = self.strategies.should_skip_memory(intent)

            # Apply custom overrides
            if intent in self.custom_skip_intents:
                return True

            return base_skip

        except Exception as e:
            logger.warning(f"Error checking skip memory for intent '{intent}': {e}")
            return False  # Safe default

    def get_memory_processing_strategy(self, intent: str) -> str:
        """
        Get memory processing strategy as string
        Convenience method for backward compatibility

        Args:
            intent: Intent to get strategy for

        Returns:
            Strategy name as string
        """
        try:
            # Create a basic routing decision
            decision = self.make_routing_decision(intent, 1.0)  # Assume high confidence
            return decision.strategy.value

        except Exception as e:
            logger.warning(f"Error getting strategy for intent '{intent}': {e}")
            return MemoryProcessingStrategy.STANDARD.value

    def get_intent_categories(self) -> Dict[str, List[str]]:
        """
        Get all intent categories
        Convenience method for backward compatibility

        Returns:
            Dictionary mapping category names to intent lists
        """
        return self.strategies.get_all_categories()

    def get_routing_summary(self) -> Dict[str, Any]:
        """Get summary of current routing configuration"""
        return {
            'total_intents': len(self.strategies._intent_to_definition),
            'categories': list(self.strategies.get_all_categories().keys()),
            'custom_skip_intents': list(self.custom_skip_intents),
            'enhanced_retrieval_intents': list(self.enhanced_retrieval_intents),
            'enhanced_storage_intents': list(self.enhanced_storage_intents),
            'logging_enabled': self.enable_logging,
            'skip_memory_count': len(self.strategies.get_skip_memory_intents()),
        }

    def validate_routing_config(self) -> List[str]:
        """
        Validate current routing configuration

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Check for unknown intents in custom configuration
        all_known_intents = set(self.strategies._intent_to_definition.keys())

        unknown_skip = self.custom_skip_intents - all_known_intents
        if unknown_skip:
            issues.append(f"Unknown intents in INTENT_SKIP_MEMORY_FOR: {unknown_skip}")

        unknown_retrieval = self.enhanced_retrieval_intents - all_known_intents
        if unknown_retrieval:
            issues.append(f"Unknown intents in INTENT_ENHANCED_RETRIEVAL_FOR: {unknown_retrieval}")

        unknown_storage = self.enhanced_storage_intents - all_known_intents
        if unknown_storage:
            issues.append(f"Unknown intents in INTENT_ENHANCED_STORAGE_FOR: {unknown_storage}")

        # Check for conflicting configurations
        skip_and_enhanced = self.custom_skip_intents & (self.enhanced_retrieval_intents | self.enhanced_storage_intents)
        if skip_and_enhanced:
            issues.append(f"Intents configured for both skip and enhanced processing: {skip_and_enhanced}")

        return issues


# Global router instance for easy access
_router_instance = None

def get_intent_router() -> IntentRouter:
    """Get or create the global intent router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = IntentRouter()
    return _router_instance


if __name__ == "__main__":
    # Test router functionality
    print("Testing Intent Router")
    print("=" * 30)

    router = IntentRouter(enable_logging=True)

    # Test routing decisions
    test_cases = [
        ("remember_fact", 0.95),
        ("general_chat", 0.85),
        ("greeting", 0.90),
        ("recall_query", 0.75),
        ("unknown_intent", 0.60),  # This should raise an error
    ]

    for intent, confidence in test_cases:
        try:
            decision = router.make_routing_decision(intent, confidence, f"Test text for {intent}")
            print(f"\n{intent} (conf: {confidence}):")
            print(f"  Strategy: {decision.strategy.value}")
            print(f"  Skip memory: {decision.skip_memory}")
            print(f"  Category: {decision.category.value}")
            print(f"  Reasoning: {decision.reasoning}")
        except Exception as e:
            print(f"\n{intent}: ERROR - {e}")

    print(f"\nRouting Summary:")
    summary = router.get_routing_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Test validation
    print(f"\nValidation Issues:")
    issues = router.validate_routing_config()
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No validation issues found")