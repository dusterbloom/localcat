"""
Intent Classification Exception Hierarchy
Specific exception types with context for better error handling
"""

from typing import Optional, Dict, Any


class IntentClassificationError(Exception):
    """Base exception for all intent classification errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} (context: {context_str})"
        return base_msg


class ModelLoadError(IntentClassificationError):
    """Raised when model loading or initialization fails"""

    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        context = {
            "model_name": model_name,
            "original_error": str(original_error) if original_error else None
        }
        message = f"Failed to load intent classification model: {model_name}"
        if original_error:
            message += f" ({type(original_error).__name__}: {original_error})"

        super().__init__(message, context)
        self.model_name = model_name
        self.original_error = original_error


class ClassificationTimeoutError(IntentClassificationError):
    """Raised when intent classification takes too long"""

    def __init__(self, text: str, timeout_ms: float):
        context = {
            "text_length": len(text),
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "timeout_ms": timeout_ms
        }
        message = f"Intent classification timed out after {timeout_ms}ms"
        super().__init__(message, context)
        self.text = text
        self.timeout_ms = timeout_ms


class LowConfidenceError(IntentClassificationError):
    """Raised when classification confidence is below threshold"""

    def __init__(self, text: str, confidence: float, threshold: float, intent: str):
        context = {
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "confidence": confidence,
            "threshold": threshold,
            "predicted_intent": intent
        }
        message = f"Low confidence classification: {confidence:.3f} < {threshold:.3f} for intent '{intent}'"
        super().__init__(message, context)
        self.text = text
        self.confidence = confidence
        self.threshold = threshold
        self.intent = intent


class CacheLookupError(IntentClassificationError):
    """Raised when cache operations fail"""

    def __init__(self, operation: str, key: str, original_error: Optional[Exception] = None):
        context = {
            "operation": operation,
            "cache_key": key,
            "original_error": str(original_error) if original_error else None
        }
        message = f"Cache {operation} failed for key: {key}"
        if original_error:
            message += f" ({type(original_error).__name__}: {original_error})"

        super().__init__(message, context)
        self.operation = operation
        self.key = key
        self.original_error = original_error


class InvalidIntentError(IntentClassificationError):
    """Raised when an unknown or invalid intent is encountered"""

    def __init__(self, intent: str, valid_intents: Optional[list] = None):
        context = {
            "invalid_intent": intent,
            "valid_intents_count": len(valid_intents) if valid_intents else None
        }
        message = f"Unknown intent: '{intent}'"
        if valid_intents:
            message += f" (valid intents: {len(valid_intents)} available)"

        super().__init__(message, context)
        self.intent = intent
        self.valid_intents = valid_intents or []


class ModelCompatibilityError(IntentClassificationError):
    """Raised when model format or version is incompatible"""

    def __init__(self, model_name: str, expected_format: str, actual_format: str):
        context = {
            "model_name": model_name,
            "expected_format": expected_format,
            "actual_format": actual_format
        }
        message = f"Model compatibility error: {model_name} expected {expected_format}, got {actual_format}"
        super().__init__(message, context)
        self.model_name = model_name
        self.expected_format = expected_format
        self.actual_format = actual_format


class RoutingDecisionError(IntentClassificationError):
    """Raised when routing decisions cannot be determined"""

    def __init__(self, intent: str, reason: str):
        context = {
            "intent": intent,
            "reason": reason
        }
        message = f"Cannot determine routing for intent '{intent}': {reason}"
        super().__init__(message, context)
        self.intent = intent
        self.reason = reason


# Exception utilities
class IntentExceptionHandler:
    """Utility class for handling intent classification exceptions gracefully"""

    @staticmethod
    def handle_classification_error(error: Exception, text: str, fallback_intent: str = "general_chat") -> Dict[str, Any]:
        """
        Handle classification errors and return a safe fallback result

        Args:
            error: The exception that occurred
            text: The original text being classified
            fallback_intent: Intent to use as fallback

        Returns:
            Safe fallback classification result
        """
        # Log the error with context
        from loguru import logger

        if isinstance(error, IntentClassificationError):
            logger.warning(f"Intent classification error: {error}")
            if hasattr(error, 'context') and error.context:
                logger.debug(f"Error context: {error.context}")
        else:
            logger.error(f"Unexpected error in intent classification: {type(error).__name__}: {error}")

        # Return safe fallback result
        return {
            'intent': fallback_intent,
            'confidence': 0.0,
            'fallback': True,
            'error': str(error),
            'error_type': type(error).__name__,
            'processing_time_ms': 0.0,
            'cached': False
        }

    @staticmethod
    def is_recoverable_error(error: Exception) -> bool:
        """
        Check if an error is recoverable (should retry) or fatal

        Args:
            error: The exception to check

        Returns:
            True if error is recoverable, False if fatal
        """
        # Recoverable errors
        recoverable_types = (
            ClassificationTimeoutError,
            CacheLookupError,
            # Network timeouts, temporary issues, etc.
        )

        # Fatal errors
        fatal_types = (
            ModelLoadError,
            ModelCompatibilityError,
            InvalidIntentError,
        )

        if isinstance(error, recoverable_types):
            return True
        elif isinstance(error, fatal_types):
            return False
        else:
            # Unknown errors are considered recoverable with backoff
            return True

    @staticmethod
    def get_recovery_suggestion(error: Exception) -> str:
        """
        Get a human-readable suggestion for recovering from an error

        Args:
            error: The exception to analyze

        Returns:
            Recovery suggestion string
        """
        if isinstance(error, ModelLoadError):
            return f"Check model path and ensure model '{error.model_name}' is available"

        elif isinstance(error, ClassificationTimeoutError):
            return f"Consider increasing timeout or using a faster model (current: {error.timeout_ms}ms)"

        elif isinstance(error, LowConfidenceError):
            return f"Lower confidence threshold (current: {error.threshold:.3f}) or retrain model"

        elif isinstance(error, CacheLookupError):
            return f"Clear cache or check cache configuration for operation: {error.operation}"

        elif isinstance(error, InvalidIntentError):
            return f"Add '{error.intent}' to valid intents or check intent mapping"

        elif isinstance(error, ModelCompatibilityError):
            return f"Update model to {error.expected_format} format or adjust loader"

        elif isinstance(error, RoutingDecisionError):
            return f"Add routing rules for intent '{error.intent}' or use fallback strategy"

        else:
            return "Check logs for detailed error information and consider using fallback processing"


# Convenience function for safe error handling
def safe_classify_with_fallback(classify_func, text: str, fallback_intent: str = "general_chat") -> Dict[str, Any]:
    """
    Safely execute classification with automatic error handling

    Args:
        classify_func: Function that performs classification
        text: Text to classify
        fallback_intent: Intent to use if classification fails

    Returns:
        Classification result or safe fallback
    """
    try:
        return classify_func(text)
    except Exception as e:
        return IntentExceptionHandler.handle_classification_error(e, text, fallback_intent)


if __name__ == "__main__":
    # Test exception hierarchy
    print("Testing Intent Classification Exceptions")
    print("=" * 50)

    # Test different exception types
    try:
        raise ModelLoadError("test-model", ValueError("Model file not found"))
    except ModelLoadError as e:
        print(f"ModelLoadError: {e}")
        print(f"Recovery: {IntentExceptionHandler.get_recovery_suggestion(e)}")
        print(f"Recoverable: {IntentExceptionHandler.is_recoverable_error(e)}")
        print()

    try:
        raise LowConfidenceError("test text", 0.3, 0.7, "unclear_intent")
    except LowConfidenceError as e:
        print(f"LowConfidenceError: {e}")
        print(f"Recovery: {IntentExceptionHandler.get_recovery_suggestion(e)}")
        print()

    # Test safe fallback
    def failing_classify(text):
        raise ClassificationTimeoutError(text, 1000.0)

    result = safe_classify_with_fallback(failing_classify, "test input")
    print(f"Safe fallback result: {result}")