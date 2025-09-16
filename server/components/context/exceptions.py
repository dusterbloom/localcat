"""
Custom exceptions for context management components.
"""


class ContextError(Exception):
    """Base exception for context-related errors."""
    pass


class ConfigurationError(ContextError):
    """Raised when configuration is invalid or missing."""
    pass


class TokenCountingError(ContextError):
    """Raised when token counting fails."""
    pass


class BudgetError(ContextError):
    """Raised when budget allocation or validation fails."""
    pass


class PackingError(ContextError):
    """Raised when context packing fails."""
    pass


class ValidationError(ContextError):
    """Raised when input validation fails."""
    pass