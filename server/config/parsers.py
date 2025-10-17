"""
Configuration parsing utilities.

Provides reusable parsing functions for environment variables with type safety
and validation. Eliminates code duplication between configuration modules.
"""

from typing import Optional, List, TypeVar, Callable
from loguru import logger

T = TypeVar('T')


def _parse_bool(value: Optional[str]) -> bool:
    """
    Parse boolean from environment variable.

    Args:
        value: String value from environment variable

    Returns:
        True if value is "true", "1", "yes", "on" (case-insensitive), False otherwise

    Examples:
        >>> _parse_bool("true")
        True
        >>> _parse_bool("FALSE")
        False
        >>> _parse_bool(None)
        False
    """
    if value is None:
        return False
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: Optional[str], default: int) -> int:
    """
    Parse integer from environment variable with fallback.

    Args:
        value: String value from environment variable
        default: Default value if parsing fails

    Returns:
        Parsed integer or default value

    Examples:
        >>> _parse_int("42", 0)
        42
        >>> _parse_int("invalid", 10)
        10
        >>> _parse_int(None, 5)
        5
    """
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value '{value}', using default {default}")
        return default


def _parse_float(value: Optional[str], default: float) -> float:
    """
    Parse float from environment variable with fallback.

    Args:
        value: String value from environment variable
        default: Default value if parsing fails

    Returns:
        Parsed float or default value

    Examples:
        >>> _parse_float("3.14", 0.0)
        3.14
        >>> _parse_float("invalid", 1.5)
        1.5
        >>> _parse_float(None, 2.0)
        2.0
    """
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value '{value}', using default {default}")
        return default


def _parse_list(value: Optional[str], default: List[str], separator: str = ',') -> List[str]:
    """
    Parse comma-separated list from environment variable.

    Args:
        value: String value from environment variable (comma-separated)
        default: Default list if value is None
        separator: Separator character (default: comma)

    Returns:
        List of trimmed non-empty strings

    Examples:
        >>> _parse_list("a,b,c", [])
        ['a', 'b', 'c']
        >>> _parse_list("  foo , bar  ", [])
        ['foo', 'bar']
        >>> _parse_list(None, ["default"])
        ['default']
    """
    if value is None:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


def _parse_enum(value: Optional[str], default: T, allowed_values: List[T]) -> T:
    """
    Parse enum/choice from environment variable with validation.

    Args:
        value: String value from environment variable
        default: Default value if parsing fails or value not in allowed list
        allowed_values: List of allowed values

    Returns:
        Parsed value if in allowed_values, otherwise default

    Examples:
        >>> _parse_enum("debug", "info", ["debug", "info", "warning"])
        'debug'
        >>> _parse_enum("invalid", "info", ["debug", "info"])
        'info'
    """
    if value is None:
        return default

    if value not in allowed_values:
        logger.warning(
            f"Invalid value '{value}', must be one of {allowed_values}. "
            f"Using default '{default}'"
        )
        return default

    return value


def parse_with_validator(
    value: Optional[str],
    parser: Callable[[str], T],
    default: T,
    validator: Optional[Callable[[T], bool]] = None,
    error_message: str = "Validation failed"
) -> T:
    """
    Parse value with custom parser and optional validation.

    Args:
        value: String value from environment variable
        parser: Function to parse the string value
        default: Default value if parsing or validation fails
        validator: Optional validation function
        error_message: Error message if validation fails

    Returns:
        Parsed and validated value, or default if parsing/validation fails

    Examples:
        >>> parse_with_validator("10", int, 5, lambda x: x > 0, "Must be positive")
        10
        >>> parse_with_validator("-5", int, 5, lambda x: x > 0, "Must be positive")
        5
    """
    if value is None:
        return default

    try:
        parsed = parser(value)

        if validator is not None and not validator(parsed):
            logger.warning(f"{error_message}: '{value}'. Using default '{default}'")
            return default

        return parsed
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse '{value}': {e}. Using default '{default}'")
        return default
