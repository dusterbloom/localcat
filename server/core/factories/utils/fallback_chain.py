"""
FallbackChainManager: execute a sequence of creation callables until one succeeds.

Provides structured logging and error aggregation while keeping engine-specific
creation functions simple and testable.
"""

from typing import Callable, List, Any, Dict
from loguru import logger


class ChainExhaustedError(Exception):
    """Raised when all functions in fallback chain fail."""

    def __init__(self, message: str, errors: List[Dict]):
        super().__init__(message)
        self.errors = errors


class FallbackChainManager:
    """
    Executes creation functions with automatic fallback handling.

    - Structured error logging
    - Fallback chain execution
    - Error aggregation
    """

    def execute(self, chain: List[Callable[[], Any]], context: str) -> Any:
        errors: List[Dict] = []

        for i, create_fn in enumerate(chain):
            name = getattr(create_fn, "__name__", str(create_fn))
            try:
                logger.debug(f"{context}: Trying {name} ({i+1}/{len(chain)})")
                result = create_fn()
                if result is not None:
                    if i == 0:
                        logger.debug(f"✅ {context}: {name} succeeded (primary)")
                    else:
                        logger.info(f"✅ {context}: {name} succeeded (fallback #{i})")
                    return result
            except Exception as e:  # noqa: BLE001 - we aggregate and re-raise
                errors.append({"function": name, "error": str(e), "type": type(e).__name__})
                # Only include traceback for last failure to avoid log noise
                logger.warning(
                    f"⚠️  {context}: {name} failed: {e}",
                    exc_info=(i == len(chain) - 1),
                )

        raise ChainExhaustedError(
            f"{context}: All {len(chain)} creation attempts failed", errors=errors
        )

