"""
Centralized constants for the memory system.

These constants capture commonly tuned thresholds to avoid magic numbers
scattered across the codebase.
"""

# Graph edge weight thresholds
WEIGHT_MIN_ACTIVE: float = 0.25   # Minimum weight considered active
WEIGHT_MIN_WEAK: float = 0.10     # Minimum weight considered weak (not negative)
MAX_CONF_CAP: float = 0.75        # Cap for initial confidence on new edges

# Recency decay
RECENCY_HALF_LIFE_MS: int = 7 * 24 * 60 * 60 * 1000  # 7 days

