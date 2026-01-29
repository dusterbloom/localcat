"""
Centralized constants for the memory system.

These constants capture commonly tuned thresholds to avoid magic numbers
scattered across the codebase.
"""

import os

# Graph edge weight thresholds
WEIGHT_MIN_ACTIVE: float = 0.15   # Minimum weight considered active (lowered from 0.25 to improve recall)
WEIGHT_MIN_WEAK: float = 0.10     # Minimum weight considered weak (not negative)
MAX_CONF_CAP: float = 0.75        # Cap for initial confidence on new edges

# Recency decay
RECENCY_HALF_LIFE_MS: int = int(os.getenv("RECENCY_HALF_LIFE_HOURS", "24")) * 3600 * 1000  # default 24h, configurable


# === Performance Targets ===
class LatencyTargets:
    """Target latency budgets for different operations (milliseconds)"""
    EXTRACTION_MS = 50
    RETRIEVAL_MS = 100
    TOTAL_MS = 200  # p95 target
    ENTITY_ENRICHMENT_WARNING_MS = 1.0  # Log warning if enrichment exceeds this


# === Confidence Thresholds ===
class ConfidenceThresholds:
    """Confidence scoring thresholds for edges and filtering"""
    NEGATION = 0.6  # Confidence score for negated edges
    MINIMUM = 0.1  # Minimum confidence to keep edge active
    DEFAULT = 0.5  # Default confidence for new edges


# === Cache Configuration ===
class CacheConfig:
    """LRU cache sizes for various components"""
    EXTRACTION_CACHE_SIZE = 128  # Size of extraction cache
    QUERY_EXPANSION_CACHE_SIZE = 128  # Size of query expansion cache
    ENTITY_ENRICHMENT_CACHE_SIZE = 256  # Size of entity enrichment cache
    PROVENANCE_CACHE_SIZE = 1024  # Size of provenance cache
    BM25_STATS_TTL_SECONDS = 60  # TTL for BM25 collection stats cache


# === Quality Filtering ===
class QualityThresholds:
    """Thresholds for quality filtering at various stages"""
    MIN_CONVERSATION_LENGTH = 10  # Minimum chars for conversation storage
    MIN_ENTITY_LENGTH = 2  # Minimum entity length
    MAX_ENTITY_ENRICHMENT_LENGTH = 50  # Maximum enriched entity length


# === FTS Configuration ===
class FTSConfig:
    """Full-text search configuration"""
    MAX_QUERY_TERMS = 5  # Maximum terms in expanded query
    MAX_SYNONYMS_PER_TERM = 2  # Maximum synonyms per term
    MIN_TERM_LENGTH = 2  # Minimum term length for FTS

