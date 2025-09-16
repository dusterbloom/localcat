#!/usr/bin/env python3
"""
Centralized spaCy model cache and helpers.

Avoids duplicate spacy.load() across components and supports simple aliasing.
"""

from typing import Dict, Tuple, Optional, Iterable

import spacy
from loguru import logger

_SPACY_CACHE: Dict[Tuple[str, Tuple[str, ...]], object] = {}


def resolve_model_alias(model_name: str) -> str:
    """Normalize common aliases (e.g., en_core_web_trf -> en_core_web_trf)."""
    if not model_name:
        return 'en_core_web_sm'
    if model_name.endswith('_rtf') or model_name == 'en_core_web_trf':
        return model_name.replace('_rtf', '_trf')
    return model_name


def get_spacy(model_name: str, disable: Optional[Iterable[str]] = None):
    """Get a cached spaCy model instance for the given model name and disabled pipes."""
    
    disable_tuple = tuple(disable or ())
    key = (model_name, disable_tuple)
    if key in _SPACY_CACHE:
        return _SPACY_CACHE[key]
    try:
        nlp = spacy.load(model_name, disable=list(disable_tuple))
        _SPACY_CACHE[key] = nlp
        logger.info(f"[nlp_cache] Loaded spaCy model: {model_name} disable={list(disable_tuple) if disable_tuple else []}")
        return nlp
    except Exception as e:
        logger.warning(f"[nlp_cache] Failed to load model '{model_name}': {e}; falling back to blank")
        try:
            lang = 'en'
            nlp = spacy.blank(lang)
            if 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer')
            _SPACY_CACHE[key] = nlp
            return nlp
        except Exception:
            return None


def get_en_model_from_env(default_model: str = 'en_core_web_sm'):
    import os
    model = os.getenv('ENHANCED_LEVEL3_SPACY_MODEL', default_model)
    model = resolve_model_alias(model)
    return get_spacy(model)


def prewarm_from_env() -> None:
    """Prewarm NLP models and extraction strategies to avoid first-use latency."""
    try:
        # Prewarm the spaCy model
        logger.info("[nlp_cache] Starting model prewarm...")
        nlp = get_en_model_from_env()

        # Do a dummy parse to fully load the model
        if nlp:
            _ = nlp("Test warmup sentence.")
            logger.info("[nlp_cache] SpaCy model warmed up successfully")

        # Also prewarm the extraction strategy to avoid first-use delay
        try:
            from components.extraction.extraction_registry import ExtractionRegistry
            registry = ExtractionRegistry()
            strategy = registry.get_strategy('enhanced_level3')
            if strategy:
                # Do a dummy extraction to warm up
                _ = strategy.extract("Warmup extraction test.", "en")
                logger.info("[nlp_cache] Extraction strategy warmed up successfully")
        except Exception as e:
            logger.debug(f"[nlp_cache] Could not prewarm extraction strategy: {e}")

    except Exception as e:
        logger.warning(f"[nlp_cache] Prewarm failed: {e}")
        pass
