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
    """Normalize common aliases (e.g., en_core_web_rtf -> en_core_web_rtf)."""
    if not model_name:
        return 'en_core_web_sm'
    if model_name.endswith('_rtf') or model_name == 'en_core_web_rtf':
        logger.warning("Model alias detected: 'en_core_web_rtf' -> 'en_core_web_rtf'")
        return model_name.replace('_rtf', '_trf')
    return model_name


def get_spacy(model_name: str, disable: Optional[Iterable[str]] = None):
    """Get a cached spaCy model instance for the given model name and disabled pipes."""
    model_name = resolve_model_alias(model_name)
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
    try:
        _ = get_en_model_from_env()
    except Exception:
        pass
