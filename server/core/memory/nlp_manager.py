"""
Shared NLP Model Manager

Consolidates spaCy model loading patterns to eliminate DRY violations.
Currently there are 3 separate implementations in:
- fact_extractor.py: _load_nlp() singleton pattern
- memory_hotpath.py: _load_nlp() singleton pattern
- Proposed CoreferenceResolver: _ensure_initialized() pattern

This manager provides a single source of truth for NLP model management
following SOLID principles and DRY methodology.
"""

import spacy
from typing import Optional, List, Dict, Tuple
from loguru import logger
from threading import Lock


class SharedNLPManager:
    """
    Centralized NLP model management following DRY and SRP principles.

    Responsibilities:
    - Load and cache spaCy models
    - Manage pipeline components
    - Thread-safe model access
    - Graceful error handling with fallbacks
    """

    def __init__(self):
        self._models: Dict[str, Optional[spacy.Language]] = {}
        self._lock = Lock()

    def get_model(self, lang: str = "en", components: Optional[List[str]] = None) -> Optional[spacy.Language]:
        """
        Get or load spaCy model with optional pipeline components.

        Args:
            lang: Language code (default: "en")
            components: Optional list of pipeline components to add

        Returns:
            spaCy Language model or None if loading failed
        """
        components = components or []
        cache_key = self._build_cache_key(lang, components)

        with self._lock:
            if cache_key not in self._models:
                self._models[cache_key] = self._load_model(lang, components)

            return self._models[cache_key]

    def _build_cache_key(self, lang: str, components: List[str]) -> str:
        """Build cache key for model + components combination."""
        sorted_components = sorted(components) if components else []
        return f"{lang}_{':'.join(sorted_components)}"

    def _load_model(self, lang: str, components: List[str]) -> Optional[spacy.Language]:
        """
        Load spaCy model with components, following existing patterns.

        This consolidates the logic from:
        - fact_extractor.py:_load_nlp() (lines 30-43)
        - memory_hotpath.py:_load_nlp() (lines 34-47)
        """
        try:
            # Load base model following existing pattern
            import os
            # Allow env override for model selection and disabled components
            override = os.getenv(f"SPACY_MODEL_{lang.upper()}") or (os.getenv("SPACY_MODEL_EN") if lang == "en" else None) or os.getenv("SPACY_MODEL")
            disable_env = os.getenv("SPACY_DISABLE", "ner,textcat").strip()
            disabled = [c.strip() for c in disable_env.split(",") if c.strip()] if disable_env else []

            model_name = override if override else ("en_core_web_sm" if lang == "en" else f"{lang}_core_news_sm")
            nlp = spacy.load(model_name, disable=disabled)
            logger.info(f"Loaded spaCy model {model_name} with disabled={disabled}")

            # Add requested components
            for component in components:
                if component not in nlp.pipe_names:
                    try:
                        nlp.add_pipe(component)
                        logger.debug(f"Added pipeline component: {component}")
                    except Exception as e:
                        logger.warning(f"Failed to add component {component}: {e}")
                        # Continue without this component rather than failing entirely

            logger.info(f"Loaded spaCy model {lang} with components: {components}")
            return nlp

        except Exception as e:
            logger.warning(f"Could not load spaCy model for {lang}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear model cache (useful for testing)."""
        with self._lock:
            self._models.clear()

    def get_cache_info(self) -> Dict[str, bool]:
        """Get information about cached models."""
        with self._lock:
            return {key: (model is not None) for key, model in self._models.items()}


# Global instance following singleton pattern used in existing code
_nlp_manager = SharedNLPManager()


def get_nlp_manager() -> SharedNLPManager:
    """Get the global NLP manager instance."""
    return _nlp_manager


# Backward compatibility functions to ease migration
def get_nlp_model(lang: str = "en") -> Optional[spacy.Language]:
    """
    Backward compatibility function matching existing _load_nlp() signature.

    This allows gradual migration of existing code:
    - Replace: nlp = _load_nlp("en")
    - With: nlp = get_nlp_model("en")
    """
    return _nlp_manager.get_model(lang)


def get_nlp_with_coref(lang: str = "en") -> Optional[spacy.Language]:
    """
    Get spaCy model with coreference resolution component.

    This is the coreference-specific entry point that will be used
    by the CoreferenceProcessor.
    """
    return _nlp_manager.get_model(lang, components=["coref"])
