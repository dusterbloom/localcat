"""
UD-based extractor with composable text processing (Phase 1.9)

Enhanced extractor that supports text preprocessing via the strategy pattern.
Follows Interface Segregation Principle by making text processing optional
and Open/Closed Principle by allowing extension without modification.

This maintains backward compatibility while enabling coreference resolution
and other text processing enhancements.
"""

from typing import Any, List, Tuple, Optional
from loguru import logger

from .base import Extractor
from ..processors.base import ProcessorChain, TextProcessor


class UDExtractor(Extractor):
    """
    UD-based extractor with optional text preprocessing.

    Follows SOLID principles:
    - SRP: Focuses on extraction coordination
    - OCP: Open for extension via text processors
    - LSP: Maintains Extractor interface contract
    - ISP: Text processing is optional
    - DIP: Depends on abstractions (TextProcessor interface)
    """

    def __init__(self, host: Any, text_processors: Optional[List[TextProcessor]] = None):
        """
        Initialize extractor with optional text preprocessing.

        Args:
            host: Host object providing extraction methods
            text_processors: Optional list of text processors to apply before extraction

        Note:
            This maintains backward compatibility - existing code that doesn't
            provide text_processors will work unchanged.
        """
        self._host = host
        self._processor_chain = ProcessorChain(text_processors or [])
        self._preprocessing_enabled = bool(text_processors)

    def prewarm(self, lang: str = "en") -> None:
        """Prewarm both host and text processors."""
        try:
            # Host prewarm loads spaCy model
            self._host.prewarm(lang)
        except Exception as e:
            logger.warning(f"Host prewarming failed: {e}")

        # Prewarm text processors if they support it
        for processor in self._processor_chain.processors:
            if hasattr(processor, 'prewarm'):
                try:
                    processor.prewarm(lang)
                except Exception as e:
                    logger.warning(f"Processor {processor.name} prewarming failed: {e}")

    def extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract entities and relations with optional text preprocessing.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Tuple of (entities, triples, neg_count, doc)

        Note:
            If text processors are configured, text will be preprocessed
            before extraction. This enables coreference resolution and
            other enhancements while maintaining interface compatibility.
        """
        try:
            if self._preprocessing_enabled:
                return self._extract_with_preprocessing(text, lang)
            else:
                return self._extract_direct(text, lang)

        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            # Return empty results on failure to maintain robustness
            return [], [], 0, None

    def _extract_direct(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """Direct extraction without preprocessing (backward compatibility path)."""
        return self._host._extract(text, lang)  # type: ignore[attr-defined]

    def _extract_with_preprocessing(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        """
        Extract with text preprocessing applied.

        This implements the enhanced flow:
        1. Get initial spaCy document
        2. Apply text processing chain (e.g., coreference resolution)
        3. Extract from processed document
        4. Apply standard refinement
        """
        # Get initial extraction to obtain spaCy document
        entities, triples, neg_count, doc = self._host._extract(text, lang)  # type: ignore[attr-defined]

        if doc is None:
            logger.debug("No spaCy document available for preprocessing")
            return entities, triples, neg_count, doc

        # Apply text processing chain
        try:
            processed_doc = self._processor_chain.process(doc)

            # If document was modified, re-extract
            if processed_doc != doc or processed_doc.text != doc.text:
                logger.debug("Text was modified by processors, re-extracting")

                # Check if host supports extraction from document
                if hasattr(self._host, '_extract_from_doc'):
                    entities, triples, neg_count, final_doc = self._host._extract_from_doc(processed_doc)  # type: ignore[attr-defined]
                    return entities, triples, neg_count, final_doc
                else:
                    # Fallback: re-extract from processed text
                    entities, triples, neg_count, final_doc = self._host._extract(processed_doc.text, lang)  # type: ignore[attr-defined]
                    return entities, triples, neg_count, final_doc
            else:
                logger.debug("No text modifications applied by processors")
                return entities, triples, neg_count, processed_doc

        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}, using original extraction")
            return entities, triples, neg_count, doc

    def refine(self, text: str, triples: List[Tuple[str, str, str]], doc: Any) -> List[Tuple[str, str, str]]:
        """Refine extracted triples (unchanged from original implementation)."""
        return self._host._refine_triples(text, triples, doc)  # type: ignore[attr-defined]

    def refine_entities(self, text: str, entities: List[str]) -> List[str]:
        """Refine extracted entities (unchanged from original implementation)."""
        return self._host._refine_entities_from_text(text, entities)  # type: ignore[attr-defined]

    def get_processor_metrics(self) -> List[dict]:
        """
        Get metrics from text processors.

        Returns:
            List of processor metrics for observability
        """
        return self._processor_chain.get_metrics_summary()

    def clear_processor_metrics(self) -> None:
        """Clear processor metrics (useful for testing)."""
        self._processor_chain.clear_metrics()

