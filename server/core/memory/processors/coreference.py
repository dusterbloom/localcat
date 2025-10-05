"""
Coreference Resolution Processor

Single-responsibility processor for resolving coreferences in text.
Follows SRP by focusing solely on coreference resolution with timeout protection.

Dependencies are injected (DIP) and the class is open for extension (OCP)
without requiring modification of existing extraction code.
"""

import time
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    import spacy

from .base import TextProcessor
# Export both functions for test patching compatibility
from ..nlp_manager import get_nlp_with_coref, get_nlp_model  # noqa: F401


class CoreferenceProcessor(TextProcessor):
    """
    Processor for resolving coreferences using spacy-coref.

    Single Responsibility: Only handles coreference resolution
    Dependencies: Injected via NLP manager (DIP principle)
    Extensibility: Can be extended without modifying existing code (OCP)

    Responsibilities:
    - Resolve pronouns and entity references across sentences
    - Apply timeout protection to prevent latency issues
    - Graceful fallback when resolution fails
    - Metrics collection for observability
    """

    def __init__(self, timeout_ms: int = 50, min_text_length: int = 10, lang: str = "en"):
        """
        Initialize coreference processor.

        Args:
            timeout_ms: Maximum processing time before fallback (default: 50ms)
            min_text_length: Minimum text length to process (default: 10 chars)
            lang: Language for spaCy model (default: "en")
        """
        super().__init__("coreference")
        self.timeout_ms = timeout_ms
        self.min_text_length = min_text_length
        self.lang = lang
        self._nlp = None

    def _ensure_model_loaded(self) -> bool:
        """
        Lazy load the coreference model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._nlp is None:
            try:
                self._nlp = get_nlp_with_coref(self.lang)
                if self._nlp is None:
                    logger.warning("Failed to load spaCy model with coreference component")
                    return False
                logger.debug(f"Loaded spaCy model with coreference for {self.lang}")
            except Exception as e:
                logger.warning(f"Exception loading coreference model: {e}")
                return False

        return True

    def process(self, doc: "spacy.Doc") -> "spacy.Doc":
        """
        Process document to resolve coreferences.

        Args:
            doc: Input spaCy document

        Returns:
            Document with resolved coreferences, or original doc if processing fails

        Note:
            This method follows the fail-safe principle - it never throws exceptions
            and always returns a valid document, even if processing fails.
        """
        start_time = time.perf_counter()

        try:
            # Skip processing if text is too short
            if len(doc.text.strip()) < self.min_text_length:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._record_metric(elapsed_ms, True, "skipped_short_text")
                logger.debug(f"Skipping coreference for short text ({len(doc.text)} chars)")
                return doc

            # Ensure model is loaded
            if not self._ensure_model_loaded():
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._record_metric(elapsed_ms, False, "model_load_failed")
                return doc

            # Process with timeout protection
            resolved_doc = self._resolve_with_timeout(doc, start_time)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if resolved_doc is not None:
                self._record_metric(elapsed_ms, True, f"resolved_{len(doc.text)}_chars")
                return resolved_doc
            else:
                self._record_metric(elapsed_ms, False, "resolution_timeout")
                logger.debug(f"Coreference resolution timed out after {elapsed_ms:.1f}ms")
                return doc

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_metric(elapsed_ms, False, f"exception_{type(e).__name__}")
            logger.warning(f"Coreference processing failed: {e}")
            return doc

    def _resolve_with_timeout(self, doc: "spacy.Doc", start_time: float) -> Optional["spacy.Doc"]:
        """
        Resolve coreferences with timeout protection.

        Args:
            doc: Input document
            start_time: Processing start time for timeout calculation

        Returns:
            Resolved document or None if timeout exceeded
        """
        try:
            # Check if we're already approaching timeout
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > self.timeout_ms * 0.5:  # 50% of timeout used already
                logger.debug(f"Skipping coreference: {elapsed:.1f}ms already elapsed")
                return None

            # Re-process text through coreference-enabled pipeline
            # Note: This creates a new Doc object with coreference clusters
            coref_doc = self._nlp(doc.text)

            # Check timeout after processing
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > self.timeout_ms:
                logger.debug(f"Coreference timeout: {elapsed:.1f}ms > {self.timeout_ms}ms")
                return None

            # Apply coreference resolution if clusters exist
            if hasattr(coref_doc._, 'coref_clusters') and coref_doc._.coref_clusters:
                resolved_text = self._apply_coreference_resolution(coref_doc)

                # Check final timeout
                elapsed = (time.perf_counter() - start_time) * 1000
                if elapsed > self.timeout_ms:
                    return None

                # Create new document with resolved text if different
                if resolved_text != doc.text:
                    # Use original nlp pipeline to maintain consistency
                    # We need the original pipeline for UD extraction
                    return self._create_resolved_doc(resolved_text, doc)
                else:
                    logger.debug("No coreference changes applied")

            return coref_doc

        except Exception as e:
            logger.debug(f"Coreference resolution error: {e}")
            return None

    def _apply_coreference_resolution(self, doc: "spacy.Doc") -> str:
        """
        Apply coreference resolution to create resolved text.

        Args:
            doc: Document with coreference clusters

        Returns:
            Text with resolved coreferences
        """
        if not hasattr(doc._, 'coref_clusters') or not doc._.coref_clusters:
            return doc.text

        # Get coreference clusters
        clusters = doc._.coref_clusters

        # Build replacement map: span -> representative mention
        replacements = {}

        for cluster in clusters:
            if len(cluster) < 2:  # Need at least 2 mentions to resolve
                continue

            # Use the first mention as the representative (usually the most informative)
            representative = cluster[0]

            # Replace other mentions with the representative
            for mention in cluster[1:]:
                # Only replace pronouns and less informative mentions
                if self._should_replace_mention(mention, representative):
                    replacements[mention] = representative

        # Apply replacements to create resolved text
        if not replacements:
            return doc.text

        # Sort replacements by position (reverse order to preserve indices)
        sorted_replacements = sorted(replacements.items(), key=lambda x: x[0].start, reverse=True)

        resolved_text = doc.text
        for mention, representative in sorted_replacements:
            # Replace mention span with representative text
            start_char = mention.start_char
            end_char = mention.end_char
            replacement = representative.text

            resolved_text = (
                resolved_text[:start_char] +
                replacement +
                resolved_text[end_char:]
            )

        logger.debug(f"Applied {len(sorted_replacements)} coreference replacements")
        return resolved_text

    def _should_replace_mention(self, mention, representative) -> bool:
        """
        Determine if a mention should be replaced with the representative.

        Args:
            mention: Mention to potentially replace
            representative: Representative mention

        Returns:
            True if the mention should be replaced
        """
        mention_text = mention.text.lower().strip()
        representative_text = representative.text.lower().strip()

        # Don't replace if they're the same
        if mention_text == representative_text:
            return False

        # Replace common pronouns
        pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their"}
        if mention_text in pronouns:
            return True

        # Replace less specific mentions with more specific ones
        if len(mention_text) < len(representative_text) and mention_text in representative_text:
            return True

        return False

    def _create_resolved_doc(self, resolved_text: str, original_doc: "spacy.Doc") -> "spacy.Doc":
        """
        Create a new document with resolved text using the original pipeline.

        Args:
            resolved_text: Text with resolved coreferences
            original_doc: Original document for pipeline reference

        Returns:
            New document processed with resolved text
        """
        try:
            # We need to use a clean pipeline without coreference for UD extraction
            # This ensures compatibility with existing extraction code
            from ..nlp_manager import get_nlp_model
            clean_nlp = get_nlp_model(self.lang)

            if clean_nlp is not None:
                return clean_nlp(resolved_text)
            else:
                logger.warning("Could not get clean NLP model, returning original")
                return original_doc

        except Exception as e:
            logger.warning(f"Failed to create resolved document: {e}")
            return original_doc