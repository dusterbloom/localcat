"""
UD-based extractor adapter (Phase 1C)

For incremental modularization, this extractor delegates to the existing
HotMemory methods to avoid behavior changes, but provides a seam for future
standalone implementations.
"""

from typing import Any, List, Tuple

from .base import Extractor


class UDExtractor(Extractor):
    def __init__(self, host: Any):
        """host must provide _extract, _refine_triples, _refine_entities_from_text."""
        self._host = host

    def prewarm(self, lang: str = "en") -> None:
        try:
            # Host prewarm loads spaCy model
            self._host.prewarm(lang)
        except Exception:
            pass

    def extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:
        return self._host._extract(text, lang)  # type: ignore[attr-defined]

    def refine(self, text: str, triples: List[Tuple[str, str, str]], doc: Any) -> List[Tuple[str, str, str]]:
        return self._host._refine_triples(text, triples, doc)  # type: ignore[attr-defined]

    def refine_entities(self, text: str, entities: List[str]) -> List[str]:
        return self._host._refine_entities_from_text(text, entities)  # type: ignore[attr-defined]

