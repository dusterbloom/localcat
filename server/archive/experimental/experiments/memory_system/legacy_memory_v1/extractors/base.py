"""
Extractor interface for HotMem (Phase 1C)
"""

from typing import Any, List, Tuple


class Extractor:
    """Abstract extractor interface.

    Implementations should provide:
      - prewarm(lang)
      - extract(text, lang) -> (entities, triples, neg_count, doc)
      - refine(text, triples, doc) -> triples
      - refine_entities(text, entities) -> entities
    """

    def prewarm(self, lang: str = "en") -> None:  # pragma: no cover - interface
        pass

    def extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str, str, str]], int, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def refine(self, text: str, triples: List[Tuple[str, str, str]], doc: Any) -> List[Tuple[str, str, str]]:  # pragma: no cover - interface
        raise NotImplementedError

    def refine_entities(self, text: str, entities: List[str]) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError

