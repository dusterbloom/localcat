"""
BERT-NER Enhanced Entity Extractor

Enhances the existing pattern-based extraction with BERT-NER for better
entity detection. Designed to work alongside the current UD-based system
without breaking existing functionality.

Usage:
    # Enable via environment variable
    os.environ['USE_BERT_NER'] = 'true'

    # Use in extraction pipeline
    bert_extractor = BertNerExtractor()
    entities = bert_extractor.extract_entities(text)
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from loguru import logger

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available - BERT-NER disabled")

@dataclass
class Entity:
    """Extracted entity with metadata."""
    text: str
    label: str
    confidence: float
    start_char: int
    end_char: int


class BertNerExtractor:
    """BERT-NER based entity extractor for enhanced entity detection."""

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        Initialize BERT-NER extractor.

        Args:
            model_name: Hugging Face model name for NER
        """
        self.model_name = model_name
        self.enabled = TRANSFORMERS_AVAILABLE and os.getenv('USE_BERT_NER', 'false').lower() in ('1', 'true', 'yes')
        self._pipeline = None
        self._load_time_ms = 0

        if self.enabled:
            self._initialize_pipeline()
        else:
            logger.info("🤖 BERT-NER disabled (USE_BERT_NER=false or transformers not available)")

    def _initialize_pipeline(self):
        """Initialize the BERT-NER pipeline with timing."""
        logger.info(f"🤖 Initializing BERT-NER with model: {self.model_name}")
        start_time = time.perf_counter()

        try:
            # Initialize the NER pipeline
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple"  # Group sub-tokens
            )

            self._load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"✅ BERT-NER initialized in {self._load_time_ms:.1f}ms")

        except Exception as e:
            logger.error(f"❌ Failed to initialize BERT-NER: {e}")
            self.enabled = False

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using BERT-NER.

        Args:
            text: Input text to extract entities from

        Returns:
            List of Entity objects with confidence scores
        """
        if not self.enabled or not text or not text.strip():
            return []

        start_time = time.perf_counter()

        try:
            # Run BERT-NER
            results = self._pipeline(text)
            extraction_time_ms = (time.perf_counter() - start_time) * 1000

            # Convert to Entity objects
            entities = []
            for result in results:
                entity = Entity(
                    text=result['entity_group'],
                    label=result['entity_group'],
                    confidence=result['score'],
                    start_char=result['start'],
                    end_char=result['end']
                )
                entities.append(entity)

            logger.debug(f"🤖 BERT-NER extracted {len(entities)} entities in {extraction_time_ms:.1f}ms")
            return entities

        except Exception as e:
            logger.error(f"❌ BERT-NER extraction failed: {e}")
            return []

    def extract_with_timing(self, text: str) -> Tuple[List[Entity], float]:
        """
        Extract entities with detailed timing information.

        Returns:
            Tuple of (entities, extraction_time_ms)
        """
        start_time = time.perf_counter()
        entities = self.extract_entities(text)
        extraction_time_ms = (time.perf_counter() - start_time) * 1000

        return entities, extraction_time_ms

    def enhance_triples(self, text: str, existing_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Enhance existing triples with BERT-NER entities.

        Args:
            text: Original text
            existing_triples: Triples from UD pattern extraction

        Returns:
            Enhanced list of triples (existing + BERT-NER enhanced)
        """
        if not self.enabled:
            return existing_triples

        # Extract entities using BERT-NER
        entities = self.extract_entities(text)

        # Convert entities to triples where they make sense
        enhanced_triples = existing_triples.copy()

        for entity in entities:
            # Only add entities that weren't already captured
            if not self._is_entity_covered(entity, existing_triples):
                triple = self._entity_to_triple(entity, text)
                if triple:
                    enhanced_triples.append(triple)

        logger.debug(f"🤖 BERT-NER enhanced triples: {len(existing_triples)} → {len(enhanced_triples)}")
        return enhanced_triples

    def _is_entity_covered(self, entity: Entity, existing_triples: List[Tuple[str, str, str]]) -> bool:
        """Check if an entity is already covered by existing triples."""
        entity_text = entity.text.lower()

        for triple in existing_triples:
            subject, relation, obj = triple
            # Check if entity appears as subject or object
            if entity_text in subject.lower() or entity_text in obj.lower():
                return True

        return False

    def _entity_to_triple(self, entity: Entity, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Convert an entity to a triple representation.

        This is a simplified conversion - in practice, you might want
        more sophisticated mapping based on entity type and context.
        """
        entity_text = text[entity.start_char:entity.end_char]
        entity_label = entity.label.lower()

        # Simple mapping based on entity type
        if entity_label == 'per':
            return (entity_text, 'is_person', entity_text)
        elif entity_label == 'org':
            return (entity_text, 'is_organization', entity_text)
        elif entity_label == 'loc':
            return (entity_text, 'is_location', entity_text)
        elif entity_label == 'misc':
            return (entity_text, 'is_entity', entity_text)
        else:
            # Generic entity representation
            return (entity_text, 'is_mentioned', entity_text)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the extractor."""
        return {
            'enabled': self.enabled,
            'model_name': self.model_name,
            'load_time_ms': self._load_time_ms,
            'transformers_available': TRANSFORMERS_AVAILABLE
        }


# Singleton instance for reuse across the system
_bert_extractor_instance: Optional[BertNerExtractor] = None


def get_bert_extractor() -> BertNerExtractor:
    """Get singleton BERT-NER extractor instance."""
    global _bert_extractor_instance

    if _bert_extractor_instance is None:
        _bert_extractor_instance = BertNerExtractor()

    return _bert_extractor_instance


def enhance_extraction_with_bert(text: str, existing_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Convenience function to enhance existing extraction with BERT-NER.

    This is the main entry point for integrating BERT-NER into the existing pipeline.

    Args:
        text: Original text
        existing_triples: Triples from current pattern-based extraction

    Returns:
        Enhanced list of triples including BERT-NER findings
    """
    extractor = get_bert_extractor()
    return extractor.enhance_triples(text, existing_triples)


if __name__ == "__main__":
    # Simple test
    bert = BertNerExtractor()

    test_text = "I work at OpenAI and live in San Francisco. My name is John."
    entities = bert.extract_entities(test_text)

    print(f"Text: {test_text}")
    print(f"Entities: {entities}")

    existing_triples = [('you', 'work_at', 'openai')]
    enhanced = bert.enhance_triples(test_text, existing_triples)
    print(f"Enhanced triples: {enhanced}")