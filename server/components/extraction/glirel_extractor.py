#!/usr/bin/env python3
"""
GLiREL-based relation extraction for zero-shot, high-performance relation extraction
"""

import time
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from glirel import GLiREL
    from transformers import AutoTokenizer
    GLIREL_AVAILABLE = True
except ImportError:
    GLIREL_AVAILABLE = False
    logger.warning("[GLiREL] GLiREL not available")

@dataclass
class RelationResult:
    """Result of relation extraction"""
    head: str
    tail: str
    relation: str
    score: float
    head_start: int
    head_end: int
    tail_start: int
    tail_end: int

class GLiRELExtractor:
    """
    Zero-shot relation extraction using GLiREL (2025 SOTA)

    Key advantages:
    - Zero-shot: Works with any relation types
    - Fast: Single forward pass for all relations
    - No pre-defined relation schemas needed
    - Lightweight: Based on GLiNER architecture
    """

    def __init__(self, model_id: str = "jackboyla/glirel-large-v0", device: str = "auto"):
        """Initialize GLiREL extractor"""
        if not GLIREL_AVAILABLE:
            raise ImportError("GLiREL not available. Install with: pip install glirel")

        self.model_id = model_id
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        # Default relation types for general domain
        self.default_relations = {
            # Person-Organization relations
            'works_at': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
            'founded': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
            'ceo_of': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
            'employed_by': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
            'board_member_of': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},

            # Organization-Organization relations
            'acquired': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
            'merged_with': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
            'partnered_with': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
            'subsidiary_of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
            'competitor_of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},

            # Location relations
            'located_in': {"allowed_head": ["ORG", "LOC"], "allowed_tail": ["LOC"]},
            'headquartered_in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC"]},
            'based_in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC"]},
            'operates_in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC"]},

            # Person-Location relations
            'born_in': {"allowed_head": ["PERSON"], "allowed_tail": ["LOC"]},
            'lives_in': {"allowed_head": ["PERSON"], "allowed_tail": ["LOC"]},
            'from': {"allowed_head": ["PERSON"], "allowed_tail": ["LOC"]},

            # Product/Service relations
            'produces': {"allowed_head": ["ORG"], "allowed_tail": ["PRODUCT"]},
            'develops': {"allowed_head": ["ORG"], "allowed_tail": ["PRODUCT"]},
            'sells': {"allowed_head": ["ORG"], "allowed_tail": ["PRODUCT"]},
            'owns': {"allowed_head": ["ORG"], "allowed_tail": ["PRODUCT"]},

            # Temporal relations
            'established_in': {},
            'founded_in': {},
            'created_in': {},

            # General relations (no restrictions)
            'associated_with': {},
            'related_to': {},
            'part_of': {},
            'member_of': {},
        }

        logger.info(f"[GLiREL] Initialized with model: {model_id}")

    def _get_device(self, device: str) -> str:
        """Determine best device"""
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def load_model(self):
        """Load GLiREL model (lazy loading)"""
        if self.is_loaded:
            return

        try:
            logger.info(f"[GLiREL] Loading model: {self.model_id}")
            start = time.perf_counter()

            # Load GLiREL model
            self.model = GLiREL.from_pretrained(self.model_id)

            # Use the correct tokenizer based on the model's base architecture
            try:
                # GLiREL is based on microsoft/deberta-v3-large
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
                logger.info("[GLiREL] Using microsoft/deberta-v3-large tokenizer")
            except Exception as tokenizer_error:
                logger.warning(f"[GLiREL] Failed to load deberta-v3 tokenizer: {tokenizer_error}")
                # Fallback to bert tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logger.info("[GLiREL] Using bert-base-uncased tokenizer fallback")

            load_time = (time.perf_counter() - start) * 1000
            self.is_loaded = True

            logger.info(f"[GLiREL] Model loaded in {load_time:.1f}ms on {self.device}")

        except Exception as e:
            logger.error(f"[GLiREL] Failed to load model: {e}")
            raise

    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: Optional[Dict[str, Dict[str, List[str]]]] = None,
        threshold: float = 0.5
    ) -> List[RelationResult]:
        """
        Extract relations from text using GLiREL

        Args:
            text: Input text
            entities: List of entity dicts with 'text', 'start', 'end', 'label'
            relations: Optional custom relation definitions (default relations used if None)
            threshold: Confidence threshold for relation extraction

        Returns:
            List of RelationResult objects
        """
        if not self.is_loaded:
            self.load_model()

        # Use default relations if none provided
        if relations is None:
            relations = self.default_relations

        # Filter entities to those that might participate in relations
        filtered_entities = []
        for ent in entities:
            # Convert entity label to standard categories
            label = self._normalize_entity_label(ent.get('label', ''))
            if label in ['PERSON', 'ORG', 'LOC', 'PRODUCT']:
                filtered_entities.append({
                    'text': ent['text'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'label': label
                })

        if len(filtered_entities) < 2:
            logger.debug("[GLiREL] Need at least 2 entities for relation extraction")
            return []

        try:
            start = time.perf_counter()

            # Tokenize text
            tokens = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=512
            )

            # Extract relations using GLiREL
            results = self.model.predict_relations(
                tokens=tokens,
                labels=relations,
                ner=filtered_entities,
                threshold=threshold
            )

            extract_time = (time.perf_counter() - start) * 1000

            # Convert to our format
            relation_results = []
            for result in results:
                relation_results.append(RelationResult(
                    head=result['head'],
                    tail=result['tail'],
                    relation=result['relation'],
                    score=result['score'],
                    head_start=self._find_entity_position(filtered_entities, result['head'])[0],
                    head_end=self._find_entity_position(filtered_entities, result['head'])[1],
                    tail_start=self._find_entity_position(filtered_entities, result['tail'])[0],
                    tail_end=self._find_entity_position(filtered_entities, result['tail'])[1]
                ))

            logger.debug(f"[GLiREL] Extracted {len(relation_results)} relations in {extract_time:.1f}ms")
            return relation_results

        except Exception as e:
            logger.error(f"[GLiREL] Extraction failed: {e}")
            return []

    def _normalize_entity_label(self, label: str) -> str:
        """Normalize entity labels to GLiREL format"""
        label_lower = label.lower()
        if any(person in label_lower for person in ['person', 'per']):
            return 'PERSON'
        elif any(org in label_lower for org in ['org', 'company', 'corp']):
            return 'ORG'
        elif any(loc in label_lower for loc in ['loc', 'place', 'city', 'country']):
            return 'LOC'
        elif any(prod in label_lower for prod in ['product', 'service']):
            return 'PRODUCT'
        else:
            return label.upper()

    def _find_entity_position(self, entities: List[Dict], entity_text: str) -> Tuple[int, int]:
        """Find entity position in text"""
        for ent in entities:
            if ent['text'] == entity_text:
                return ent['start'], ent['end']
        return -1, -1

    def extract_with_gliner_integration(
        self,
        text: str,
        gliner_result: Optional[List[Dict]] = None,
        relations: Optional[Dict[str, Dict[str, List[str]]]] = None,
        threshold: float = 0.5
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relations using GLiNER entities + GLiREL relations

        Args:
            text: Input text
            gliner_result: GLiNER entity extraction results (optional)
            relations: Custom relation definitions (optional)
            threshold: Confidence threshold

        Returns:
            List of (subject, relation, object) tuples
        """
        # If no GLiNER results provided, we can't extract relations
        if not gliner_result:
            logger.debug("[GLiREL] No entities provided for relation extraction")
            return []

        # Convert GLiNER results to GLiREL format
        entities = []
        for ent in gliner_result:
            entities.append({
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'label': ent['label']
            })

        # Extract relations
        relation_results = self.extract_relations(text, entities, relations, threshold)

        # Convert to triple format
        triples = []
        for rel in relation_results:
            triples.append((rel.head, rel.relation, rel.tail))

        return triples

    def get_custom_relations(self, domain_relations: Dict[str, Dict]) -> Dict[str, Dict]:
        """Get relations for specific domain"""
        # Merge with default relations
        custom_relations = self.default_relations.copy()
        custom_relations.update(domain_relations)
        return custom_relations

    def __call__(self, text: str, entities: List[Dict], **kwargs) -> List[Tuple[str, str, str]]:
        """Convenience method for direct use"""
        relations = self.extract_relations(text, entities, **kwargs)
        return [(r.head, r.relation, r.tail) for r in relations]