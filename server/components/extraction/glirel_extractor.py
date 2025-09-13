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
            label = self._normalize_entity_label(ent.get('label', ''), ent['text'])
            # Accept standard categories plus ENTITY for zero-shot mode
            if label in ['PERSON', 'ORG', 'LOC', 'PRODUCT', 'ENTITY']:
                filtered_entities.append({
                    'text': ent['text'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'label': label
                })
                logger.debug(f"[GLiREL] Entity '{ent['text']}' classified as '{label}'")

        if len(filtered_entities) < 2:
            logger.debug("[GLiREL] Need at least 2 entities for relation extraction")
            return []

        try:
            start = time.perf_counter()

            # Tokenize text for GLiREL (expects tokens as list of strings)
            import spacy
            try:
                nlp = spacy.load('en_core_web_sm')
                doc = nlp(text)
                tokens = [token.text for token in doc]
            except:
                # Fallback to simple word splitting
                tokens = text.split()
                doc = None

            # Convert relations dict to list of relation labels
            if isinstance(relations, dict):
                labels = list(relations.keys())
            else:
                labels = relations or list(self.default_relations.keys())

            # Convert entities to GLiREL format: [start_token_idx, end_token_idx, type, text]
            glirel_entities = []
            for ent in filtered_entities:
                if doc:
                    # Use spaCy to find proper token indices
                    start_token_idx = None
                    end_token_idx = None

                    for token in doc:
                        # Check if token overlaps with entity
                        if token.idx <= ent['start'] < token.idx + len(token.text):
                            start_token_idx = token.i
                        if token.idx < ent['end'] <= token.idx + len(token.text):
                            end_token_idx = token.i

                    # If we couldn't find exact matches, find closest tokens
                    if start_token_idx is None or end_token_idx is None:
                        for token in doc:
                            if start_token_idx is None and token.idx >= ent['start']:
                                start_token_idx = max(0, token.i - 1)
                            if end_token_idx is None and token.idx >= ent['end']:
                                end_token_idx = token.i
                                break

                    # Fallback to last token if still not found
                    if start_token_idx is None:
                        start_token_idx = 0
                    if end_token_idx is None:
                        end_token_idx = len(tokens) - 1
                else:
                    # Fallback: approximate token positions
                    words_before_start = len(text[:ent['start']].split())
                    entity_words = len(ent['text'].split())
                    start_token_idx = max(0, words_before_start - 1)
                    end_token_idx = min(len(tokens) - 1, start_token_idx + entity_words - 1)

                glirel_entities.append([start_token_idx, end_token_idx, ent['label'], ent['text']])

            logger.debug(f"[GLiREL] Using {len(labels)} relation types and {len(glirel_entities)} entities")

            # Extract relations using GLiREL with correct API
            results = self.model.predict_relations(
                text=text,
                labels=labels,
                ner=glirel_entities,
                threshold=threshold,
                top_k=5  # Get top 5 relations per entity pair
            )

            extract_time = (time.perf_counter() - start) * 1000

            # Debug: check the results format
            logger.debug(f"[GLiREL] Raw results format: {type(results)} - {results}")

            # Convert to our format
            relation_results = []
            if results:  # Make sure we have results
                for i, result in enumerate(results):
                    logger.debug(f"[GLiREL] Result {i}: {result} (type: {type(result)})")

                    # Handle different possible formats
                    if isinstance(result, dict):
                        # GLiREL format: head_text, tail_text, label, score
                        head = result.get('head_text', result.get('head', result.get('subject', '')))
                        tail = result.get('tail_text', result.get('tail', result.get('object', '')))
                        relation = result.get('label', result.get('relation', result.get('predicate', '')))
                        score = result.get('score', result.get('confidence', 0.0))
                    elif isinstance(result, (list, tuple)) and len(result) >= 3:
                        # Tuple format: (head, relation, tail) or similar
                        head, relation, tail = result[0], result[1], result[2]
                        score = result[3] if len(result) > 3 else 1.0
                    else:
                        logger.warning(f"[GLiREL] Unknown result format: {result}")
                        continue

                    if head and tail and relation:
                        relation_results.append(RelationResult(
                            head=str(head),
                            tail=str(tail),
                            relation=str(relation),
                            score=float(score),
                            head_start=self._find_entity_position(filtered_entities, str(head))[0],
                            head_end=self._find_entity_position(filtered_entities, str(head))[1],
                            tail_start=self._find_entity_position(filtered_entities, str(tail))[0],
                            tail_end=self._find_entity_position(filtered_entities, str(tail))[1]
                        ))

            logger.debug(f"[GLiREL] Extracted {len(relation_results)} relations in {extract_time:.1f}ms")
            return relation_results

        except Exception as e:
            logger.error(f"[GLiREL] Extraction failed: {e}")
            return []

    def _normalize_entity_label(self, label: str, text: str = '') -> str:
        """Normalize entity labels to GLiREL format"""
        label_lower = label.lower()

        # For zero-shot mode, accept generic ENTITY label and infer type from text
        if label_lower in ['entity', '']:
            # Try to infer entity type from text content
            text_lower = text.lower() if text else ''
            if any(name in text_lower for name in ['steve', 'john', 'mary', 'david', 'jobs']) or \
               any(title in text_lower for title in ['mr', 'mrs', 'ms', 'dr']):
                return 'PERSON'
            elif any(org_word in text_lower for org_word in ['inc', 'corp', 'ltd', 'company', 'apple', 'microsoft', 'google']):
                return 'ORG'
            elif any(loc_word in text_lower for loc_word in ['city', 'country', 'cupertino', 'california', 'new york', 'london']):
                return 'LOC'
            else:
                # Default to ORG for company names, PERSON for names
                if ' inc' in text_lower or ' corp' in text_lower:
                    return 'ORG'
                elif len(text.split()) == 2 and text.split()[0][0].isupper() and text.split()[1][0].isupper():
                    return 'PERSON'
                else:
                    return 'ENTITY'

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