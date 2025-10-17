"""
Unified entity resolution and canonicalization for memory systems.

Handles pronoun resolution, entity canonicalization, and compound entity extraction
with single source of truth for all entity-related logic.
"""

import spacy
from typing import Optional, Dict, Set, Tuple, List


def _canon_entity_text(text: str) -> str:
    """Canonicalize entity text (lowercase, strip, normalize)"""
    if not text:
        return ""
    normalized = text.strip().lower()
    # Remove common articles and determiners
    for prefix in ["the ", "a ", "an "]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


# Pronoun mappings for quick lookup
_PRON_FIRST = {"i", "me", "my", "mine", "myself"}


class EntityResolver:
    """
    Single source of truth for entity resolution in memory systems.

    Responsibilities:
    - Pronoun resolution (I, you, he, she, etc.)
    - Entity canonicalization (normalization)
    - Compound entity extraction (noun chunks)
    - Entity mapping for graph construction
    """

    def __init__(self, user_eid: str, agent_eid: str, nlp: spacy.Language):
        """
        Initialize entity resolver.

        Args:
            user_eid: Canonical user entity ID (e.g., "user")
            agent_eid: Canonical agent entity ID (e.g., "pipecat")
            nlp: Spacy NLP model for linguistic analysis
        """
        self.user_eid = user_eid
        self.agent_eid = agent_eid
        self.nlp = nlp

    def resolve_pronoun(self, token) -> str:
        """
        Resolve pronoun to canonical entity.

        Args:
            token: Spacy token (PRON or other POS)

        Returns:
            Canonical entity string (user_eid, agent_eid, or lemma)
        """
        if token.pos_ != "PRON":
            return _canon_entity_text(token.text)

        person = token.morph.get("Person")
        person_val = person[0] if person else None

        if person_val == "1":
            # First person: I, me, my, mine → user
            return self.user_eid
        elif person_val == "2":
            # Second person: you, your, yours → agent
            return self.agent_eid
        else:
            # Third person or other: he, she, it, they → lemma
            return _canon_entity_text(token.lemma_)

    def canonicalize(self, text: str) -> str:
        """
        Canonicalize entity text.

        Args:
            text: Raw entity text

        Returns:
            Normalized entity string
        """
        return _canon_entity_text(text)

    def extract_entities(self, doc) -> Set[str]:
        """
        Extract all entities from spacy doc.

        Args:
            doc: Spacy Doc object

        Returns:
            Set of canonical entity strings
        """
        entities = set()

        # Named entities
        for ent in doc.ents:
            entities.add(_canon_entity_text(ent.text))

        # Noun chunks
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ("NOUN", "PROPN", "PRON"):
                resolved = self.resolve_pronoun(chunk.root)
                entities.add(resolved)

        return entities

    def build_entity_map(self, doc) -> Dict[int, str]:
        """
        Build mapping from token indices to canonical entities.

        Replaces memory_hotpath.py:443-505 (_build_entity_map)

        Args:
            doc: Spacy Doc object

        Returns:
            Dict mapping token indices to canonical entities
        """
        entity_map = {}
        entities = set()  # Collect entities as we build the map

        # Named entities
        for ent in doc.ents:
            norm_text = _canon_entity_text(ent.text)
            entities.add(norm_text)
            for token in ent:
                entity_map[token.i] = norm_text

        # Noun chunks
        for chunk in doc.noun_chunks:
            # Map noun chunks with pronouns to role-aware IDs
            if chunk.root.pos_ == "PRON":
                person = chunk.root.morph.get("Person")
                person_val = person[0] if person else None
                if person_val == "1":
                    # Actor (speaker)
                    entities.add(self.user_eid)
                    entity_map[chunk.root.i] = self.user_eid
                    continue
                if person_val == "2":
                    # Addressee (agent in dyadic chat)
                    entities.add(self.agent_eid)
                    entity_map[chunk.root.i] = self.agent_eid
                    continue

            chunk_text = _canon_entity_text(chunk.text)
            entities.add(chunk_text)
            entity_map[chunk.root.i] = chunk_text

        # Individual tokens
        for token in doc:
            if token.i not in entity_map:
                if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                    # Person-aware pronoun handling using UD morphology
                    if token.pos_ == "PRON":
                        person = token.morph.get("Person")
                        person_val = person[0] if person else None

                        if person_val == "1":
                            # First person: actor (speaker)
                            entity_text = self.user_eid
                        elif person_val == "2":
                            # Second person: addressee (agent in dyadic chat)
                            entity_text = self.agent_eid
                        elif person_val == "3":
                            # Third person: keep as-is (he, she, they)
                            entity_text = _canon_entity_text(token.lemma_)
                        else:
                            # Fallback: use old logic for pronouns without Person feature
                            entity_text = _canon_entity_text(token.text)
                            if entity_text in _PRON_FIRST:
                                entity_text = self.user_eid
                    else:
                        # NOUN/PROPN: use current logic
                        entity_text = _canon_entity_text(token.text)

                    entities.add(entity_text)
                    entity_map[token.i] = entity_text

        return entity_map

    def get_entity_with_context(
        self,
        chunk,
        include_compounds: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Extract entity with optional compound context.

        Replaces memory_hotpath.py:537-667 (_get_entity_with_context)

        Args:
            chunk: Spacy noun chunk
            include_compounds: Whether to extract compound entities

        Returns:
            (main_entity, compound_entities)
        """
        # Resolve main entity
        if chunk.root.pos_ == "PRON":
            main_entity = self.resolve_pronoun(chunk.root)
        else:
            main_entity = _canon_entity_text(chunk.root.text)

        compounds = []

        if include_compounds and len(chunk) >= 2:
            # Extract compound entities (multi-word phrases)
            compound_text = " ".join(token.text for token in chunk)
            canonical_compound = _canon_entity_text(compound_text)

            if canonical_compound != main_entity:
                compounds.append(canonical_compound)

        return main_entity, compounds

    def resolve_query_entities(self, query: str) -> List[str]:
        """
        Resolve entities in query text for graph retrieval.

        Replaces inline entity resolution in retrieval.py:223-401

        Args:
            query: User query text

        Returns:
            List of canonical entities for graph lookup
        """
        doc = self.nlp(query)
        entities = []

        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PRON":
                entity = self.resolve_pronoun(chunk.root)
            else:
                entity = _canon_entity_text(chunk.root.text)

            if entity and entity not in entities:
                entities.append(entity)

        return entities

    def get_entity_base(self, entity: str) -> str:
        """
        Get base form of entity for indexing.

        Examples:
        - "swimming in sea" -> "swimming" (prep pattern)
        - "red car" -> "car" (last word if multiple)
        - "machine learning" -> "learning" (last word)

        Strategy: If multi-word, check if contains prepositions (in, on, at, with, for).
        If yes, base is first word. Otherwise, base is last word (compound pattern).
        """
        words = entity.split()
        if len(words) <= 1:
            return entity

        # Check for prep pattern: "X in Y", "X on Y"
        preps = {"in", "on", "at", "with", "for", "from", "to", "by"}
        if any(w in preps for w in words[1:]):
            return words[0]  # "swimming" from "swimming in sea"

        # Otherwise assume compound: "machine learning" -> "learning"
        return words[-1]
