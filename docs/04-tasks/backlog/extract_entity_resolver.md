# Extract EntityResolver to Eliminate DRY Violation

**Priority**: Critical (DRY Violation)
**Effort**: 3 days
**Assigned To**: Memory Systems Specialist

## Problem Statement

Entity resolution logic is duplicated across 3 locations with ~300 lines of duplicate code. This causes:
- Bug fixes requiring changes in 3 places
- Inconsistent behavior between modules
- High maintenance burden

**Duplicate Locations**:
1. `memory_hotpath.py:443-505` (_build_entity_map)
2. `memory_hotpath.py:537-667` (_get_entity_with_context)
3. `retrieval.py:223-401` (_graph_collect_candidates - inline entity resolution)

**Example Duplication**:
```python
# Repeated 3+ times across files:
if token.pos_ == "PRON":
    person = token.morph.get("Person")
    person_val = person[0] if person else None
    if person_val == "1":
        entity_text = self.user_eid
    elif person_val == "2":
        entity_text = self.agent_eid
    # ... repeated logic
```

## Impact

- **Maintenance**: 3x effort for bug fixes
- **Consistency**: Risk of divergent implementations
- **Code Quality**: 300 lines of unnecessary duplication

## Success Metrics

- ✓ Single EntityResolver class replaces all 3 duplications
- ✓ ~300 lines of duplicate code eliminated
- ✓ All existing memory tests pass
- ✓ Entity resolution behavior consistent across modules
- ✓ Performance maintained (<50ms memory retrieval)

## Implementation Approach

### Step 1: Create EntityResolver Class

```python
# server/core/memory/entity_resolver.py (NEW FILE)
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

    def build_entity_map(self, doc) -> Dict[str, str]:
        """
        Build mapping from raw text to canonical entities.

        Replaces memory_hotpath.py:443-505 (_build_entity_map)

        Args:
            doc: Spacy Doc object

        Returns:
            Dict mapping raw text spans to canonical entities
        """
        entity_map = {}

        # Map named entities
        for ent in doc.ents:
            entity_map[ent.text] = _canon_entity_text(ent.text)

        # Map noun chunks
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PRON":
                canonical = self.resolve_pronoun(chunk.root)
            else:
                canonical = _canon_entity_text(chunk.root.text)

            entity_map[chunk.text] = canonical

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
```

### Step 2: Refactor memory_hotpath.py

```python
# server/core/memory/memory_hotpath.py
from .entity_resolver import EntityResolver

class HotMemory:
    def __init__(self, ...):
        # ... existing init code ...

        # NEW: Initialize entity resolver
        self.entity_resolver = EntityResolver(
            user_eid=self.user_eid,
            agent_eid=self.agent_eid,
            nlp=self.nlp
        )

    def _build_entity_map(self, doc) -> Dict[str, str]:
        """REPLACE with EntityResolver method"""
        return self.entity_resolver.build_entity_map(doc)

    def _get_entity_with_context(self, chunk, include_compounds=True):
        """REPLACE with EntityResolver method"""
        return self.entity_resolver.get_entity_with_context(
            chunk,
            include_compounds
        )
```

### Step 3: Refactor retrieval.py

```python
# server/core/memory/retrieval.py
from .entity_resolver import EntityResolver

class MemRetrieval:
    def __init__(self, ...):
        # ... existing init code ...

        # NEW: Initialize entity resolver
        self.entity_resolver = EntityResolver(
            user_eid=self.user_eid,
            agent_eid=self.agent_eid,
            nlp=self.nlp
        )

    def _graph_collect_candidates(self, query: str, ...):
        """REFACTOR: Use EntityResolver for entity extraction"""

        # REPLACE inline entity resolution (lines 223-401)
        entities = self.entity_resolver.resolve_query_entities(query)

        # ... rest of graph collection logic using resolved entities ...
```

## Testing Requirements

### Unit Tests

```python
# server/tests/unit/memory_resolver/test_entity_resolver.py (NEW)
import pytest
from server.core.memory.entity_resolver import EntityResolver

def test_pronoun_resolution_first_person():
    """I, me, my → user_eid"""
    resolver = EntityResolver(user_eid="alice", agent_eid="bot", nlp=nlp)

    doc = nlp("I like pizza")
    token = doc[0]  # "I"

    assert resolver.resolve_pronoun(token) == "alice"

def test_pronoun_resolution_second_person():
    """you, your → agent_eid"""
    resolver = EntityResolver(user_eid="alice", agent_eid="bot", nlp=nlp)

    doc = nlp("you are helpful")
    token = doc[0]  # "you"

    assert resolver.resolve_pronoun(token) == "bot"

def test_entity_canonicalization():
    """Normalize entities consistently"""
    resolver = EntityResolver(user_eid="user", agent_eid="agent", nlp=nlp)

    assert resolver.canonicalize("The Pizza") == "pizza"
    assert resolver.canonicalize("  San Francisco  ") == "san francisco"

def test_extract_entities_from_doc():
    """Extract all entities from text"""
    resolver = EntityResolver(user_eid="user", agent_eid="agent", nlp=nlp)

    doc = nlp("I live in San Francisco with my cat")
    entities = resolver.extract_entities(doc)

    assert "user" in entities  # "I" resolved
    assert "san francisco" in entities
    assert "cat" in entities

def test_build_entity_map():
    """Build mapping from text to canonical entities"""
    resolver = EntityResolver(user_eid="user", agent_eid="agent", nlp=nlp)

    doc = nlp("I visited New York")
    entity_map = resolver.build_entity_map(doc)

    assert entity_map["I"] == "user"
    assert entity_map["New York"] == "new york"

def test_get_entity_with_context_compounds():
    """Extract compound entities (multi-word)"""
    resolver = EntityResolver(user_eid="user", agent_eid="agent", nlp=nlp)

    doc = nlp("San Francisco is beautiful")
    chunk = list(doc.noun_chunks)[0]  # "San Francisco"

    main, compounds = resolver.get_entity_with_context(chunk, include_compounds=True)

    assert main == "francisco"  # Root of chunk
    assert "san francisco" in compounds  # Full compound

def test_resolve_query_entities():
    """Resolve entities in user query"""
    resolver = EntityResolver(user_eid="user", agent_eid="agent", nlp=nlp)

    entities = resolver.resolve_query_entities("Tell me about my cat in San Francisco")

    assert "user" in entities  # "my" → user
    assert "cat" in entities
    assert "san francisco" in entities
```

### Integration Tests

```python
# server/tests/integration/test_entity_consistency.py (NEW)
def test_entity_resolution_consistency_across_modules():
    """Verify HotMemory and MemRetrieval use same entity resolution"""

    # Store fact via HotMemory
    hot = HotMemory(user_eid="alice", agent_eid="bot")
    hot.observe("I like pizza")

    # Retrieve via MemRetrieval
    retrieval = MemRetrieval(user_eid="alice", agent_eid="bot")
    bullets = retrieval.retrieve("What do I like?")

    # Verify entity "alice" (from "I") is consistent
    assert any("alice" in bullet or "like pizza" in bullet for bullet in bullets)
```

### Regression Tests

```python
# server/tests/unit/test_memory_system.py (EXISTING - must still pass)
# All existing tests must pass after refactoring
pytest server/tests/unit/test_memory_system.py -v
```

## Performance Validation

```bash
# Benchmark memory operations before/after
pytest server/tests/performance/test_memory_retrieval.py -v

# Expected: No regression in retrieval time (<50ms)
# Expected: Possible improvement due to code efficiency
```

## Files to Modify

1. **server/core/memory/entity_resolver.py** (NEW)
   - EntityResolver class with all methods
   - ~200 lines (consolidates 300 duplicate lines)

2. **server/core/memory/memory_hotpath.py**
   - Remove _build_entity_map implementation (lines 443-505)
   - Remove _get_entity_with_context implementation (lines 537-667)
   - Delegate to EntityResolver

3. **server/core/memory/retrieval.py**
   - Remove inline entity resolution (lines 223-401)
   - Use EntityResolver.resolve_query_entities()

4. **server/tests/unit/memory_resolver/** (NEW)
   - test_entity_resolver.py (comprehensive unit tests)

5. **server/tests/integration/** (EXISTING)
   - test_entity_consistency.py (NEW - cross-module consistency)

## Definition of Done

- [ ] EntityResolver class implemented with all methods
- [ ] memory_hotpath.py refactored to use EntityResolver
- [ ] retrieval.py refactored to use EntityResolver
- [ ] ~300 lines of duplicate code removed
- [ ] All existing tests pass (regression check)
- [ ] New unit tests pass (entity resolver behavior)
- [ ] Integration tests pass (cross-module consistency)
- [ ] Performance tests pass (<50ms retrieval maintained)
- [ ] Code review completed
- [ ] Documentation added to entity_resolver.py

## Delegation Command

```bash
# Manager delegates to Memory Systems Specialist
droid exec memory-systems-specialist --auto medium -f tasks/extract_entity_resolver.md
```

---

**Related Issues**: Part of technical debt cleanup (Phase 1, Critical Priority)
**Blocks**: None (standalone refactor)
**References**: Tech debt guardian report - Critical Issue #2 (DRY violation)
