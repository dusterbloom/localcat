# NeuralOS ↔ LocalCat Integration Plan

This document captures the current strategy for unifying NeuralOS and
LocalCat into a single, coherent memory OS that works for personal
memory, general facts, episodes, and documents; and can be driven by
different SLM backends (Direct MLX, HTTP/OpenAI, etc.).

The goal is to make future work incremental, SOLID, and DRY.

---

## 1. Goals

- **General Memory OS**: NeuralOS should handle personal facts, world
  facts, episodes, and documents with the same quality and semantics.
- **Single Source of Truth**: Avoid multiple independent memory
  systems. NeuralOS is the memory OS; other systems are storage
  backends or frontends.
- **SLM-Agnostic**: NeuralOS should work with:
  - LocalCat's DirectMLX SLM (Gemma/Qwen3).
  - Tiny Qwen 0.5B for fact extraction.
  - OpenAI/litellm backends in other deployments.
- **Meta-Memory & Preferences**: Maintain the high-quality behaviour
  already achieved in NeuralOS:
  - "Did I tell you X?" / "Did we ever talk about Y?"
  - Signed preferences (favorite vs least vs dislike).
  - Entity-aware personal summaries.
- **Reuse LocalCat's Storage Strengths**: Leverage LMDB/SQLite
  infrastructure for persistence and speed without duplicating
  semantics.

---

## 2. Current Systems (High-Level)

### 2.1 NeuralOS (this repo)

- **Core abstractions**:
  - `Brain` / `QwenBrain` (LLM backend).
  - `Router` / `SentenceRouter` (embeddings).
  - `Vault` / `FactStore` / `SemanticStore` / `ColdStore`.
  - `JitSemanticEngine` (tier selection + context).
  - `Scribe` (intent/entity intent).
  - Meta-memory, signed preferences, entity summaries.
- **Metadata on `MemoryItem`**:
  - `role`: `"identity" | "preference" | "meta" | "context"`.
  - `source`: `"user" | "extracted" | "system"`.
  - `preference_polarity`: `+1` / `-1` for likes/dislikes.
  - `topics`: `["music", "food", "books", "location", "pets", ...]`.
  - `reliability`, `superseded`, etc.

### 2.2 LocalCat (server/)

- **SLM-centric pipeline**:
  - Voice → STT → intent → SLM (Gemma/Qwen3) → TTS.
- **Current memory system**:
  - `MemoryOrchestrator` coordinating:
    - `FactExtractor` (triples).
    - `MemoryRetriever` (entity + recency).
    - `MemoryStore` (SQLite + LMDB).
  - HotMem tools wired into the LLM for remember/recall/search.

---

## 3. Desired Architecture (Unified System)

We want one conceptual memory system and one SLM per deployment:

1. **Frontends**: LocalCat voice agent, REPL, HTTP APIs.
2. **Reasoning Layer**: SLM + tool router (model-agnostic).
3. **Memory OS**: NeuralOS, with different storage backends.

Key principles:

- NeuralOS is the **only** memory OS the SLM sees (via tools and
  context).
- LocalCat's LMDB/SQLite store becomes a **backend** for NeuralOS,
  not a separate memory universe.
- A small SLM (Qwen2.5-0.5B) remains dedicated to fact extraction.

---

## 4. Components and Adapters

### 4.1 `LocalCatBrain` (SLM adapter)

Implement a `Brain` subclass that wraps LocalCat's existing LLM
service (DirectMLX or HTTP):

```python
from neuralos.brain import Brain

class LocalCatBrain(Brain):
    def __init__(self, llm_service):
        self.llm = llm_service

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        return self.llm.generate(prompt, max_tokens=max_new_tokens, **kwargs)
```

NeuralOS can then use `LocalCatBrain` instead of `QwenBrain` when
embedded inside LocalCat.

### 4.2 `LocalCatVault` (storage adapter)

Implement a Vault variant that uses LocalCat's `MemoryStore` as a
backend:

- On **write**:
  - For personal facts, preferences, episodes: mirror `MemoryItem`
    fields into triples/edges in LMDB (`MemoryStore.observe_edge`).
  - Maintain `topics` and `role` as part of metadata/provenance.
- On **read**:
  - Use existing LMDB indices (entity index, recency) to assist
    NeuralOS's retrieval when needed.

This preserves LocalCat's performance and durability while keeping
NeuralOS semantics central.

### 4.3 Small Extractor SLM (Qwen 0.5B)

Keep a dedicated tiny model for fact extraction:

- Use spaCy/UD/LocalCat's NLP pipelines for deterministic extraction.
- Use Qwen2.5-0.5B for:
  - Ambiguous or complex sentences.
  - Canonical key/question generation.
  - Deciding if something is a fact at all.

This remains entirely within NeuralOS's `FactExtractor` and does not
depend on the main SLM used for conversation.

---

## 5. Tool Integration (SLM ↔ NeuralOS)

Instead of exposing two memory APIs (HotMem tools + NeuralOS), we want
the SLM to only see **NeuralOS-style tools**:

- `neuralos_remember(text, user_id)`:
  - TEACH path; lets the user explicitly store crucial facts.
- `neuralos_query(query, user_id)`:
  - ASK path; uses NeuralOS.answer() with full JIT/meta/prefs logic.
- `neuralos_introspect(query, user_id)`:
  - Meta-memory path for “Did I tell you X?”, “Did we ever talk about Y?”.
- `neuralos_forget(id_or_text, user_id)`:
  - Correction path; drives reliability/superseding.

LocalCat's HotMem tools can be refactored or gradually mapped to these
operations. LMDB remains the persistence mechanism underneath, not a
second high-level API.

---

## 6. Phased Plan

### Phase 1 – Injection Points

1. **Brain injection**:
   - DONE in NeuralOS: `NeuralOS.__init__` now accepts an optional
     `brain: Brain` argument (and keeps `brain_id` for defaults); when
     omitted it still constructs a `QwenBrain` as before.
   - Next in LocalCat: implement `LocalCatBrain` wrapping the existing
     LLM service and pass it into `NeuralOS(brain=LocalCatBrain(...))`.

2. **Vault backend choice**:
   - DONE in NeuralOS: `NeuralOS.__init__` now accepts an optional
     `vault` argument plus an optional `fact_extractor`; when `vault`
     is not provided it continues to honour the existing
     `get_neuralos_config()` / `FalkorDBVault` / `Vault` selection.
   - Future work: add a `LocalCatVault` implementation that uses
     LocalCat's `MemoryStore` and pass it via `vault=LocalCatVault(...)`.

### Phase 2 – LocalCat integration

3. In LocalCat server:
   - Construct a single `NeuralOS` instance at startup with:
     - `brain=LocalCatBrain(llm_service)`.
     - `vault=FalkorDBVault` or in-memory `Vault` initially.

4. Feed all user turns into NeuralOS:

```python
neuralos.process_turn(text, user_id=session_id)
```

5. Expose NeuralOS tools to the LLM:
   - Register `neuralos_remember`, `neuralos_query`,
     `neuralos_introspect` as tools in the same way HotMem tools are.

6. A/B:
   - For a while, log both HotMem and NeuralOS answers for “memory”
     queries and compare.

### Phase 3 – LocalCatVault and Consolidation

7. Implement `LocalCatVault` in NeuralOS:
   - Reuse `MemoryStore` and LMDB for persistence.

8. Gradually remove duplicated logic:
   - Let NeuralOS own:
     - “What do you know about me/us/X?”
     - “Did I tell you X?/Did we talk about Y?”
     - Preferences (favorite/least/dislikes).
     - Corrections.
   - Keep LocalCat's pipeline code focused on:
     - Voice/STT/TTS.
     - Tool wiring.
     - Performance monitoring.

---

## 7. Review & Iteration

Before making any large changes, we should:

- Run existing NeuralOS tests:
  - `pytest tests/test_neuralos_personal.py`
  - `pytest tests/test_end_to_end_system.py`
  - `pytest tests/test_repl_entity_awareness.py`
- Run LocalCat's memory tests:
  - `pytest server/tests/test_memory_*`
  - `pytest server/tests/test_integration_*`
- Perform REPL/voice sessions that match real usage:
  - Identity, preferences, locations, pets.
  - “Did I tell you X?”, “Did we talk about Y?”
  - Document/file questions where applicable.

Ablation is key: we only remove modules (e.g., T5/deberta) if tests +
logs show no regression.

This document should be kept up to date as we refine the integration.
