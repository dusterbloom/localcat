# Memory System Architecture Map

**Purpose**: Reference guide for any LLM to quickly understand and improve the memory system without wasting time on assumptions.

**Last Updated**: 2025-10-28
**Database Location**: `../data/memory.db` (from server/ directory)
**Current State**: Multi-source retrieval (graph+convo+summary, optional semantic), FTS-indexed conversations

---

## System Overview

The memory system extracts knowledge from conversations and stores it in a dual-storage architecture (SQLite + LMDB) for fast retrieval. It combines pattern-based extraction with semantic search (LEANN).

**Key Characteristics**:
- ⚡ **Fast**: ~40–60 ms average extraction with contextual modifier capture (sub‑200 ms p95 hot path)
- 🧠 **Smart**: Coreference resolution + contextual object enrichment (prep/adjective/compound)
- 🔁 **Alias-Aware**: Enriched triples indexed under both enriched and canonical base entities for retrieval fan-out
- 📊 **Dual Storage**: SQLite (persistence) + LMDB (fast adjacency lookups)
- 🔍 **Semantic Search**: Optional semantic sidecar (FAISS). Disabled by default; no LEANN code in current tree
- 🎯 **Quality**: Context-preserving triples reduce contradictions (e.g., swimming vs swimming in lakes)

> Reality check (Oct 28, 2025):
> - Semantic search uses an optional FAISS sidecar (`semantic_sidecar.py`); there is no LEANN implementation in the current codebase.
> - Coreference is implemented as a processor but not wired into `HotMemory` by default.
> - DSPy-assisted extraction is gated in code but the extractor lives under `server/archive/experimental/...` and is not imported by default.
> - Prosody is captured and can influence retrieval and confidence when enabled; communicating prosody/emotion to the LLM works best with headers-first injection.

---

## Architecture Components

### 1. Entry Points (Pipecat Integration)

```
bot.py (line 114)
  ↓ factory.create_voice_agent()
  ↓ Based on MEMORY_BACKEND env var:
  ├─→ MEMORY_BACKEND=hotmem → HotMemService (Pipecat MemoryService interface)
  └─→ MEMORY_BACKEND=hotpath → HotPathMemoryProcessor (FrameProcessor interface)
```

**Current Config**: `.env` currently sets `MEMORY_BACKEND=hotpath` (HotPathMemoryProcessor)

**HotMemService** (core/memory/hotmem_service.py)
- Pipecat-compatible MemoryService
- Provides tool-based interface (remember, recall, forget, search)
- Stores messages and retrieves context automatically

**HotPathMemoryProcessor** (core/memory/hotpath_processor.py)
- Pipecat FrameProcessor that sits in pipeline
- Processes TranscriptionFrame → extracts facts → injects bullets into context
- More integrated with pipeline flow

### 2. Core Extraction Pipeline

```
User Message
  ↓
HotMemory.process_turn() (memory_hotpath.py:115)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Coreference Resolution (if enabled)               │
│ ─────────────────────────────────────────────────────────── │
│ Input: "I'm working on my startup. The company grows."      │
│ Output: "I'm working on my startup. My startup grows."      │
│ Component: processors/coreference.py (FastCoref)            │
│ Config: MEMORY_COREFERENCE_ENABLED defaults to false;       │
│         not currently wired into HotMemory (UDExtractor     │
│         is constructed without processors). See             │
│         coreference_integration.py for planned wiring.      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Entity & Relation Extraction                      │
│ ─────────────────────────────────────────────────────────── │
│ Component: extractors/ud.py (UDExtractor) +                │
│            memory_hotpath._get_entity_with_context()       │
│ Method: 27 dependency pattern rules + contextual enrichers │
│ Time: ~5-6 ms (caps enforce predictability)                 │
│                                                              │
│ Patterns processed:                                         │
│   - nsubj (nominal subject)                                 │
│   - dobj/obj (direct object)                                │
│   - nmod:poss (possessive modifier)                         │
│   - compound (compound nouns)                               │
│   - [... 23 more patterns]                                  │
│                                                              │
│ Output:                                                     │
│   - Enriched triples: `(subject, relation, enriched_object)`│
│     • Preposition, adjective, compound modifiers preserved  │
│     • Caps: ≤5 adjectives, ≤3 compounds, ≤3 preps, ≤50 chars│
│   - Alias map: enriched → canonical root (e.g.,             │
│     "swimming in sea" → "swimming") for dual indexing       │
│   - Metrics: enrichment time, length, truncation count      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Filtering                                         │
│ ─────────────────────────────────────────────────────────── │
│ Component: memory_hotpath.py:747 (_is_meaningful_fact)     │
│                                                              │
│ Filters out:                                                │
│   - Stop entities: {it, this, that, there, here, been}     │
│   - Stop relations: {and, know, remember, say, tell, ...}  │
│   - Short entities: len < 2                                 │
│   - Low confidence: < threshold                             │
│                                                              │
│ Keeps: High-quality triples                                 │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: Storage                                           │
│ ─────────────────────────────────────────────────────────── │
│ Component: memory_store.py                                  │
│                                                              │
│ Dual Storage:                                               │
│   SQLite (../data/memory.db):                              │
│     - entity table: canonical entities                      │
│     - edge table: (src, rel, dst, weight, pos, neg, status)│
│     - mention table: conversation fragments                 │
│     - chunks_fts: FTS5 full-text search                    │
│                                                              │
│   LMDB (../data/graph.lmdb, 2.0GB):                        │
│     - Adjacency lists for O(1) neighbor lookups            │
│     - Alias mappings                                        │
│                                                              │
│ Confidence System:                                          │
│   - observe_edge(): Reinforces fact (weight → 1.0)         │
│   - negate_edge(): Demotes fact (weight → 0.0)             │
│   - Status: active (≥0.25) | stale (≥0.10) | archived      │
└─────────────────────────────────────────────────────────────┘
```

### 3. Retrieval Pipeline

```
User Query
  ↓
retrieve_bullets() (memory_hotpath.py + retrieval.py)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Multi-Source Retrieval (ordered by MEMORY_SOURCES)         │
│ ─────────────────────────────────────────────────────────── │
│ 1. Graph Retrieval (retrieval.py:46):                      │
│    - Extract entities from query                            │
│    - Lookup in entity_index (RAM) using enriched + base keys│
│    - Rank by: relation_priority × timestamp                 │
│    - Priority weights: lives_in(100), works_at(95), ...    │
│    ⚠️ NOTE: confidence weight ignored in ranking!           │
│                                                              │
│ 2. Conversation Retrieval (if MEMORY_CONVO_INDEX=true):    │
│    - SQLite FTS5 search on chunks_fts (eid='conversation')  │
│    - Indexed automatically when storing conversation turns  │
│    - Returns: recent conversation snippets matching query   │
│                                                              │
│ 3. Summary Retrieval (if enabled):                          │
│    - Returns: LLM-generated summaries                       │
│    - Currently DISABLED (.env:124)                          │
│                                                              │
│ 4. LEANN Semantic Search (if MEMORY_USE_LEANN=true):       │
│    - HNSW vector similarity search                          │
│    - Index: ../data/memory_vectors.leann                    │
│    - Backend: HNSW, complexity: 16                          │
└─────────────────────────────────────────────────────────────┘
  ↓
Format as bullets: "• [graph] your name is alice"
  ↓
Inject into LLM context (as system or user message)
```

---

## Database Schema

### SQLite Tables

```sql
-- Entity table (canonical entities)
CREATE TABLE entity(
  id TEXT PRIMARY KEY,        -- Hash of entity name
  name TEXT,                  -- Canonical form
  aliases TEXT,               -- Comma-separated aliases
  created_at INT,             -- Unix timestamp
  updated_at INT
);

-- Edge table (relationships/facts)
CREATE TABLE edge(
  id TEXT PRIMARY KEY,        -- Hash of (src, rel, dst)
  src TEXT,                   -- Subject entity
  rel TEXT,                   -- Relation type
  dst TEXT,                   -- Object entity
  weight REAL DEFAULT 1.0,    -- Confidence [0.0-1.0]
  pos INT DEFAULT 0,          -- Positive evidence count
  neg INT DEFAULT 0,          -- Negative evidence count
  status INT DEFAULT 1,       -- 1=active, 0=stale, -1=archived, -9=deleted
  updated_at INT              -- Last modification timestamp
);
CREATE INDEX idx_edge_src ON edge(src);
CREATE INDEX idx_edge_status ON edge(status);

-- Mention table (conversation storage)
CREATE TABLE mention(
  id TEXT PRIMARY KEY,
  eid TEXT,                   -- Entity ID this mentions
  text TEXT,                  -- Conversation text (max 500 chars)
  ts INT,                     -- Timestamp
  session_id TEXT,            -- Session identifier
  turn_id INT                 -- Turn number in session
);
CREATE INDEX idx_mention_eid ON mention(eid);

-- Conversation turn table (full conversation context)
CREATE TABLE conversation_turn(
  id TEXT PRIMARY KEY,        -- Hash(session_id|turn_id)
  text TEXT,                  -- Full conversation text
  session_id TEXT,            -- Session identifier
  turn_id INT,                -- Turn number
  ts INT                      -- Timestamp
);

-- Full-text search index (includes conversations + mentions)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text,                       -- Searchable text
  eid UNINDEXED,              -- Entity/source reference ('conversation', 'summary', or entity_id)
  rel UNINDEXED,
  dst UNINDEXED,
  ts UNINDEXED,
  tokenize='porter'           -- Porter stemming
);
-- Conversations indexed with eid='conversation' for convo retrieval
-- Summaries indexed with eid='summary' for summary retrieval
```

### LMDB Structure

```
Database: alias
  Key: "alias:{entity_name}"
  Value: canonical_entity_id (msgpack)

Database: adj (adjacency lists)
  Key: "adj:{src}|{rel}"
  Value: [dst1, weight1, ts1, pos1, neg1, status1, dst2, ...] (msgpack array)
```

---

## Configuration (.env)

### Memory Core Settings
```bash
MEMORY_ENABLED=true                    # Master switch
MEMORY_BACKEND=hotmem                  # hotmem | hotpath
MEMORY_BULLETS_MAX=3                   # Max bullets to inject
MEMORY_SOURCES=graph,convo,summary     # Retrieval sources (ordered)
```

### Storage Paths
```bash
MEMORY_SQLITE_PATH=../data/memory.db   # SQLite database
MEMORY_LMDB_PATH=../data/graph.lmdb    # LMDB adjacency cache
```

### Coreference Resolution
```bash
MEMORY_COREFERENCE_ENABLED=true        # Enable/disable
MEMORY_COREFERENCE_TIMEOUT_MS=50       # Max processing time
MEMORY_COREFERENCE_MIN_LENGTH=10       # Min text length to process
```

### LEANN Semantic Search
```bash
MEMORY_USE_LEANN=true                  # Enable/disable
MEMORY_LEANN_INDEX_PATH=../data/memory_vectors.leann
MEMORY_LEANN_BACKEND=hnsw              # Backend type
MEMORY_LEANN_COMPLEXITY=16             # Search complexity
MEMORY_REBUILD_LEANN_ON_SESSION_END=true
```

### Extraction Settings
```bash
MEMORY_CONFIDENCE_THRESHOLD=0.3        # Min confidence to extract
MEMORY_DECOMPOSE_CLAUSES=false         # Split compound sentences
```

### Intent Classification (Optional)
```bash
# Not set in .env, defaults to enabled in code
# INTENT_CLASSIFICATION_ENABLED=true
```

---

## Code Organization

### Core Module Structure
```
core/memory/
├── __init__.py                      # Exports HotMemService
├── hotmem_service.py                # Pipecat MemoryService interface
├── hotpath_processor.py             # Pipecat FrameProcessor interface
├── memory_hotpath.py                # Main HotMemory class (extraction + retrieval)
├── memory_store.py                  # Dual storage (SQLite + LMDB)
├── retrieval.py                     # Multi-source retrieval logic
├── nlp_manager.py                   # Shared spaCy model manager
├── session_tracker.py               # Session statistics
│
├── extractors/
│   ├── base.py                      # Extractor interface
│   └── ud.py                        # UDExtractor (27 patterns)
│
├── processors/
│   ├── base.py                      # TextProcessor interface
│   └── coreference.py               # CoreferenceProcessor (FastCoref)
│
└── [other supporting files...]
```

### Key Classes

**HotMemory** (memory_hotpath.py)
- Main extraction and retrieval orchestrator
- `process_turn()`: Extract facts from text
- `retrieve_bullets()`: Get relevant context
- Uses: UDExtractor, Retrieval, MemoryStore

**MemoryStore** (memory_store.py)
- Dual storage backend (SQLite + LMDB)
- `observe_edge()`: Reinforce fact
- `negate_edge()`: Demote conflicting fact
- `flush()`: Batch writes every N ops or M ms

**UDExtractor** (extractors/ud.py)
- 27 spaCy dependency pattern rules
- Optional coreference preprocessing
- Returns: (entities, triples, negation_count, doc)

**Retrieval** (retrieval.py)
- Multi-source retrieval coordinator
- Entity-based graph search
- FTS5 conversation search
- LEANN semantic search

---

## Current Quality Issues (Evidence-Based)

### Database Sample (191 edges)

**Good Extractions**:
```
Steve Jobs|founded|Apple|0.987        ← Excellent
dog Potola|is|5 years old|0.936       ← Good
you|lives_in|sardinia|0.806           ← Useful
```

**Noisy Extractions**:
```
we|talk_about|it|0.44                 ← Meaningless
we|talk_about|what|0.75               ← Low information
you|get|go|0.473                      ← Nonsense
you|is|part|0.3945                    ← Grammatically wrong
```

**Estimated Noise Rate**: 40-50%

### Known Gaps

1. **Retrieval ignores confidence** (retrieval.py:75)
   - Ranks by priority × timestamp only
   - Low-confidence facts rank equally with high-confidence

2. **No access tracking**
   - Schema lacks `access_count` and `last_accessed`
   - Can't identify which facts are useful
   - Can't prune never-used extractions

3. **Entity duplication**
   - "dog" vs "dog Potola" stored separately
   - "my startup" vs "startup" may not merge correctly

4. **No time decay**
   - Facts persist at initial confidence forever
   - Old facts don't fade unless conflicted

5. **Hardcoded priorities** (retrieval.py:56)
   - Cannot adapt to user patterns
   - One-size-fits-all ranking

---

## Testing Infrastructure

### Unit Tests
```
tests/unit/
├── test_hotmem_service.py           # HotMemService interface tests
├── test_memory_system.py            # End-to-end extraction & retrieval
├── test_hotmem_phase0.py            # Phase 0 functionality
├── test_hotmem_corrections.py       # Correction handling
├── test_hotmem_env.py               # Environment config tests
└── test_coreference_integration.py  # Coreference resolution tests
```

### Integration Tests
```
tests/integration/
├── test_hotmem_factory_integration.py  # Factory integration
├── test_summarization_integration.py   # Summarization (disabled)
└── test_intent_pipeline_simple.py      # Intent classification
```

### Running Tests
```bash
cd /Users/peppi/Dev/localcat/server

# Run specific test
pytest tests/unit/test_memory_system.py -v

# Run all memory tests
pytest tests/unit/test_hotmem*.py -v

# Run with markers
pytest -m fast  # Fast tests only
pytest -m ci    # CI tests only
```

---

## Optimization Opportunities

### Short-Term (High Impact, Low Effort)

1. **Fix retrieval ranking** (retrieval.py:75)
   - Include confidence in scoring: `score = priority × confidence × recency`
   - Estimated impact: +10-15% precision

2. **Add access tracking**
   - ALTER TABLE edge ADD COLUMN access_count INT DEFAULT 0
   - ALTER TABLE edge ADD COLUMN last_accessed INT
   - Track on every retrieval

3. **Remove low-confidence edges**
   - Filter: `weight < 0.5 AND access_count == 0 AND age > 30d`
   - Run weekly pruning job

### Medium-Term (GEPA Integration)

4. **Optimize extraction patterns**
   - Use GEPA to learn which of 27 patterns are high-quality
   - Disable noisy patterns (e.g., "compound", "and")
   - Evolve confidence weights per pattern

5. **Entity consolidation**
   - DSPy module for semantic similarity
   - Merge duplicates: "dog" + "dog Potola" → "dog Potola"
   - GEPA learns consolidation rules

6. **Adaptive priorities**
   - Log: (query, retrieved, used)
   - GEPA learns relation priorities from access patterns
   - Update priority weights periodically

### Long-Term (Quality Compound)

7. **Full pipeline optimization**
   - GEPA optimizes coreference prompts
   - GEPA optimizes entity extraction rules
   - GEPA optimizes relation mapping
   - GEPA optimizes filtering thresholds
   - Result: 60% → 90%+ precision over time

---

## GEPA Integration Plan

### Architecture
```
TIER 1 (Real-Time):
  Fast extraction (~1-2ms) → Store with traces

TIER 2 (Offline):
  GEPA analyzes traces → Evolves pipeline config → Refines graph

TIER 1 (Next Session):
  Uses evolved config → Better extraction!
```

### What GEPA Optimizes

1. **Coreference prompts** (DSPy)
   - Improve "company" → "startup" resolution
   - Handle cross-sentence references

2. **Pattern selection** (GEPA)
   - Which of 27 UD patterns to use
   - Confidence weights per pattern

3. **Entity rules** (GEPA)
   - Stop words, min length, normalization
   - Possessive handling

4. **Relation mapping** (GEPA)
   - UD dep → semantic relation
   - Context-aware selection

5. **Filtering thresholds** (GEPA)
   - Quality thresholds per relation
   - Duplicate detection strategies

### Trajectory Format
```python
{
  "conversation": ["My name is Alice", "I work at Google"],
  "extracted": [("you", "name", "alice"), ("you", "works_at", "google")],
  "retrieved": ["• your name is alice"],
  "quality_score": 0.8,
  "stage_metrics": {
    "coref_accuracy": 0.9,
    "extraction_precision": 0.85,
    "retrieval_recall": 0.7
  }
}
```

---

## How to Improve This System

### Step-by-Step Process

1. **Measure Baseline**
   - Run test_memory_system.py
   - Sample 100 edges, manually label quality
   - Calculate precision/recall

2. **Identify Bottleneck**
   - Which stage has lowest quality?
   - Coreference? Extraction? Filtering? Retrieval?

3. **Design Fix**
   - ONE specific improvement
   - Evidence-based (not assumption-based)

4. **Implement & Test**
   - Write test first
   - Implement fix
   - Measure improvement

5. **Iterate**
   - Next bottleneck
   - Compound improvements

### Quick Start Commands

```bash
# Activate venv
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate

# Check database
sqlite3 ../data/memory.db "SELECT COUNT(*) FROM edge;"

# Run extraction test
python tests/unit/test_memory_system.py

# Check logs
tail -f core/memory/.logs/hotmem.log

# Sample recent edges
sqlite3 ../data/memory.db "SELECT src, rel, dst, weight FROM edge ORDER BY updated_at DESC LIMIT 20;"
```

---

## Critical Invariants (Do NOT Break)

1. **Latency**: Extraction + retrieval must stay <5ms combined
2. **Schema**: Changing edge table breaks LMDB sync
3. **Coreference**: Timeout at 50ms (hard limit for real-time)
4. **Storage**: Use batched writes (flush_if_needed)
5. **Tests**: All tests must pass before merging

---

## Questions to Ask Before Changing Anything

1. ✅ **Have you read the actual code?** (Not assumed)
2. ✅ **Have you checked the database?** (../data/memory.db)
3. ✅ **Have you run the tests?** (pytest tests/unit/test_memory*.py)
4. ✅ **Have you measured current quality?** (baseline metrics)
5. ✅ **Do you have evidence of the problem?** (sample data)
6. ✅ **Will this stay under 5ms?** (latency budget)
7. ✅ **Does this solve a real problem?** (not imagined)

---

## End of Map

This document should be treated as **ground truth** for the memory system. Any changes to architecture should update this map. Any LLM working on this system should read this FIRST before making assumptions.

**Last verified**: 2025-09-29 with actual database and code inspection.
