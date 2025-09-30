# GEPA-Powered Graph Refinement Architecture

## Executive Summary

A two-tier extraction system that combines fast real-time pattern matching with offline SOTA refinement using GEPA (Genetic-Pareto optimizer) to achieve both **latency edge** AND **quality edge**.

## Core Insight

**Current**: Fast extraction (~1-2ms) but quality never improves
**Proposed**: Fast extraction PLUS reflective evolution that learns from conversation traces

## Architecture

### Tier 1: Real-Time Fast Extraction

**Purpose**: Immediate graph updates for instant retrieval (0 latency penalty)

```python
# On every turn
def process_turn(text: str, session_id: str, turn_id: int):
    # Fast pattern extraction (~1-2ms)
    entities, triples = extract_with_patterns(text)

    # Store rough graph + conversation trace
    store_rough_graph(triples, confidence="initial")
    store_conversation_trace(session_id, turn_id, text, triples)

    # Retrieve from current best graph (rough or refined)
    bullets = retrieve(text, entities)
    return bullets, triples
```

**Characteristics**:
- Ultra-fast (<2ms extraction)
- Permissive (captures everything, some noise)
- Immediate availability
- Stores full conversation trace for offline refinement

### Tier 2: Offline GEPA Refinement

**Purpose**: Analyze traces, evolve extraction, refine graph (no latency impact)

```python
# During idle time (e.g., every 6 hours or after 100 new conversations)
class GEPAGraphRefiner:
    """
    Uses GEPA to analyze conversation traces and evolve graph extraction.

    GEPA's reflective loop:
    1. Sample trajectories (conversation → extraction → retrieval → quality)
    2. Reflect on failures in natural language
    3. Propose improvements (prompt mutations)
    4. Test mutations on held-out traces
    5. Select Pareto-optimal configurations
    """

    def refine_cycle(self):
        # 1. Collect trajectories
        trajectories = self.collect_recent_trajectories(limit=100)

        # 2. GEPA optimization loop
        evolved_config = gepa.optimize(
            trajectories=trajectories,
            metrics=["precision", "recall", "retrieval_quality"],
            constraints=["latency_ms < 5"],  # Keep it fast
            iterations=50
        )

        # 3. Re-extract with evolved config
        refined_graph = self.reextract_with_evolved_config(
            traces=trajectories,
            config=evolved_config
        )

        # 4. Deploy refined graph + evolved extraction
        self.deploy_refined_graph(refined_graph)
        self.update_extraction_config(evolved_config)
```

### Trajectory Structure

**What GEPA sees:**

```python
trajectory = {
    # Input context
    "session_id": "user123_session_5",
    "conversation": [
        {"turn": 1, "text": "My name is Alice"},
        {"turn": 2, "text": "I work at Google as a software engineer"},
        {"turn": 3, "text": "I'm working on a search engine project"},
        {"turn": 4, "text": "Tell me about my project"}  # Retrieval query
    ],

    # Real-time extraction output
    "extracted_rough": [
        (1, "you", "name", "alice", 0.95),
        (2, "you", "works_at", "google", 0.90),
        (2, "you", "role", "software engineer", 0.85),
        (3, "you", "works_on", "search engine project", 0.90),
        (3, "you", "works_on", "project", 0.90),  # DUPLICATE!
    ],

    # Retrieval behavior
    "retrieved_bullets": [
        "• [graph] your name is alice",
        "• [graph] you works at google"
        # Missing: project information!
    ],

    # Ground truth / quality signal
    "user_query": "Tell me about my project",
    "expected_retrieval": ["search engine project", "project details"],
    "quality_score": 0.3,  # Low - didn't retrieve project info

    # Reflection target
    "failure_analysis": {
        "issue": "Failed to retrieve project information when asked about 'my project'",
        "root_causes": [
            "Entity duplication: 'search engine project' vs 'project' stored as separate entities",
            "Low priority: 'works_on' relation has priority 50, below retrieval threshold",
            "Coreference miss: 'my project' didn't match 'search engine project' or 'project'"
        ]
    }
}
```

## GEPA Reflective Evolution Loop

### Step 1: Analyze Trajectories

GEPA examines batches of trajectories to find patterns in failures:

```
GEPA Reflection:
"Across 47 trajectories, I observe that:
1. 'works_on' relations are frequently extracted but rarely retrieved (12% retrieval rate)
2. Project-related entities have 3.2x duplication rate vs other entities
3. Coreference resolution fails on possessives ('my X' → 'X') in 68% of cases
4. When users ask about 'my [entity]', retrieval precision drops to 0.31

Hypothesis:
- Priority of 'works_on' is too low
- Entity consolidation is needed for project-related terms
- Need coreference-aware entity matching in retrieval"
```

### Step 2: Propose Mutations

GEPA generates candidate improvements:

```python
# Mutation 1: Adjust priorities
mutation_1 = {
    "type": "priority_adjustment",
    "changes": {
        "works_on": 50 → 85,  # Increase project relation priority
        "role": 60 → 70
    },
    "rationale": "work-related info frequently queried but under-prioritized"
}

# Mutation 2: Entity consolidation rules
mutation_2 = {
    "type": "consolidation_rule",
    "rule": "if entity_b contains entity_a and len(entity_a) < 15: merge(entity_a, entity_b, keep=longer)",
    "example": "project" + "search engine project" → "search engine project",
    "rationale": "reduce duplication by merging substring entities"
}

# Mutation 3: Coreference-aware retrieval
mutation_3 = {
    "type": "retrieval_enhancement",
    "logic": "if query contains possessive ('my', 'your'), expand to: [entity, 'you has entity', 'you works_on entity']",
    "rationale": "handle possessive queries better"
}
```

### Step 3: Test Mutations

```python
# Re-run extraction on held-out traces with each mutation
for mutation in mutations:
    config_variant = apply_mutation(base_config, mutation)
    results = test_on_holdout(config_variant, test_traces)

    score = {
        "precision": results.precision,
        "recall": results.recall,
        "retrieval_f1": results.retrieval_f1,
        "latency_ms": results.avg_latency,
        "graph_size": results.unique_entities  # Smaller is better (less duplication)
    }

    pareto_population.add(config_variant, score)
```

### Step 4: Pareto Selection

Select configurations on the Pareto frontier:

```
Pareto Frontier:
Config A: precision=0.91, recall=0.78, latency=1.8ms, graph_size=450  ← Balanced
Config B: precision=0.95, recall=0.72, latency=2.1ms, graph_size=380  ← High precision
Config C: precision=0.88, recall=0.85, latency=1.5ms, graph_size=520  ← High recall
```

Keep all Pareto-optimal configs for different scenarios.

### Step 5: Deploy Evolved Configuration

```python
# Update live system
def deploy_evolved_config(pareto_config):
    # Update extraction priorities
    update_relation_priorities(pareto_config.priorities)

    # Update entity consolidation rules
    update_consolidation_rules(pareto_config.consolidation)

    # Update retrieval logic
    update_retrieval_matcher(pareto_config.retrieval)

    # Re-extract historical traces with new config
    refined_graph = reextract_all_traces(pareto_config)

    # Switch to refined graph
    swap_active_graph(refined_graph)

    logger.info(f"Deployed evolved config: {pareto_config.metrics}")
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ LIVE CONVERSATION                                             │
├──────────────────────────────────────────────────────────────┤
│ User: "My name is Alice, I work at Google"                   │
│   ↓                                                           │
│ Fast Extraction (current best config)                        │
│   ↓                                                           │
│ Rough Graph: [(you, name, alice), (you, works_at, google)]   │
│   ↓                                                           │
│ Store: SQLite (conversation) + LMDB (rough graph)            │
│   ↓                                                           │
│ Retrieve: Use refined graph if available, else rough          │
└──────────────────────────────────────────────────────────────┘
                            ↓
            (Every 6 hours or 100 conversations)
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ GEPA REFINEMENT CYCLE                                         │
├──────────────────────────────────────────────────────────────┤
│ 1. Load: Recent conversation traces (last 100 sessions)      │
│ 2. Analyze: Build trajectories with quality scores           │
│ 3. Reflect: GEPA identifies patterns in failures             │
│ 4. Mutate: Generate candidate config improvements            │
│ 5. Test: Re-extract traces with each mutation                │
│ 6. Select: Pareto-optimal configs                            │
│ 7. Deploy: Update extraction logic + build refined graph     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ REFINED GRAPH (High Quality)                                 │
├──────────────────────────────────────────────────────────────┤
│ • Consolidated entities (no duplicates)                       │
│ • Optimized priorities (learned from usage)                   │
│ • Better coreference resolution                               │
│ • Higher retrieval precision                                  │
│                                                               │
│ Used for: Retrieval in next conversations                    │
└──────────────────────────────────────────────────────────────┘
```

## Schema: Dual Graph Storage

### Rough Graph (Real-Time)

```sql
CREATE TABLE rough_edge(
  id TEXT PRIMARY KEY,
  src TEXT,
  rel TEXT,
  dst TEXT,
  confidence REAL,
  session_id TEXT,
  turn_id INT,
  created_at INT,
  status INT DEFAULT 1  -- 1=active, 0=superseded_by_refined
);
```

### Refined Graph (GEPA Output)

```sql
CREATE TABLE refined_edge(
  id TEXT PRIMARY KEY,
  src TEXT,
  rel TEXT,
  dst TEXT,
  confidence REAL,
  evidence_count INT,  -- How many traces support this
  source_sessions TEXT,  -- JSON array of contributing sessions
  created_at INT,
  refined_at INT,
  gepa_iteration INT  -- Which GEPA cycle produced this
);
```

### Retrieval Strategy

```python
def retrieve_bullets(query, entities):
    # Prefer refined graph
    refined_bullets = query_refined_graph(query, entities)
    if len(refined_bullets) >= 3:
        return refined_bullets

    # Fallback to rough graph for recent facts
    rough_bullets = query_rough_graph(query, entities, limit=3 - len(refined_bullets))

    return refined_bullets + rough_bullets
```

## Quality Metrics Evolution

### Before GEPA (Baseline)

| Metric | Value |
|--------|-------|
| Entity duplicates | 30-40% |
| Retrieval precision | 60% |
| Retrieval recall | 70% |
| Graph size growth | Linear with conversations |
| Never-accessed facts | 50% |

### After GEPA (Target)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Entity duplicates | <5% | 6-8x better |
| Retrieval precision | >90% | 1.5x better |
| Retrieval recall | >85% | 1.2x better |
| Graph size growth | Logarithmic | Stabilizes |
| Never-accessed facts | <10% | 5x better |

## Implementation Phases

### Phase 1: Dual Graph Infrastructure (Week 1)
- [x] Add refined_edge table
- [x] Update retrieval to check refined first
- [x] Store conversation traces with session metadata
- [x] Build trajectory collector

### Phase 2: GEPA Integration (Week 2)
- [ ] Integrate GEPA library
- [ ] Define trajectory format for graph extraction
- [ ] Implement reflection module (analyze failures)
- [ ] Implement mutation module (propose improvements)
- [ ] Implement testing harness (evaluate mutations)

### Phase 3: Refinement Pipeline (Week 3)
- [ ] Build re-extraction engine (apply evolved config to traces)
- [ ] Build graph consolidation module
- [ ] Build deployment mechanism (swap graphs)
- [ ] Add monitoring/metrics

### Phase 4: Production Optimization (Week 4)
- [ ] Tune GEPA parameters (population size, iterations)
- [ ] Optimize re-extraction performance
- [ ] Add incremental refinement (don't reprocess all traces)
- [ ] Build quality dashboard

## Key Advantages

### 1. No Latency Penalty
- Real-time extraction unchanged (~1-2ms)
- GEPA runs offline during idle time
- Refinement is invisible to users

### 2. Self-Improving System
- Gets smarter with every conversation
- Learns user-specific patterns
- No manual tuning required

### 3. Quality Compounds Over Time
```
Week 1: Precision 60% (baseline)
Week 2: Precision 68% (first GEPA cycle)
Week 4: Precision 78% (adapted to user patterns)
Week 8: Precision 87% (highly optimized)
```

### 4. Handles Edge Cases
- GEPA discovers rare patterns in traces
- Evolves rules for corner cases
- Robust to distribution shift

### 5. Explainable Evolution
- GEPA's reflections are in natural language
- We can see WHY it made changes
- We can audit/review mutations

## Risk Mitigation

### Safeguard 1: Gradual Deployment
```python
# Deploy to 10% of queries first
if random.random() < 0.10:
    use_refined_graph()
else:
    use_rough_graph()

# Monitor quality delta, scale up if positive
```

### Safeguard 2: Rollback Capability
```python
# Keep last N refined graphs
refined_graphs = [g1, g2, g3, ...]  # sorted by recency

if quality_drops():
    rollback_to_previous_graph(refined_graphs[-2])
```

### Safeguard 3: Human Review
```python
# Log GEPA mutations for review
for mutation in proposed_mutations:
    logger.info(f"GEPA proposes: {mutation.description}")
    logger.info(f"Rationale: {mutation.reflection}")
    logger.info(f"Test score: {mutation.test_results}")
```

### Safeguard 4: Bounded Evolution
```python
# Prevent extreme mutations
mutation_constraints = {
    "max_priority_delta": 20,  # Can't change priority by more than 20
    "max_latency_increase": 0.5,  # Can't add more than 0.5ms
    "min_precision": 0.80,  # Must maintain 80% precision
}
```

## Success Criteria

### Must-Have (Phase 1-2)
- [x] Dual graph storage working
- [ ] GEPA integration functional
- [ ] First refinement cycle completes
- [ ] Quality improves on test set

### Should-Have (Phase 3)
- [ ] Refinement cycle runs every 6 hours
- [ ] Quality improvement visible in metrics
- [ ] No user-facing latency increase
- [ ] Explainable mutations logged

### Nice-to-Have (Phase 4)
- [ ] Dashboard showing evolution over time
- [ ] A/B testing refined vs rough graph
- [ ] Multi-objective optimization (speed vs quality vs size)
- [ ] Transfer learning (one user's patterns help others)

## Conclusion

By combining fast real-time extraction with GEPA-powered offline refinement, we achieve:
- ✅ **Latency edge**: <2ms extraction unchanged
- ✅ **Quality edge**: SOTA refinement learns from traces
- ✅ **Self-improving**: Gets better with every conversation
- ✅ **Zero maintenance**: GEPA handles optimization
- ✅ **Explainable**: Natural language reflections

This is the path to winning the battle: **Fast + Smart + Self-Improving**.