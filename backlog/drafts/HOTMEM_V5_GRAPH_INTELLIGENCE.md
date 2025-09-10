# HotMem V5: Graph Intelligence & Context Evolution Plan

## Executive Summary
Transform LocalCat's memory system from a simple fact store into an intelligent, graph-traversing, context-aware system that leverages relationships, summaries, and dual-graph architecture for superior conversation understanding.

## Current State Analysis (2025-09-10)

### What's Working
- **GLiNER Entity Extraction**: 96.7% accuracy with compound entity detection
- **Graph Storage**: 190 edges, 126 mentions in database
- **Quality Extraction**: UD patterns + LLM assistance working well
- **Voice-Optimized**: 394ms pipeline acceptable for conversations

### Critical Gaps Identified
1. **Retrieval Failure**: 0 bullets retrieved despite correct entity extraction
2. **Unused Summaries**: Configured but not generated/retrieved
3. **No LEANN Indexes**: Semantic search capability unused
4. **No Graph Traversal**: Not leveraging relationship connections
5. **No Dual Graph**: Missing agent learning graph concept
6. **Context Assembly**: Not building optimal context layers

## Phase 1: Fix Core Retrieval (1-2 days)

### Problem
Entities are extracted correctly but retrieval returns 0 bullets despite 190 edges in database.

### Investigation Points
- Entity canonicalization mismatch between storage and retrieval
- Database path configuration issues
- Query construction bugs in `_retrieve_context()`
- Missing entity index entries

### Deliverables
- Fixed retrieval returning relevant bullets
- FTS summary retrieval working
- LEANN index generation on session end
- Comprehensive retrieval test suite

## Phase 2: Graph Traversal Intelligence (2-3 days)

### Vision
Enable the system to answer relationship queries by traversing the knowledge graph.

### Key Features

#### 2-Hop Traversal
```
Query: "How are Sarah and Tesla connected?"
Traversal: Sarah → works_at → Company X → owns → Tesla
Result: "Sarah works at Company X which owns a Tesla"
```

#### Entity Resolution via Graph
- Resolve "my car" → graph shows user owns Tesla → retrieve Tesla facts
- Handle aliases: "my wife" → graph shows married_to Sarah → retrieve Sarah facts

#### Transitive Inference
- If A manages B and B manages C, infer A's organizational relationship to C
- If X is_parent_of Y and Y is_parent_of Z, infer X is_grandparent_of Z

### Implementation
- Add `traverse_graph(start_entity, max_hops=2)` method
- Build relationship path finder
- Implement query expansion using graph neighborhoods
- Add confidence scoring based on path length

## Phase 3: Dual Graph Architecture (3-4 days)

### Concept
Split memory into two complementary graphs:

### User Facts Graph (Current)
- What the user explicitly tells us
- High confidence, authoritative
- Immutable except for corrections
- Source: User statements

### Agent Learning Graph (New)
- What the agent observes and infers
- Variable confidence scores
- Continuously updated
- Sources:
  - Query patterns (frequently asked topics)
  - Entity co-occurrence (related concepts)
  - Successful retrievals (what worked)
  - Response references (what was useful)

### Benefits
- Learn user interests over time
- Identify important entities from access patterns
- Build semantic clusters automatically
- Improve retrieval through usage learning

### Implementation
```python
class AgentLearningGraph:
    def track_query(self, query: str, entities: List[str])
    def track_retrieval(self, entity: str, was_useful: bool)
    def track_cooccurrence(self, entity1: str, entity2: str)
    def get_entity_importance(self, entity: str) -> float
    def get_related_entities(self, entity: str) -> List[str]
```

## Phase 4: Context Orchestration 2.0 (2-3 days)

### Multi-Layer Context Assembly

#### Layer 1: Direct Facts (Current)
- Facts directly about queried entities
- Highest confidence, most relevant
- Example: "Sarah is a software engineer"

#### Layer 2: Graph Neighborhood (New)
- 1-2 hop related facts
- Relationship context
- Example: "Sarah works at TechCorp, TechCorp is in San Francisco"

#### Layer 3: Session Summaries (Fix)
- Temporal context from conversation
- Recent topics and themes
- Example: "Recent discussion about Sarah's career progression"

#### Layer 4: Semantic Matches (Enable)
- LEANN vector similarity results
- Conceptually related information
- Example: "Similar discussion about professional development"

### Smart Orchestration
```python
def assemble_context(query: str, entities: List[str]) -> Context:
    budget = TokenBudget(max_tokens=2000)
    
    # Prioritize by query type
    if is_relationship_query(query):
        prioritize_graph_traversal()
    elif is_temporal_query(query):
        prioritize_summaries()
    elif is_semantic_query(query):
        prioritize_leann()
    
    # Build layers with budget management
    context = []
    context += get_direct_facts(entities, budget.allocate(0.4))
    context += get_graph_neighborhood(entities, budget.allocate(0.3))
    context += get_summaries(query, budget.allocate(0.2))
    context += get_semantic_matches(query, budget.allocate(0.1))
    
    return optimize_context(context, budget)
```

## Phase 5: Continuous Learning (1-2 days)

### Feedback Loop
Track which retrieved information actually helps:

1. **Usage Tracking**
   - Monitor which facts appear in agent responses
   - Track which retrievals lead to successful conversations
   - Identify retrieval failures

2. **Weight Adjustment**
   - Boost weights for frequently useful facts
   - Decay weights for never-referenced information
   - Learn optimal patterns per query type

3. **Automatic Optimization**
   - Prune unused edges after N days
   - Consolidate redundant information
   - Merge similar entities

### Implementation
```python
class RetrievalOptimizer:
    def track_retrieval(self, facts: List[Fact], query: str)
    def track_usage(self, fact: Fact, was_referenced: bool)
    def update_weights(self)
    def prune_unused(self, days_threshold: int)
```

## Success Metrics

### Phase 1 Success
- [ ] Retrieval returns relevant bullets for test queries
- [ ] Summaries generated and retrieved
- [ ] LEANN indexes created and searchable

### Phase 2 Success
- [ ] Graph traversal answers relationship queries
- [ ] Entity resolution via graph works
- [ ] 2-hop queries return connected facts

### Phase 3 Success
- [ ] Dual graph architecture implemented
- [ ] Agent learning from query patterns
- [ ] Entity importance scoring working

### Phase 4 Success
- [ ] Multi-layer context assembly
- [ ] Token budget optimization
- [ ] Query-appropriate prioritization

### Phase 5 Success
- [ ] Retrieval quality improves over time
- [ ] Unused information automatically pruned
- [ ] System learns from usage patterns

## Technical Considerations

### Performance Targets
- Maintain <500ms total pipeline for voice
- Graph traversal <50ms for 2 hops
- Context assembly <100ms
- Keep memory footprint reasonable

### Storage Architecture
- User graph: SQLite (current)
- Agent graph: Could use Redis/LMDB for speed
- Summaries: FTS table (current)
- LEANN: Vector index files

### Compatibility
- Maintain backward compatibility
- No breaking changes to existing API
- Gradual rollout with feature flags

## Risk Mitigation

1. **Performance Degradation**
   - Cache frequently traversed paths
   - Limit traversal depth
   - Use async where possible

2. **Memory Bloat**
   - Implement automatic pruning
   - Set retention policies
   - Monitor growth metrics

3. **Complexity Explosion**
   - Start simple, iterate
   - Feature flag each phase
   - Comprehensive testing

## Implementation Order

1. **Fix retrieval bug** (Critical, blocks everything)
2. **Enable summaries/LEANN** (Foundation for context)
3. **Add graph traversal** (Unlock relationship queries)
4. **Implement dual graph** (Learning capability)
5. **Build context orchestration** (Optimal assembly)
6. **Add continuous learning** (Self-improvement)

## Expected Outcomes

### For Users
- More relevant and complete responses
- System understands relationships
- Remembers conversation context better
- Improves over time

### For Agents
- Rich, multi-layered context
- Relationship understanding
- Temporal awareness
- Semantic connections

### For System
- Self-optimizing retrieval
- Automatic knowledge organization
- Efficient resource usage
- Scalable architecture

## Next Steps

1. Commit current GLiNER improvements
2. Fix retrieval bug in Phase 1
3. Implement phases incrementally
4. Test thoroughly at each phase
5. Monitor production metrics

---

**Created**: 2025-09-10
**Status**: Draft - Ready for implementation
**Priority**: High - Core functionality improvements