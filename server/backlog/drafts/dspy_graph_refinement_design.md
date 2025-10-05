# DSPy Graph Refinement Design

## Overview

Design for background graph quality improvement using DSPy modules to refine entity extraction and relationship data during idle conversation periods.

## Problem Statement

**Current State:**
- HotPath fast extraction (27 patterns) creates entities/relationships in <200ms
- Intent classification (42.9% accuracy) determines what to process
- Removing intent classification → all text gets extracted → potential graph pollution

**Target State:**
- Remove intent classification bottleneck
- Maintain fast real-time extraction
- Use background DSPy refinement for quality improvement
- Zero maintenance operation

## Architecture Components

### Real-time Extraction Path
```
User Input → Fast Pattern Extraction → Confidence Scoring → Graph Storage → Tools
```

### Background Refinement Path
```
Graph Data → DSPy Analysis → Quality Decisions → Graph Updates
```

## DSPy Refinement Modules (Design Questions)

### Module 1: Entity Quality Analysis
**Purpose**: Identify low-value entities for removal
**Input**: Entity + context
**Output**: Keep/Remove decision with confidence
**Questions**:
- What features determine entity value?
- How to balance precision vs recall?
- Should it be domain-agnostic or domain-aware?

### Module 2: Entity Consolidation
**Purpose**: Merge similar/duplicate entities
**Input**: Entity pairs + relationship context
**Output**: Merge/Separate decision with canonical form
**Questions**:
- How to detect semantic similarity vs surface similarity?
- When to prefer one form over another?
- How to handle conflicting relationship data?

### Module 3: Relationship Validation
**Purpose**: Assess relationship meaningfulness
**Input**: (Entity1, Relation, Entity2) + context
**Output**: Quality score and Keep/Remove decision
**Questions**:
- What makes a relationship meaningful?
- How to handle temporal relationships?
- Should relationships have confidence decay?

### Module 4: Graph Coherence
**Purpose**: Ensure logical consistency across entity clusters
**Input**: Entity neighborhoods + relationship patterns
**Output**: Suggested restructuring actions
**Questions**:
- How to define coherent clusters?
- When to split vs merge clusters?
- How to handle cross-domain relationships?

## Idle Time Refinement Strategy

### Trigger Conditions
**Periodic Refinement**: Every 3600 seconds (1 hour)
**Long Idle Refinement**: After 300 seconds (5 minutes) of conversation silence

### Refinement Scope

#### Periodic Tasks (Full Graph)
- Entity consolidation across all domains
- Relationship quality assessment
- Orphaned entity cleanup
- Graph coherence analysis

#### Long-Idle Tasks (Recent Session)
- Clean entities from last N conversation turns
- Merge duplicates from current session
- Remove low-confidence recent extractions
- Optimize new relationships

## Interface Design

### Core Refiner Class
```python
class GraphRefiner:
    def __init__(self, hotpath_storage):
        # Initialize DSPy modules
        # Set timing parameters
        # Connect to graph storage

    def on_conversation_activity(self):
        # Reset idle timers

    def on_conversation_idle(self):
        # Check idle duration
        # Trigger appropriate refinement

    def run_periodic_refinement(self):
        # Full graph analysis

    def run_long_idle_refinement(self):
        # Recent session cleanup
```

### DSPy Module Interface
```python
class EntityQualityModule:
    def __init__(self):
        # Define DSPy signature
        # Load training examples

    def analyze(self, entity, context):
        # Return quality assessment

class EntityConsolidationModule:
    def __init__(self):
        # Define merge detection logic

    def should_merge(self, entity1, entity2, context):
        # Return merge decision
```

## Integration Points

### With HotPath Storage
- Read entities and relationships for analysis
- Write refinement decisions back to graph
- Maintain transaction integrity during updates

### With HotMem Tools
- Refined graph improves tool response quality
- No changes needed to tool interface
- Performance improvements transparent to users

### With Conversation Pipeline
- Idle detection hooks into conversation flow
- No interference with real-time processing
- Background operation invisible to users

## Technical Considerations

### Performance Requirements
- Refinement operations must not impact real-time performance
- Background processing should use available CPU/memory efficiently
- Gradual improvement preferred over batch operations

### Data Consistency
- Concurrent access patterns between real-time and background operations
- Transaction handling for multi-entity updates
- Rollback capability for failed refinements

### Error Handling
- Failed refinement should not corrupt existing graph
- Logging for refinement decisions and outcomes
- Graceful degradation if DSPy modules unavailable

## Quality Metrics

### Measurable Improvements
- Reduction in duplicate entities over time
- Improvement in search result relevance
- Decrease in low-confidence relationships
- Better clustering of related concepts

### Success Indicators
- Tool response quality improves with usage
- Graph size stabilizes despite continued extraction
- Search precision increases over time
- Reduced false positive retrievals

## Implementation Phases

### Phase 1: Infrastructure
- Idle time detection mechanism
- Basic refinement scheduling
- DSPy module framework
- Storage integration points

### Phase 2: Core Refinement Modules
- Entity quality assessment
- Basic duplicate detection
- Relationship validation
- Integration testing

### Phase 3: Advanced Features
- Entity consolidation with semantic analysis
- Graph coherence optimization
- Performance tuning
- Quality metrics collection

## Open Design Questions

### DSPy Training Strategy
- How to generate training examples for graph refinement?
- Should training be domain-specific or general?
- How to handle continuous learning from user behavior?

### Refinement Aggressiveness
- How conservative vs aggressive should cleanup be?
- Trade-offs between precision and recall in entity removal?
- Recovery mechanisms for over-aggressive cleanup?

### Resource Management
- CPU/memory budgets for background operations?
- Priority levels for different refinement tasks?
- Scheduling around system resource usage?

### Quality Validation
- How to validate refinement improvements?
- Metrics for measuring graph quality over time?
- User feedback integration possibilities?

## SOLID/DRY/KISS Compliance

### Single Responsibility Principle (SRP)
- Separate classes for timing, refinement execution, and DSPy modules
- Each refinement module handles one aspect of quality

### Open/Closed Principle (OCP)
- Plugin architecture for new refinement modules
- Extensible without modifying core scheduler

### Liskov Substitution Principle (LSP)
- Consistent interfaces for all refinement modules
- Swappable DSPy implementations

### Interface Segregation Principle (ISP)
- Focused interfaces for timing, refinement, and storage
- No forced dependencies on unused capabilities

### Dependency Inversion Principle (DIP)
- Abstract interfaces for storage and DSPy modules
- Concrete implementations injected at runtime

### Don't Repeat Yourself (DRY)
- Shared utilities for entity comparison and graph traversal
- Common base classes for refinement modules
- Reusable DSPy signature patterns

### Keep It Simple, Stupid (KISS)
- Start with basic duplicate detection and low-confidence removal
- Add complexity only when proven necessary
- Clear separation between real-time and background operations

## Next Steps

1. **Define DSPy module signatures** based on actual HotPath data patterns
2. **Implement basic idle detection** in conversation pipeline
3. **Create infrastructure** for scheduling and executing refinements
4. **Develop training data** for initial DSPy modules
5. **Test integration** with existing HotPath storage
6. **Measure baseline** graph quality metrics before refinement
7. **Implement Phase 1** components with thorough testing