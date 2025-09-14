# HotMemory Architecture Documentation

## Overview

HotMemory is a sophisticated memory processing system that has been refactored from a monolithic 3,501-line class into a collection of focused, single-responsibility services. This refactoring improves maintainability, testability, and code organization while preserving full backward compatibility.

## Architecture Goals

1. **Single Responsibility**: Each service has one clear purpose
2. **Maintainability**: Smaller, focused services are easier to understand and modify
3. **Testability**: Services can be tested independently
4. **Performance**: Optimized with caching and lazy loading
5. **Backward Compatibility**: Existing code continues to work without changes

## Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HotMemoryFacade                           │
│                 (Backward Compatibility)                     │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
    │ MemoryStore  │ │  Config   │ │  Metrics    │
    │ (External)   │ │ Management│ │  Tracking   │
    └──────────────┘ └───────────┘ └─────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───▼────┐ ┌───────────▼───┐ ┌───▼────────┐ ┌─────▼─────┐
│Memory  │ │   Coreference  │ │  Assisted  │ │Retrieval   │
│Extractor│ │   Resolver     │ │ Extractor  │ │ Service   │
└────────┘ └────────────────┘ └────────────┘ └───────────┘
```

## Service Details

### 1. MemoryExtractor (`components/extraction/memory_extractor.py`)

**Purpose**: Dedicated service for entity and relation extraction from text.

**Key Features**:
- Multiple extraction strategies (UD, SRL, ONNX, ReLiK, DSPy)
- Intent-aware extraction
- Performance optimization with caching
- Light entity extraction for retrieval context

**Methods**:
- `extract(text, lang)` - Main extraction entry point
- `extract_entities_light(text)` - Fast entity extraction for retrieval
- `get_metrics()` - Performance metrics

**Configuration**:
- `use_srl` - Enable semantic role labeling
- `use_onnx_ner` - Enable ONNX NER
- `use_relik` - Enable ReLiK extraction
- `cache_size` - Cache size for optimization

### 2. MemoryRetriever (`components/retrieval/memory_retriever.py`)

**Purpose**: Dedicated service for context retrieval and memory search.

**Key Features**:
- MMR (Maximal Marginal Relevance) algorithm for diverse selection
- Entity expansion with aliases and relationships
- Multi-hop graph traversal
- LEANN semantic search integration
- FTS (Full-Text Search) fusion
- Type-safe tuple processing

**Methods**:
- `retrieve_context(query, entities, turn_id, intent)` - Main retrieval entry point
- `_expand_query_entities(entities, query)` - Entity expansion
- `_apply_mmr_selection(query, candidates, turn_id)` - MMR algorithm
- `_multi_hop_expansion(base_entities, query)` - Graph traversal

**Algorithms**:
- **MMR**: Balances relevance vs diversity for memory selection
- **Entity Expansion**: Finds related entities through aliases and relationships
- **Multi-hop Traversal**: Expands context through graph relationships

### 3. CoreferenceResolver (`components/coreference/coreference_resolver.py`)

**Purpose**: Dedicated service for resolving pronouns and entity references.

**Key Features**:
- Neural coreference with FCoref
- Rule-based fallback
- Pronoun resolution (I → you, my → your)
- Performance optimization with caching
- Type-safe processing

**Methods**:
- `resolve_coreferences(triples, doc, text)` - Main resolution entry point
- `_apply_neural_coreference(triples, doc)` - Neural resolution
- `_apply_rule_based_coreference(triples, doc, text)` - Rule-based fallback
- `prewarm()` - Model prewarming

**Configuration**:
- `use_coref` - Enable neural coreference
- `coref_max_entities` - Maximum entities for neural processing
- `coref_device` - Device for neural model

### 4. AssistedExtractor (`components/extraction/assisted_extractor.py`)

**Purpose**: Dedicated service for LLM-assisted relation extraction.

**Key Features**:
- Multiple extraction strategies (classifier, JSON, fallback)
- Intelligent triggering based on context
- Performance optimization with caching
- Graceful degradation when LLM unavailable

**Methods**:
- `extract_assisted(text, entities, base_triples, session_id)` - Main assisted extraction
- `should_assist(text, triples, doc)` - Trigger condition checking
- `_extract_with_classifier()` - Classifier-based extraction
- `_extract_with_json()` - JSON-based extraction
- `_extract_with_fallback()` - Fallback extraction

**Trigger Conditions**:
- Few triples extracted (≤ 2)
- High uncertainty in relations
- Complex sentences (> 20 words)
- Question marks in text

### 5. Configuration Management (`components/memory/config.py`)

**Purpose**: Centralized configuration for all services.

**Key Features**:
- Environment variable parsing
- Feature flag management
- Model configuration
- Service-specific configuration methods
- Validation and path setup

**Methods**:
- `get_extractor_config()` - MemoryExtractor configuration
- `get_retriever_config()` - MemoryRetriever configuration
- `get_coreference_config()` - CoreferenceResolver configuration
- `get_assisted_config()` - AssistedExtractor configuration

**Environment Variables**:
- `HOTMEM_USE_SRL` - Enable SRL extraction
- `HOTMEM_USE_COREF` - Enable coreference resolution
- `HOTMEM_LLM_ASSISTED` - Enable LLM assistance
- `LEANN_INDEX_PATH` - LEANN index path
- `HOTMEM_MAX_RECENCY` - Maximum recency buffer size

### 6. HotMemoryFacade (`components/memory/hotmemory_facade.py`)

**Purpose**: Backward compatibility layer using extracted services internally.

**Key Features**:
- Maintains original HotMemory interface
- Uses extracted services internally
- No breaking changes for existing code
- Performance tracking across services

**Methods**:
- `process_turn(text, session_id, turn_id)` - Main processing entry point
- `prewarm(lang)` - Prewarm all services
- `get_metrics()` - Aggregated metrics from all services

## Processing Pipeline

```
Input Text
    │
    ▼
Intent Analysis ──┐
    │            │
    ▼            │
Entity Extraction │
    │            │
    ▼            │
Relation Extraction◄┘
    │
    ▼
Assisted Extraction (if triggered)
    │
    ▼
Coreference Resolution (if enabled)
    │
    ▼
Quality Filtering & Storage
    │
    ▼
Context Retrieval (MMR)
    │
    ▼
Output Bullets + Stored Triples
```

## Key Improvements

### 1. Type Safety
- All tuple operations include type checking
- Graceful handling of malformed data
- Comprehensive error logging

### 2. Performance Optimization
- Caching for repeated operations
- Lazy loading of expensive models
- Performance guardrails for large inputs

### 3. Modular Design
- Each service has single responsibility
- Services can be tested independently
- Easy to extend or replace individual services

### 4. Backward Compatibility
- Original interface preserved
- No breaking changes for existing code
- Gradual migration possible

## Configuration Example

```python
# Environment variables
export HOTMEM_USE_COREF=true
export HOTMEM_LLM_ASSISTED=true
export HOTMEM_MAX_RECENCY=100
export LEANN_INDEX_PATH="/data/memory_vectors.leann"

# Programmatic configuration
from components.memory.config import create_config

config = create_config()
config.features.use_coref = True
config.features.assisted_enabled = True
config.max_recency = 100
```

## Usage Examples

### Basic Usage (Backward Compatible)
```python
from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.memory_store import MemoryStore

store = MemoryStore()
hotmem = HotMemoryFacade(store)

# Process a conversation turn
bullets, triples = hotmem.process_turn(
    "I live in San Francisco and work at Google",
    "session_123",
    1
)
```

### Direct Service Usage
```python
from components.extraction.memory_extractor import MemoryExtractor
from components.retrieval.memory_retriever import MemoryRetriever
from components.memory.config import create_config

config = create_config()

# Use services directly
extractor = MemoryExtractor(config.get_extractor_config())
retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())

# Extract entities and relations
result = extractor.extract("I live in San Francisco", "en")

# Retrieve context
context = retriever.retrieve_context("Where do I live?", ["you"], 1)
```

## Performance Considerations

1. **Memory Usage**: Each service maintains its own caches and indexes
2. **Latency**: Neural models (Coreference, LLM) are loaded lazily
3. **Scalability**: Services are designed to handle high-volume processing
4. **Caching**: Multiple layers of caching for optimization

## Testing

The refactored architecture includes comprehensive tests:

- **Unit Tests**: Individual service testing
- **Integration Tests**: Service interaction testing
- **Performance Tests**: Latency and memory usage testing
- **Backward Compatibility Tests**: Ensure existing functionality preserved

Run tests:
```bash
python tests/test_integration.py
```

## Migration Guide

### For Existing Code
No changes required! The HotMemoryFacade maintains complete backward compatibility.

### For New Development
Consider using services directly for more control:

```python
# Instead of facade
hotmem = HotMemoryFacade(store)

# Use services directly
config = create_config()
extractor = MemoryExtractor(config.get_extractor_config())
retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
coreference = CoreferenceResolver(config.get_coreference_config())
assisted = AssistedExtractor(config.get_assisted_config())
```

## Future Enhancements

1. **Additional Services**: Extract more functionality from the facade
2. **Advanced Algorithms**: Enhanced MMR and entity expansion
3. **Performance Monitoring**: Real-time performance metrics
4. **Configuration UI**: Web-based configuration management
5. **Service Discovery**: Dynamic service registration and discovery

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Performance Issues**: Check configuration settings and cache sizes
3. **Memory Usage**: Monitor cache sizes and adjust as needed
4. **Model Loading**: Ensure neural models are properly configured

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed logging from all services.

## Conclusion

The refactored HotMemory architecture represents a significant improvement in code organization, maintainability, and testability while preserving full backward compatibility. The service-oriented design makes it easier to understand, modify, and extend the system for future requirements.