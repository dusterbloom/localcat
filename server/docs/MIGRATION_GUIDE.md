# HotMemory Migration Guide

## Overview

This guide explains how to migrate from the original monolithic HotMemory to the new refactored service-oriented architecture. The good news is that **most existing code requires no changes** due to the backward-compatible facade.

## Migration Levels

### Level 1: No Changes (Recommended for most users)
Simply continue using the existing HotMemoryFacade interface.

### Level 2: Gradual Migration
Use the facade but start incorporating new services where beneficial.

### Level 3: Full Migration
Use individual services directly for maximum control and performance.

## Level 1: No Changes Required

For existing code, no changes are needed:

```python
# Before (original code)
from components.memory.memory_hotpath import HotMemory
from components.memory.memory_store import MemoryStore

store = MemoryStore()
hotmem = HotMemory(store)

# After (refactored code) - IDENTICAL!
from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.memory_store import MemoryStore

store = MemoryStore()
hotmem = HotMemoryFacade(store)  # Just change class name

# All existing method calls work exactly the same
bullets, triples = hotmem.process_turn(text, session_id, turn_id)
hotmem.prewarm()
metrics = hotmem.get_metrics()
```

## Level 2: Gradual Migration

### 2.1 Use New Configuration System

```python
# Old way (hardcoded)
hotmem = HotMemory(store, max_recency=50)

# New way (centralized configuration)
from components.memory.config import create_config

config = create_config()
config.max_recency = 100  # Customize as needed
hotmem = HotMemoryFacade(store, config.max_recency)
```

### 2.2 Enable New Features

```python
# Enable new features through configuration
config.features.use_coref = True  # Enable coreference resolution
config.features.assisted_enabled = True  # Enable LLM assistance
config.features.use_leann = True  # Enable semantic search
```

### 2.3 Access Individual Services

```python
# Access specific services for advanced usage
from components.memory.hotmemory_facade import HotMemoryFacade

hotmem = HotMemoryFacade(store)

# Access individual services
extractor = hotmem.extractor
retriever = hotmem.retriever
coreference = hotmem.coreference_resolver
assisted = hotmem.assisted_extractor

# Use services directly when needed
entities = extractor.extract_entities_light("Where do I live?")
context = retriever.retrieve_context("Where do I live?", entities, 1)
```

## Level 3: Full Migration to Individual Services

### 3.1 Basic Service Setup

```python
from components.memory.config import create_config
from components.extraction.memory_extractor import MemoryExtractor
from components.retrieval.memory_retriever import MemoryRetriever
from components.coreference.coreference_resolver import CoreferenceResolver
from components.extraction.assisted_extractor import AssistedExtractor
from components.memory.memory_store import MemoryStore
from collections import defaultdict

# Setup configuration
config = create_config()

# Initialize services
store = MemoryStore()
entity_index = defaultdict(set)  # Your entity index

extractor = MemoryExtractor(config.get_extractor_config())
retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
coreference = CoreferenceResolver(config.get_coreference_config())
assisted = AssistedExtractor(config.get_assisted_config())
```

### 3.2 Complete Processing Pipeline

```python
import time
from components.memory.memory_intent import get_intent_classifier, get_quality_filter

def process_text(text: str, session_id: str, turn_id: int):
    """Complete text processing using individual services"""
    start_time = time.perf_counter()
    
    # Step 1: Intent analysis
    intent_classifier = get_intent_classifier()
    intent = intent_classifier.analyze(text, "en")
    
    # Step 2: Extract entities and relations
    extraction_result = extractor.extract(text, "en")
    entities = extraction_result.entities
    triples = extraction_result.triples
    
    # Step 3: Assisted extraction (if triggered)
    if assisted.should_assist(text, triples, extraction_result.doc):
        assisted_result = assisted.extract_assisted(
            text, entities, triples, session_id
        )
        # Merge results
        seen = set(map(tuple, triples))
        for triple in assisted_result.triples:
            if tuple(triple) not in seen:
                triples.append(tuple(triple))
                seen.add(tuple(triple))
    
    # Step 4: Coreference resolution
    coreference_result = coreference.resolve_coreferences(
        triples, extraction_result.doc, text
    )
    resolved_triples = coreference_result.resolved_triples
    
    # Step 5: Quality filtering
    quality_filter = get_quality_filter()
    stored_triples = []
    for triple in resolved_triples:
        should_store, confidence = quality_filter.should_store_fact(
            triple[0], triple[1], triple[2], intent
        )
        if should_store and confidence >= config.confidence_threshold:
            stored_triples.append(triple)
    
    # Step 6: Store in memory
    store.update_session_triples(session_id, stored_triples)
    
    # Step 7: Retrieve context
    retrieval_result = retriever.retrieve_context(text, entities, turn_id, intent)
    
    # Step 8: Return results
    processing_time = (time.perf_counter() - start_time) * 1000
    
    return {
        'bullets': retrieval_result.bullets,
        'triples': stored_triples,
        'entities': entities,
        'intent': intent,
        'processing_time_ms': processing_time,
        'metrics': {
            'extraction_time': extraction_result.processing_time_ms,
            'retrieval_time': retrieval_result.retrieval_stats.get('elapsed_ms', 0),
            'coreference_time': coreference_result.processing_time_ms,
            'assisted_time': getattr(assisted_result, 'processing_time_ms', 0)
        }
    }
```

### 3.3 Advanced Configuration

```python
# Custom configuration for specific use cases
config = create_config()

# High-performance configuration
config.max_recency = 200
config.cache_size = 2000
config.features.use_leann = True
config.features.retrieval_fusion = True

# Research configuration with all features
config.features.use_srl = True
config.features.use_coref = True
config.features.assisted_enabled = True
config.features.use_relik = True

# Minimal configuration for production
config.features.use_coref = False
config.features.assisted_enabled = False
config.confidence_threshold = 0.5
```

## Configuration Migration

### Environment Variables

```bash
# Old way (no centralized configuration)
# (various config options scattered throughout code)

# New way (centralized environment variables)
export HOTMEM_MAX_RECENCY=100
export HOTMEM_CONFIDENCE_THRESHOLD=0.3
export HOTMEM_USE_COREF=true
export HOTMEM_LLM_ASSISTED=true
export HOTMEM_USE_LEANN=true
export LEANN_INDEX_PATH="/data/memory_vectors.leann"
export HOTMEM_CACHE_SIZE=1000
```

### Configuration Files

```python
# config.py (new way)
from components.memory.config import HotMemoryConfig, FeatureFlags, ModelConfig

config = HotMemoryConfig(
    max_recency=100,
    confidence_threshold=0.3,
    features=FeatureFlags(
        use_coref=True,
        assisted_enabled=True,
        use_leann=True,
        retrieval_fusion=True
    ),
    assisted_model=ModelConfig(
        name="google/gemma-3-270m",
        base_url="http://127.0.0.1:1234/v1",
        timeout_ms=120
    )
)
```

## Performance Optimization

### 1. Prewarming

```python
# Old way (manual prewarming)
hotmem._prewarm_legacy_components("en")

# New way (automatic service prewarming)
hotmem.prewarm("en")  # Prewarms all services

# Or prewarm individual services
extractor.extract("test", "en")  # Prewarm extractor
coreference.prewarm()  # Prewarm coreference model
```

### 2. Caching

```python
# Individual service caching
extractor_metrics = extractor.get_metrics()
retriever_metrics = retriever.get_metrics()
coreference_metrics = coreference.get_metrics()
assisted_metrics = assisted.get_metrics()

# Clear caches if needed
coreference.clear_cache()
assisted.clear_cache()
```

### 3. Performance Monitoring

```python
# Detailed performance metrics
def get_detailed_metrics(hotmem):
    """Get comprehensive performance metrics"""
    return {
        'extractor': hotmem.extractor.get_metrics(),
        'retriever': hotmem.retriever.get_metrics(),
        'coreference': hotmem.coreference_resolver.get_metrics(),
        'assisted': hotmem.assisted_extractor.get_metrics(),
        'facade': hotmem.metrics
    }
```

## Error Handling

### Old vs New Error Handling

```python
# Old way (monolithic error handling)
try:
    hotmem.process_turn(text, session_id, turn_id)
except Exception as e:
    print(f"HotMemory error: {e}")

# New way (granular error handling)
try:
    bullets, triples = hotmem.process_turn(text, session_id, turn_id)
except Exception as e:
    # Get detailed error information
    extractor_metrics = hotmem.extractor.get_metrics()
    retriever_metrics = hotmem.retriever.get_metrics()
    
    print(f"Error in processing: {e}")
    print(f"Extractor calls: {extractor_metrics.get('total_calls', 0)}")
    print(f"Retriever cache size: {retriever_metrics.get('cache_size', 0)}")
```

## Testing Migration

### Unit Tests

```python
# Old way (testing monolith)
def test_hotmem_processing():
    hotmem = HotMemory(store)
    bullets, triples = hotmem.process_turn("test", "session", 1)
    assert len(bullets) > 0

# New way (testing individual services)
def test_extractor():
    extractor = MemoryExtractor(config.get_extractor_config())
    result = extractor.extract("test", "en")
    assert len(result.entities) > 0

def test_retriever():
    retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
    result = retriever.retrieve_context("test", ["you"], 1)
    assert len(result.bullets) > 0
```

### Integration Tests

```python
# Test service interactions
def test_service_integration():
    config = create_config()
    extractor = MemoryExtractor(config.get_extractor_config())
    retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
    
    # Test extraction -> retrieval pipeline
    result = extractor.extract("I live in San Francisco", "en")
    context = retriever.retrieve_context("Where do I live?", result.entities, 1)
    
    assert len(context.bullets) > 0
```

## Migration Checklist

### Pre-Migration
- [ ] Backup existing code
- [ ] Review current HotMemory usage
- [ ] Identify performance bottlenecks
- [ ] Test with current dataset

### Level 1 Migration (No Changes)
- [ ] Change import from `HotMemory` to `HotMemoryFacade`
- [ ] Test existing functionality
- [ ] Monitor performance

### Level 2 Migration (Gradual)
- [ ] Enable new configuration system
- [ ] Add new features (coreference, assisted extraction)
- [ ] Monitor performance improvements
- [ ] Gradually use individual services

### Level 3 Migration (Full)
- [ ] Replace facade with individual services
- [ ] Implement custom processing pipelines
- [ ] Add comprehensive error handling
- [ ] Performance tuning and optimization

### Post-Migration
- [ ] Performance benchmarking
- [ ] Memory usage monitoring
- [ ] Error rate tracking
- [ ] User feedback collection

## Troubleshooting

### Common Migration Issues

1. **Import Errors**
   ```python
   # Error: ImportError: No module named 'components.memory.hotmemory_facade'
   # Solution: Ensure you're using the correct import path
   from components.memory.hotmemory_facade import HotMemoryFacade
   ```

2. **Configuration Issues**
   ```python
   # Error: Missing configuration options
   # Solution: Use create_config() or set required options manually
   config = create_config()
   config.max_recency = 100
   ```

3. **Performance Issues**
   ```python
   # Issue: Slower performance after migration
   # Solution: Check cache sizes and prewarm services
   hotmem.prewarm("en")
   config.cache_size = 2000  # Increase cache size
   ```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use specific service logging
logger = logging.getLogger('components.memory')
logger.setLevel(logging.DEBUG)
```

## Best Practices

### 1. Start with Level 1
Begin with the facade to ensure compatibility, then gradually migrate.

### 2. Monitor Performance
Track metrics before and after migration to ensure improvements.

### 3. Test Thoroughly
Use the comprehensive test suite to verify functionality.

### 4. Use Configuration
Leverage the new configuration system for easier management.

### 5. Document Changes
Keep documentation updated with migration progress.

## Conclusion

The migration to the new service-oriented architecture provides significant benefits in terms of maintainability, testability, and performance. Start with Level 1 (no changes) and gradually migrate to take advantage of the new features and improvements.

For additional support, refer to the main architecture documentation or run the integration tests:
```bash
python tests/test_integration.py
```