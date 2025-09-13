# Test Organization

## Active Tests

### Unit Tests (`/unit/`)
- `test_coref_resolver.py` - Coreference resolution component tests
- `test_entity_resolver.py` - Entity resolution and normalization tests
- `test_graph_analyzer.py` - Knowledge graph analysis tests
- `test_semantic_filter.py` - Semantic filtering component tests
- `test_temporal_extractor.py` - Temporal information extraction tests
- `test_memory_processor.py` - Memory processing unit tests

### Integration Tests (`/integration/`)
- `test_glirel_extraction.py` - GLiREL entity/relation extraction integration
- `test_memory_retrieval.py` - Memory retrieval system tests
- `test_session_storage.py` - Session storage integration tests
- `test_memory_rebuild_and_retrieval.py` - Full memory pipeline tests
- `test_pipeline_integration.py` - Complete pipeline integration tests

### Performance Tests (`/performance/`)
- `test_benchmarks.py` - Performance benchmarks for key components

### Debug Scripts (`/debug/`)
Contains debugging and analysis scripts for development use:
- Various debug_*.py files for troubleshooting specific components
- demo_*.py files for demonstrations
- *_analysis.py files for system analysis

## Archived Tests

### Relik Tests (`/archive/relik/`)
Tests related to the deprecated Relik extraction system. Kept for reference but no longer maintained.

### Tier Experiments (`/archive/tier_experiments/`)
Experimental tiered extraction approaches. Contains various tier1/2/3 testing and comparison scripts.

### Old Integration (`/archive/old_integration/`)
Previous integration test attempts that have been superseded by current tests.

## Running Tests

### Run all active tests:
```bash
pytest server/tests/unit/ server/tests/integration/
```

### Run specific test category:
```bash
pytest server/tests/unit/
pytest server/tests/integration/
```

### Run with coverage:
```bash
pytest --cov=server.components server/tests/
```

## Test Guidelines

1. **Unit Tests**: Focus on individual component behavior in isolation
2. **Integration Tests**: Test component interactions and full workflows
3. **Performance Tests**: Benchmark critical paths and ensure performance requirements are met
4. **Debug Scripts**: Not part of CI/CD, used for development and troubleshooting

## Current Focus

The test suite is being migrated from Relik to GLiREL for entity/relation extraction. Key areas:
- GLiREL integration with spaCy pipeline
- Memory extraction and storage using GLiREL
- Performance optimization for production use