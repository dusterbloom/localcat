# Testing Documentation

This directory contains comprehensive testing infrastructure for LocalCat pipeline components.

## Directory Structure

```
testing/
├── benchmarks/           # Performance benchmarking
│   ├── run_quick_benchmarks.py     # Comprehensive benchmark suite
│   └── benchmark_results.json      # Latest benchmark results
├── integration/         # Integration tests
│   ├── run_pipeline_tests.py       # Full pipeline integration tests
│   └── run_minimal_tests.py        # Minimal component tests
├── scripts/            # Utility scripts
└── README.md           # This file
```

## Quick Start

### Run Comprehensive Benchmarks
```bash
cd testing/benchmarks
python run_quick_benchmarks.py
```

### Run Integration Tests
```bash
cd testing/integration
python run_pipeline_tests.py
```

### Run Minimal Tests
```bash
cd testing/integration
python run_minimal_tests.py
```

## Test Coverage

### Benchmark Tests
- **Intent Classification**: <1ms latency target
- **Memory Extraction**: <200ms latency target
- **Context Building**: <50ms latency target
- **Complex Retrieval**: <100ms latency target
- **Full Pipeline**: <300ms latency target

### Integration Tests
- **API Compatibility**: HotMemoryFacade process_turn() method
- **Memory Operations**: Full CRUD operations
- **Session Management**: Session lifecycle and persistence
- **Component Integration**: End-to-end pipeline functionality

## Performance Targets

Based on .env configuration:
- **Total Budget**: 300ms (HOTMEM_TOTAL_BUDGET_MS)
- **Retrieval Timeout**: 100ms (HOTMEM_RETRIEVAL_TIMEOUT_MS)
- **Extraction Timeout**: 200ms (HOTMEM_EXTRACTION_TIMEOUT_MS)
- **Context Budget**: 4096 tokens (CONTEXT_BUDGET_TOKENS)

## Current Performance Status

**✅ Working Well:**
- Intent Classification: 0.01ms avg
- Memory Extraction: 0.14ms avg
- Context Building: 0.38ms avg

**⚠️ Needs Optimization:**
- Complex Retrieval: 1600ms+ (exceeds target)
- Full Pipeline: Limited by retrieval performance

## V7 Performance Improvement

The Enhanced Level3 (V7) extractor shows **228x performance improvement** over baseline:
- V7: 0.2ms average
- Baseline: 45.6ms average
- Quality: Equivalent relation extraction

## Configuration

Tests use the current .env configuration for realistic performance measurement. Key settings:
- HOTMEM_USE_LEANN=true
- CONTEXT_PROGRESSIVE_MODE=true
- HOTMEM_RETRIEVAL_TIMEOUT_MS=100
- CONTEXT_BUDGET_TOKENS=4096