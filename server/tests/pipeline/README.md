# Pipeline Testing Framework

This framework provides comprehensive testing for the localcat pipeline excluding STT/TTS components, focusing on memory management, context handling, intent processing, and overall data flow integrity.

## Overview

The testing framework is designed to evaluate the core pipeline components that enable intelligent conversation capabilities:

- **Memory System**: Fact extraction, storage, retrieval, and quality filtering
- **Context Management**: Token budgeting, progressive context, memory injection
- **Intent Processing**: Classification, quality filtering, conversation flow
- **Performance**: Latency, throughput, resource utilization

## Test Structure

```
tests/pipeline/
├── text_pipeline_tester.py          # Main testing framework
├── performance_benchmark.py         # Performance benchmarking suite
├── test_memory_system.py            # Memory system tests
├── test_context_system.py           # Context system tests
├── test_intent_processing.py       # Intent processing tests
├── README.md                        # This documentation
└── run_pipeline_tests.py            # Command-line test runner
```

## Quick Start

### Basic Testing
```bash
# Run basic pipeline tests
python run_pipeline_tests.py --mode basic

# Run with detailed output
python run_pipeline_tests.py --mode basic --verbose
```

### Performance Testing
```bash
# Run all performance benchmarks
python run_pipeline_tests.py --mode performance

# Run specific benchmarks
python run_pipeline_tests.py --mode performance --benchmarks basic_throughput concurrent_load

# Save benchmark results
python run_pipeline_tests.py --mode performance --output benchmark_results/
```

### Quality Evaluation
```bash
# Run quality evaluation tests
python run_pipeline_tests.py --mode quality
```

### Latency Testing
```bash
# Run latency optimization tests
python run_pipeline_tests.py --mode latency
```

### Comprehensive Testing
```bash
# Run all test types
python run_pipeline_tests.py --mode all --output test_results/
```

## Test Components

### 1. TextPipelineTester (`text_pipeline_tester.py`)

Core testing framework that simulates the complete pipeline:

**Key Features:**
- Intent classification accuracy testing
- Entity and relation extraction evaluation
- Memory creation and retrieval validation
- Context building and injection testing
- Quality filtering verification
- Performance metrics collection

**Metrics Tracked:**
- Latency (ms)
- Memory operations count
- Tokens processed
- Context size
- Retrieval accuracy
- Extraction quality
- Intent accuracy
- Memory efficiency

### 2. PerformanceBenchmark (`performance_benchmark.py`)

Comprehensive performance testing suite:

**Benchmark Types:**
- `basic_throughput`: Single-threaded throughput testing
- `concurrent_load`: Multi-threaded load testing
- `stress_test`: High-concurrency stress testing
- `sustained_load`: Long-duration load testing

**Performance Metrics:**
- Average, P50, P95, P99 latency
- Minimum/maximum latency
- Throughput (requests/second)
- Memory usage (MB)
- CPU usage (%)
- Error rate

### 3. Memory System Tests (`test_memory_system.py`)

Comprehensive memory system testing:

**Test Areas:**
- Memory creation and persistence
- Retrieval accuracy and relevance
- Quality filtering effectiveness
- Concurrent memory operations
- Relationship extraction
- Session isolation
- Memory correction and updates
- Search performance

### 4. Context System Tests (`test_context_system.py`)

Context management and building tests:

**Test Areas:**
- Context building with token budgeting
- Progressive context mode
- Memory injection and prioritization
- Session isolation
- Summary integration
- Performance optimization
- Configuration options

### 5. Intent Processing Tests (`test_intent_processing.py`)

Intent classification and processing tests:

**Test Areas:**
- Intent classification accuracy
- Confidence scoring
- Context-aware intent recognition
- Quality filtering
- Memory gating based on intent
- Error handling
- Performance optimization
- Conversation flow analysis

## Test Cases

### Standard Test Cases

The framework includes standard test cases covering common scenarios:

```python
STANDARD_TEST_CASES = [
    TestCase(
        name="Simple Fact",
        input_text="My name is John and I live in New York",
        expected_intent="FACT",
        expected_entities=["John", "New York"],
        expected_relations=["lives_in"]
    ),
    TestCase(
        name="Question",
        input_text="What's the weather like today?",
        expected_intent="PURE_QUESTION",
        expected_entities=["weather", "today"],
        expected_relations=[]
    ),
    # ... more test cases
]
```

### Custom Test Cases

You can create custom test cases:

```python
custom_case = TestCase(
    name="Custom Test",
    input_text="Your test input here",
    expected_intent="FACT",
    expected_entities=["entity1", "entity2"],
    expected_relations=["relation_type"],
    context_budget=4000,
    session_id="custom_session"
)
```

## Quality Metrics

### Accuracy Metrics
- **Intent Accuracy**: Percentage of correctly classified intents
- **Extraction Quality**: F1-score for entity and relation extraction
- **Retrieval Accuracy**: Relevance score of retrieved memories
- **Memory Efficiency**: Ratio of useful memories to total memories created

### Performance Metrics
- **Latency**: End-to-end processing time
- **Throughput**: Requests processed per second
- **Resource Usage**: Memory and CPU utilization
- **Error Rate**: Percentage of failed operations

### Quality Metrics
- **Coherence**: Logical consistency of responses
- **Relevance**: Appropriateness of retrieved information
- **Completeness**: Coverage of relevant information
- **Conciseness**: Efficiency of information presentation

## Benchmark Results

### Example Benchmark Output

```json
{
  "benchmark_name": "Basic Throughput Test",
  "timestamp": "2024-01-15T10:30:00",
  "metrics": {
    "avg_latency_ms": 245.67,
    "p95_latency_ms": 512.34,
    "p99_latency_ms": 789.12,
    "throughput_rps": 4.07,
    "memory_usage_mb": 156.78,
    "error_rate": 0.02
  },
  "performance_analysis": {
    "latency_grade": "Good",
    "throughput_grade": "Fair",
    "resource_efficiency": "Good"
  },
  "recommendations": [
    "Consider optimizing the slowest pipeline components",
    "Implement caching for frequent operations"
  ]
}
```

## Integration with Development Workflow

### Pre-commit Testing
```bash
# Run quick tests before committing
python run_pipeline_tests.py --mode basic
```

### CI/CD Integration
```bash
# Run comprehensive tests in CI
python run_pipeline_tests.py --mode all --output ci_results/
```

### Performance Regression Testing
```bash
# Run performance benchmarks regularly
python run_pipeline_tests.py --mode performance --output performance_baseline/
```

## Configuration

### Environment Variables
- `TEST_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `TEST_OUTPUT_DIR`: Default output directory for test results
- `TEST_DATABASE_PATH`: Path for test databases
- `TEST_LLM_ENDPOINT`: Local LLM endpoint for testing

### Configuration Files
Test configurations can be customized via JSON files:

```json
{
  "test_settings": {
    "default_token_budget": 4000,
    "max_test_duration": 300,
    "enable_performance_tracking": true
  },
  "benchmark_settings": {
    "default_concurrency": 1,
    "default_iterations": 50,
    "warmup_iterations": 5
  }
}
```

## Best Practices

### Writing Tests
1. **Isolate Tests**: Each test should be independent
2. **Use Fixtures**: Leverage pytest fixtures for setup/teardown
3. **Assert Meaningfully**: Check both success and quality metrics
4. **Handle Errors**: Gracefully handle expected and unexpected errors

### Performance Testing
1. **Warm Up**: Always run warm-up iterations
2. **Measure Multiple Times**: Run multiple iterations for reliable metrics
3. **Monitor Resources**: Track memory and CPU usage
4. **Test Different Loads**: Vary concurrency and duration

### Quality Evaluation
1. **Define Clear Metrics**: Establish success criteria
2. **Test Edge Cases**: Include unusual inputs
3. **Validate Context**: Ensure context is built correctly
4. **Check Memory Quality**: Verify memory relevance and accuracy

## Troubleshooting

### Common Issues

**Test Failures**
- Check database initialization
- Verify component dependencies
- Review test case configurations

**Performance Issues**
- Monitor system resources
- Check for memory leaks
- Profile slow components

**Memory Issues**
- Verify database cleanup
- Check session isolation
- Monitor memory usage patterns

### Debug Mode

Run tests with verbose output:
```bash
python run_pipeline_tests.py --mode basic --verbose
```

Enable debug logging:
```bash
export TEST_LOG_LEVEL=DEBUG
python run_pipeline_tests.py --mode all
```

## Contributing

### Adding New Tests
1. Create test files following the existing structure
2. Use appropriate fixtures and test cases
3. Include comprehensive documentation
4. Update this README as needed

### Performance Optimization
1. Profile existing tests
2. Identify bottlenecks
3. Implement optimizations
4. Verify improvements with benchmarks

### Quality Improvements
1. Analyze test coverage
2. Add missing test scenarios
3. Improve assertions
4. Enhance error handling

## References

This framework is inspired by:
- [Locomo](https://locomo.dev/) - Local model evaluation framework
- [Memobase](https://github.com/memodb-io/memobase) - Memory system evaluation
- [MDPI Information Journal](https://www.mdpi.com/2078-2489/13/6/298) - Conversational AI evaluation metrics