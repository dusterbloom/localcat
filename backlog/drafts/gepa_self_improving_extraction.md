# GEPA Integration for Self-Improving Graph Extraction
*Evolving UD/SRL Patterns Through Reflective Optimization*

## Executive Summary

This document outlines the integration of GEPA (Genetic-Pareto Algorithm) with LocalCat's UD/SRL-based graph extraction system. GEPA will enable automatic pattern discovery and optimization without impacting the hot path, learning from extraction failures to continuously improve accuracy while maintaining <30ms latency.

## 🧬 Understanding GEPA

### What GEPA Really Is
GEPA is not just another optimizer - it's a **reflective text evolution framework** that:
- Uses LLMs to analyze execution traces and propose improvements
- Maintains a Pareto frontier of solutions (patterns that excel in different scenarios)
- Learns from textual feedback, not just scalar rewards
- Achieves 10-20% better performance than RL methods with 35x fewer iterations

### Why GEPA + LocalCat is Revolutionary
Your UD/SRL patterns are **text-based rules** - exactly what GEPA optimizes best:
```python
# Current: Fixed UD patterns
PATTERN_SVO = "nsubj + root + obj"

# With GEPA: Evolved, domain-specific patterns
PATTERN_SVO_EVOLVED = "nsubj[person] + root[communication_verb] + obj[entity]"
PATTERN_SVO_CONTEXT = "when confidence < 0.8, check for compound entities"
```

## 🎯 Strategic Implementation

### Architecture Overview
```
┌─────────────────────────────────────┐
│     Production Pipeline (HOT PATH)   │
│  Audio → Text → UD/SRL → Graph      │
│         ↓                            │
│    Execution Traces                  │
└─────────────────────────────────────┘
         ↓ (Async, every 24h)
┌─────────────────────────────────────┐
│        GEPA Optimization Loop        │
│   Analyze → Reflect → Mutate        │
│         ↓                            │
│    New Pattern Candidates           │
└─────────────────────────────────────┘
         ↓ (Weekly deployment)
┌─────────────────────────────────────┐
│     Updated Production Patterns      │
└─────────────────────────────────────┘
```

## 🔧 Implementation Phases

### Phase 1: GEPA Adapter for UD/SRL Patterns

#### Custom GEPAAdapter Implementation
```python
from gepa import GEPAAdapter, optimize
from typing import List, Dict, Any

class UDPatternAdapter(GEPAAdapter):
    """Adapter for optimizing UD/SRL extraction patterns"""
    
    def __init__(self, current_patterns, test_corpus):
        self.patterns = current_patterns
        self.test_corpus = test_corpus
        self.extractor = FastUDSRLExtractor()
    
    def evaluate(self, candidate: Dict[str, str], minibatch: List) -> Dict:
        """Evaluate pattern candidate on minibatch"""
        
        # Update extractor with candidate patterns
        temp_extractor = self.extractor.with_patterns(candidate)
        
        scores = []
        traces = []
        
        for example in minibatch:
            # Extract using candidate patterns
            extracted = temp_extractor.extract(example.text)
            
            # Compare with ground truth
            precision = calculate_precision(extracted, example.gold)
            recall = calculate_recall(extracted, example.gold)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
            
            scores.append(f1)
            
            # Capture execution trace
            traces.append({
                'input': example.text,
                'extracted': extracted,
                'gold': example.gold,
                'patterns_fired': temp_extractor.get_fired_patterns(),
                'confidence': temp_extractor.get_confidence(),
                'failures': identify_failures(extracted, example.gold)
            })
        
        return {
            'scores': scores,
            'traces': traces,
            'avg_score': np.mean(scores)
        }
    
    def extract_trace(self, traces: List[Dict], component: str) -> str:
        """Extract relevant trace for pattern reflection"""
        
        component_traces = []
        
        for trace in traces:
            if component in trace['patterns_fired']:
                component_traces.append({
                    'pattern': component,
                    'input': trace['input'],
                    'extracted': trace['extracted'].get(component, []),
                    'expected': trace['gold'].get(component, []),
                    'confidence': trace['confidence'],
                    'failure_type': trace['failures'].get(component, 'none')
                })
        
        # Format for LLM reflection
        return self.format_trace_for_reflection(component_traces)
    
    def format_trace_for_reflection(self, traces: List[Dict]) -> str:
        """Format traces for LLM understanding"""
        
        reflection_prompt = "Pattern performance analysis:\n\n"
        
        for t in traces[:5]:  # Limit to 5 examples
            reflection_prompt += f"""
            Pattern: {t['pattern']}
            Input: "{t['input']}"
            Extracted: {t['extracted']}
            Expected: {t['expected']}
            Confidence: {t['confidence']:.2f}
            Failure: {t['failure_type']}
            ---
            """
        
        reflection_prompt += "\nKey issues to address:"
        
        # Identify patterns in failures
        failure_types = [t['failure_type'] for t in traces]
        common_failures = Counter(failure_types).most_common(3)
        
        for failure, count in common_failures:
            reflection_prompt += f"\n- {failure}: {count} occurrences"
        
        return reflection_prompt
```

### Phase 2: Pattern Evolution Strategy

#### Domain-Specific Pattern Templates
```python
PATTERN_TEMPLATES = {
    'subject_verb_object': {
        'base': "nsubj + root + obj",
        'mutations': [
            "nsubj[animate] + root[action] + obj",
            "nsubj + root + obj[organization]",
            "compound:nsubj + root + obj"
        ]
    },
    'entity_relationships': {
        'base': "compound + flat",
        'mutations': [
            "compound[proper] + flat[name]",
            "amod + compound + flat",
            "det + compound + flat"
        ]
    },
    'temporal_expressions': {
        'base': "case[time] + obl:tmod",
        'mutations': [
            "case[in/on/at] + obl:tmod[date]",
            "advmod[when] + obl:tmod",
            "mark + advcl:tmod"
        ]
    }
}
```

#### Feedback Function for LocalCat Domain
```python
def localcat_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Generate domain-specific feedback for GEPA"""
    
    feedback = []
    
    # Check for common LocalCat-specific issues
    
    # 1. Coreference resolution failures
    if has_pronouns(gold.text) and not resolved_pronouns(pred):
        feedback.append(
            "Failed to resolve pronouns. Consider context from previous utterances."
        )
    
    # 2. Implicit relations
    if has_implicit_relations(gold) and missing_relations(pred, gold):
        feedback.append(
            "Missing implicit relations. Look for semantic connections beyond syntax."
        )
    
    # 3. Domain-specific entities
    if has_technical_terms(gold.text) and missed_entities(pred, gold):
        feedback.append(
            f"Missed domain entities: {get_missed_entities(pred, gold)}"
        )
    
    # 4. Conversational patterns
    if is_conversational(gold.text) and poor_extraction(pred):
        feedback.append(
            "Poor extraction from conversational text. Need patterns for informal speech."
        )
    
    # 5. Confidence calibration
    if pred.confidence > 0.8 and f1_score(pred, gold) < 0.5:
        feedback.append(
            "Overconfident on poor extraction. Adjust confidence thresholds."
        )
    
    # Aggregate feedback with score
    score = f1_score(pred, gold)
    
    return {
        'score': score,
        'feedback': '\n'.join(feedback),
        'failure_category': categorize_failure(pred, gold)
    }
```

### Phase 3: Optimization Pipeline

#### GEPA Configuration for LocalCat
```python
from gepa import optimize

def optimize_patterns():
    # Load current patterns
    current_patterns = load_ud_patterns()
    
    # Load training data (your conversation logs)
    train_data = load_conversation_corpus()
    val_data = load_validation_corpus()
    
    # Initialize adapter
    adapter = UDPatternAdapter(
        current_patterns=current_patterns,
        test_corpus=val_data
    )
    
    # Run GEPA optimization
    result = optimize(
        adapter=adapter,
        metric=localcat_feedback,
        trainset=train_data,
        valset=val_data,
        
        # GEPA parameters
        n_iterations=20,  # Few iterations needed
        n_candidates=10,  # Patterns to maintain
        minibatch_size=16,  # Examples per iteration
        
        # Use reflection
        reflection_model="gpt-4",
        reflection_temperature=0.7,
        
        # Pareto frontier
        use_pareto=True,  # Keep patterns good for specific cases
        
        # Budget control
        auto="heavy",  # Thorough optimization (run weekly)
        
        # Tracking
        track_stats=True,
        verbose=True
    )
    
    return result.best_patterns
```

#### Continuous Learning Loop
```python
class PatternEvolutionScheduler:
    def __init__(self):
        self.execution_buffer = []
        self.optimization_history = []
        
    async def collect_execution_traces(self, extraction_result):
        """Collect traces during normal operation"""
        self.execution_buffer.append({
            'timestamp': time.time(),
            'input': extraction_result.text,
            'output': extraction_result.entities_relations,
            'patterns_used': extraction_result.patterns,
            'confidence': extraction_result.confidence,
            'latency_ms': extraction_result.latency
        })
        
        # Trigger optimization if buffer is full
        if len(self.execution_buffer) > 10000:
            await self.trigger_optimization()
    
    async def trigger_optimization(self):
        """Run GEPA optimization (offline)"""
        
        # Prepare dataset from execution buffer
        dataset = self.prepare_dataset_from_buffer()
        
        # Run GEPA in background
        optimized_patterns = await run_async(
            optimize_patterns,
            dataset=dataset
        )
        
        # Validate improvements
        if self.validate_patterns(optimized_patterns):
            await self.deploy_patterns(optimized_patterns)
        
        # Clear buffer
        self.execution_buffer = []
    
    def validate_patterns(self, new_patterns):
        """Ensure new patterns maintain performance"""
        
        metrics = evaluate_patterns(new_patterns)
        
        return (
            metrics['latency_ms'] < 30 and
            metrics['f1_score'] > self.current_f1 and
            metrics['memory_mb'] < 200
        )
```

### Phase 4: Book Processing Pipeline

#### Batch Optimization for Books
```python
class BookGraphExtractor:
    def __init__(self, patterns):
        self.patterns = patterns
        self.batch_size = 1000  # Process 1000 sentences at once
        
    def process_book(self, book_path):
        """Extract graph from entire book in <5 minutes"""
        
        start = time.time()
        
        # Load and preprocess
        sentences = self.load_and_split(book_path)
        
        # Batch extraction
        all_graphs = []
        for batch in chunks(sentences, self.batch_size):
            graphs = self.parallel_extract(batch)
            all_graphs.extend(graphs)
        
        # Merge graphs
        final_graph = self.merge_graphs(all_graphs)
        
        # Generate test queries for GEPA
        test_queries = self.generate_test_queries(final_graph)
        
        elapsed = time.time() - start
        
        return {
            'graph': final_graph,
            'test_queries': test_queries,
            'extraction_time': elapsed,
            'sentences': len(sentences)
        }
    
    def generate_test_queries(self, graph):
        """Generate queries for GEPA evaluation"""
        
        queries = []
        
        # Entity-based queries
        for entity in graph.top_entities(10):
            queries.append(f"What do we know about {entity}?")
            queries.append(f"How is {entity} related to others?")
        
        # Relation-based queries
        for relation in graph.top_relations(5):
            queries.append(f"Find all {relation} relationships")
        
        # Path queries
        for e1, e2 in graph.random_entity_pairs(5):
            queries.append(f"How are {e1} and {e2} connected?")
        
        return queries
```

## 📊 Performance Optimization

### Pattern Compilation to ONNX
```python
class PatternCompiler:
    """Compile GEPA-optimized patterns to ONNX for speed"""
    
    def compile_to_onnx(self, patterns):
        # Convert patterns to neural network
        model = self.patterns_to_model(patterns)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            "optimized_patterns.onnx",
            opset_version=11,
            do_constant_folding=True,
            input_names=['text'],
            output_names=['entities', 'relations']
        )
        
        return "optimized_patterns.onnx"
```

### Caching Strategy
```python
PATTERN_CACHE = {
    'level1': {},  # Exact text matches (0.1ms)
    'level2': {},  # Pattern matches (1ms)
    'level3': {}   # Similar patterns (5ms)
}

def cached_extraction(text, patterns):
    # Check L1 cache
    if text in PATTERN_CACHE['level1']:
        return PATTERN_CACHE['level1'][text]
    
    # Check L2 cache
    pattern_key = hash_patterns(patterns, text)
    if pattern_key in PATTERN_CACHE['level2']:
        return PATTERN_CACHE['level2'][pattern_key]
    
    # Extract
    result = extract_with_patterns(text, patterns)
    
    # Update cache
    PATTERN_CACHE['level1'][text] = result
    PATTERN_CACHE['level2'][pattern_key] = result
    
    return result
```

## 📈 Evaluation Framework

### Metrics for Pattern Quality
```python
EVALUATION_METRICS = {
    'accuracy': {
        'f1_score': 0.9,  # Target
        'precision': 0.92,
        'recall': 0.88
    },
    'performance': {
        'latency_p50': 15,  # ms
        'latency_p99': 30,
        'throughput': 100  # sentences/sec
    },
    'robustness': {
        'noise_tolerance': 0.8,
        'domain_transfer': 0.75,
        'temporal_stability': 0.95
    }
}
```

### A/B Testing Framework
```python
class PatternABTest:
    def __init__(self, control_patterns, treatment_patterns):
        self.control = control_patterns
        self.treatment = treatment_patterns
        self.results = defaultdict(list)
    
    def run_test(self, traffic_split=0.5):
        for request in incoming_requests():
            if random.random() < traffic_split:
                result = extract_with_patterns(request, self.treatment)
                self.results['treatment'].append(result)
            else:
                result = extract_with_patterns(request, self.control)
                self.results['control'].append(result)
            
            yield result
    
    def analyze_results(self):
        treatment_f1 = mean([r.f1 for r in self.results['treatment']])
        control_f1 = mean([r.f1 for r in self.results['control']])
        
        improvement = (treatment_f1 - control_f1) / control_f1
        
        return {
            'improvement': improvement,
            'significant': self.t_test(
                self.results['treatment'],
                self.results['control']
            )
        }
```

## 🎯 Success Metrics

### Short-term (1 month)
- [ ] GEPA adapter implemented and tested
- [ ] 10% improvement in extraction F1 score
- [ ] Pattern optimization runs weekly
- [ ] No impact on hot path latency

### Medium-term (3 months)
- [ ] 20% improvement in domain-specific extraction
- [ ] Book processing in <5 minutes
- [ ] Pattern library with 100+ evolved patterns
- [ ] Automatic A/B testing of new patterns

### Long-term (6 months)
- [ ] Self-improving system requiring no manual tuning
- [ ] 95%+ extraction accuracy on domain data
- [ ] Pattern marketplace for sharing with community
- [ ] GEPA-optimized patterns outperform LLMs

## 🚨 Risk Management

### Technical Risks
1. **Pattern explosion**: Limit to top-K patterns per category
2. **Overfitting**: Use cross-validation and domain transfer tests
3. **Latency regression**: Automatic rollback if latency exceeds threshold

### Operational Risks
1. **Resource usage**: Run optimization during off-peak hours
2. **Pattern conflicts**: Maintain pattern precedence rules
3. **Version management**: Git-based pattern versioning

## 🔬 Research Opportunities

### Novel Applications
1. **Cross-lingual pattern transfer**: Learn patterns that work across languages
2. **Multimodal patterns**: Combine text + prosody patterns
3. **Temporal pattern evolution**: Patterns that adapt to language drift

### Publications
- "GEPA for Grammar Pattern Evolution in Real-time Systems"
- "Self-Improving NLP without LLMs: A Case Study"
- "Pareto-Optimal Pattern Selection for Knowledge Extraction"

## 📚 References

### GEPA Resources
- Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- GitHub: https://github.com/gepa-ai/gepa
- DSPy Integration: https://dspy.ai/api/optimizers/GEPA/

### UD/SRL Resources
- Universal Dependencies: https://universaldependencies.org/
- CoNLL-2009 Shared Task: Semantic Role Labeling
- spaCy UD Models: https://spacy.io/models

## 📅 Implementation Timeline

**Week 1**: GEPA adapter implementation
**Week 2**: Execution trace collection system
**Week 3**: First optimization cycle
**Week 4**: Pattern validation and deployment
**Month 2**: Book processing pipeline
**Month 3**: Production deployment with monitoring

---

*GEPA integration will transform LocalCat from a static rule-based system to a continuously learning extraction engine that improves with every conversation, without sacrificing the speed that makes it unique.*
