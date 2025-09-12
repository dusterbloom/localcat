# HotMem V6: Tree Search Optimization Strategy for 90% Accuracy

## Executive Summary

LocalCat's HotMem V4 has achieved significant success with fuzzy entity matching and GLiNER integration, reaching 96.7% entity accuracy. However, relationship extraction remains at 67% accuracy, preventing the system from reaching its 90% target. This document proposes integrating Monte Carlo Tree Search (MCTS) techniques from parallel predictor generation to systematically optimize extraction patterns, retrieval weights, and context allocation.

**Current State**:
- ✅ **Fuzzy Entity Matching**: Successfully implemented, enabling rich multi-hop graph traversal
- ✅ **GLiNER Integration**: 96.7% entity extraction accuracy achieved
- ✅ **Enhanced Retrieval**: 3x more context bullets (15 vs 5) with semantic-first scoring
- ❌ **Relationship Extraction**: 67% accuracy (bottleneck for 90% goal)
- ❌ **Static Configurations**: Fixed weights and patterns miss query-specific optimizations

**Tree Search Solution**: Apply MCTS to discover optimal extraction patterns, dynamically adjust weights per query type, and evolve the system through continuous learning from failures.

## Phase 1: Extraction Pattern Discovery (Days 1-3)

### Goal
Use MCTS to automatically discover and evolve relationship extraction patterns that humans might miss, targeting 85%+ extraction accuracy.

### Implementation: `server/tree_search/extraction_optimizer.py`

```python
import asyncio
from typing import List, Dict, Tuple, Any
import math
import random
from dataclasses import dataclass

@dataclass
class ExtractionPattern:
    """Represents a UD-based extraction pattern"""
    pattern_type: str  # 'husband', 'wife', 'parent', etc.
    dependency_rules: List[Tuple[str, str, str]]  # (token, dep, head) patterns
    confidence_threshold: float
    success_rate: float = 0.0
    test_count: int = 0

class TreeNode:
    """MCTS node for extraction pattern exploration"""
    def __init__(self, pattern: ExtractionPattern = None, parent=None):
        self.pattern = pattern
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0
        self.untried_patterns = []
        
    def uct_score(self, c_puct=1.0):
        """Calculate UCT score for node selection"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.score / self.visits
        exploration = c_puct * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class ExtractionPatternEvolver:
    """Evolves extraction patterns using MCTS"""
    
    def __init__(self, test_suite_path: str):
        self.root = TreeNode()
        self.test_cases = self._load_test_cases(test_suite_path)
        self.best_patterns: Dict[str, ExtractionPattern] = {}
        self.c_puct = 1.0
        self.pattern_cache = {}
        
    def _load_test_cases(self, path: str) -> List[Dict]:
        """Load test cases from test_entity_resolution_simple.py"""
        # Parse test file for conversation examples and expected extractions
        test_cases = []
        # Example format:
        # {"text": "I'm married to Dr. Michael Chen", 
        #  "expected": [("you", "husband", "michael chen")]}
        return test_cases
        
    def generate_pattern_variation(self, base_pattern: ExtractionPattern) -> ExtractionPattern:
        """Generate variations of extraction patterns using LLM suggestions"""
        # Use LLM to suggest pattern variations based on failures
        variations = [
            # Add new dependency combinations
            self._add_dependency_rule(base_pattern),
            # Modify existing rules
            self._modify_dependency_rule(base_pattern),
            # Adjust confidence thresholds
            self._adjust_threshold(base_pattern),
            # Combine with other successful patterns
            self._hybridize_patterns(base_pattern)
        ]
        return random.choice(variations)
        
    def evaluate_pattern(self, pattern: ExtractionPattern) -> float:
        """Test pattern against test suite"""
        correct = 0
        total = len(self.test_cases)
        
        for test_case in self.test_cases:
            extracted = self._apply_pattern(pattern, test_case["text"])
            if self._matches_expected(extracted, test_case["expected"]):
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        pattern.success_rate = accuracy
        pattern.test_count += 1
        return accuracy
        
    def run_mcts(self, iterations: int = 1000, parallel_branches: int = 5):
        """Run MCTS to discover optimal patterns"""
        
        for iteration in range(iterations):
            # Selection: Choose promising node using UCT
            node = self._select_node(self.root)
            
            # Expansion: Generate new pattern variations
            if iteration % parallel_branches == 0:
                # Parallel exploration of multiple branches
                new_patterns = [
                    self.generate_pattern_variation(node.pattern)
                    for _ in range(parallel_branches)
                ]
            else:
                new_patterns = [self.generate_pattern_variation(node.pattern)]
            
            # Simulation: Evaluate patterns
            scores = [self.evaluate_pattern(p) for p in new_patterns]
            
            # Backpropagation: Update tree with results
            for pattern, score in zip(new_patterns, scores):
                child = TreeNode(pattern, parent=node)
                node.children.append(child)
                self._backpropagate(child, score)
            
            # Hybridization: Every 50 iterations
            if iteration % 50 == 0:
                self._hybridize_top_patterns()
                
            # Stagnation detection: Every 10 iterations
            if iteration % 10 == 0:
                if self._detect_stagnation():
                    self._force_exploration()
                    
            # Reflection: Every 3 iterations
            if iteration % 3 == 0:
                self._reflect_on_progress()
                
        return self.best_patterns
        
    def _hybridize_top_patterns(self):
        """Combine successful patterns to create hybrids"""
        top_patterns = sorted(
            [n.pattern for n in self._get_all_nodes() if n.pattern],
            key=lambda p: p.success_rate,
            reverse=True
        )[:3]
        
        if len(top_patterns) >= 2:
            # Create hybrid combining best aspects
            hybrid = self._create_hybrid(top_patterns[0], top_patterns[1])
            score = self.evaluate_pattern(hybrid)
            
            if score > top_patterns[0].success_rate:
                self.best_patterns[hybrid.pattern_type] = hybrid
                
    def integrate_with_hotmem(self):
        """Generate code to integrate discovered patterns into HotMem"""
        code = []
        code.append("# Auto-generated extraction patterns from MCTS")
        code.append("DISCOVERED_PATTERNS = {")
        
        for pattern_type, pattern in self.best_patterns.items():
            code.append(f"    '{pattern_type}': {{")
            code.append(f"        'rules': {pattern.dependency_rules},")
            code.append(f"        'confidence': {pattern.confidence_threshold},")
            code.append(f"        'accuracy': {pattern.success_rate}")
            code.append("    },")
            
        code.append("}")
        return "\n".join(code)
```

### Expected Improvements
- **Relationship Patterns**: Discover patterns for husband/wife, parent/child, colleague relationships
- **Complex Sentences**: Handle "I'm married to X" → extract ("you", "husband", "X")
- **Accuracy Target**: 67% → 85%+ extraction accuracy

## Phase 2: Dynamic Weight Optimization (Days 4-5)

### Goal
Discover optimal weight configurations for different query types using tree search.

### Implementation: `server/tree_search/weight_optimizer.py`

```python
from typing import Dict, List, Tuple
import numpy as np

class WeightConfiguration:
    """Represents a weight configuration for retrieval scoring"""
    def __init__(self, alpha=0.15, beta=0.35, gamma=0.45, delta=0.05):
        self.alpha = alpha  # Priority weight
        self.beta = beta    # Recency weight  
        self.gamma = gamma  # Similarity weight
        self.delta = delta  # Graph weight
        
    def normalize(self):
        """Ensure weights sum to 1"""
        total = self.alpha + self.beta + self.gamma + self.delta
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total

class QueryClassifier:
    """Classifies queries to determine optimal weight configuration"""
    
    def classify(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ["who is", "tell me about"]):
            return "identity"
        elif any(phrase in query_lower for phrase in ["where", "when", "what time"]):
            return "factual"
        elif any(phrase in query_lower for phrase in ["husband", "wife", "family", "children"]):
            return "relationship"
        elif any(phrase in query_lower for phrase in ["remember", "forget", "update"]):
            return "memory_management"
        else:
            return "general"

class WeightOptimizer:
    """Optimizes weights using tree search"""
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.optimal_weights: Dict[str, WeightConfiguration] = {}
        self.test_queries = self._load_test_queries()
        
    def _load_test_queries(self) -> List[Dict]:
        """Load test queries with expected results"""
        return [
            {
                "query": "Who is Michael Chen?",
                "type": "identity",
                "expected_keywords": ["husband", "cardiologist", "married"]
            },
            {
                "query": "Where do I work?",
                "type": "factual",
                "expected_keywords": ["google", "software", "engineer"]
            },
            {
                "query": "Tell me about my family",
                "type": "relationship",
                "expected_keywords": ["husband", "children", "emma", "liam"]
            }
        ]
        
    def generate_weight_variation(self, base: WeightConfiguration) -> WeightConfiguration:
        """Generate weight variations for exploration"""
        new_config = WeightConfiguration(
            alpha=base.alpha + np.random.normal(0, 0.05),
            beta=base.beta + np.random.normal(0, 0.05),
            gamma=base.gamma + np.random.normal(0, 0.05),
            delta=base.delta + np.random.normal(0, 0.05)
        )
        new_config.normalize()
        return new_config
        
    def evaluate_weights(self, config: WeightConfiguration, query_type: str) -> float:
        """Evaluate weight configuration for query type"""
        relevant_queries = [q for q in self.test_queries if q["type"] == query_type]
        
        total_score = 0
        for query_data in relevant_queries:
            # Simulate retrieval with these weights
            results = self._simulate_retrieval(query_data["query"], config)
            
            # Score based on presence of expected keywords
            score = self._score_results(results, query_data["expected_keywords"])
            total_score += score
            
        return total_score / len(relevant_queries) if relevant_queries else 0
        
    def optimize_for_query_type(self, query_type: str, iterations: int = 500):
        """Find optimal weights for specific query type"""
        
        # Start with current HotMem weights
        best_config = WeightConfiguration(0.15, 0.35, 0.45, 0.05)
        best_score = self.evaluate_weights(best_config, query_type)
        
        for _ in range(iterations):
            # Generate variation
            candidate = self.generate_weight_variation(best_config)
            
            # Evaluate
            score = self.evaluate_weights(candidate, query_type)
            
            # Update if better
            if score > best_score:
                best_config = candidate
                best_score = score
                
        self.optimal_weights[query_type] = best_config
        return best_config
        
    def generate_adaptive_code(self) -> str:
        """Generate code for query-adaptive weights"""
        code = []
        code.append("# Query-adaptive weight configurations from tree search")
        code.append("def get_optimal_weights(query: str) -> Tuple[float, float, float, float]:")
        code.append('    """Return optimal weights based on query type"""')
        code.append("    query_lower = query.lower()")
        
        for query_type, config in self.optimal_weights.items():
            if query_type == "identity":
                code.append('    if any(p in query_lower for p in ["who is", "tell me about"]):')
            elif query_type == "factual":
                code.append('    elif any(p in query_lower for p in ["where", "when", "what"]):')
            elif query_type == "relationship":
                code.append('    elif any(p in query_lower for p in ["husband", "wife", "family"]):')
                
            code.append(f"        return {config.alpha:.2f}, {config.beta:.2f}, {config.gamma:.2f}, {config.delta:.2f}")
            
        code.append("    else:")
        code.append("        return 0.15, 0.35, 0.45, 0.05  # Default semantic-first")
        
        return "\n".join(code)
```

### Expected Improvements
- **Identity Queries**: Higher semantic weight for "Who is X?" questions
- **Factual Queries**: Balanced weights for "Where/What/When" questions  
- **Relationship Queries**: Graph-heavy weights for family/social connections
- **Adaptive System**: Automatic weight adjustment based on query classification

## Phase 3: Context Budget Optimization (Days 6-7)

### Goal
Optimize context allocation to maximize LLM response quality.

### Implementation: `server/tree_search/context_optimizer.py`

```python
from typing import Dict, List
import json

class ContextBudgetConfiguration:
    """Represents context budget allocation"""
    def __init__(self, total_budget: int = 4096):
        self.total_budget = total_budget
        self.system_percent = 0.12
        self.memory_percent = 0.25  # Currently 25%
        self.summary_percent = 0.10
        self.dialogue_percent = 0.53
        
    def get_token_allocations(self) -> Dict[str, int]:
        """Get token allocations for each section"""
        return {
            "system": int(self.total_budget * self.system_percent),
            "memory": int(self.total_budget * self.memory_percent),
            "summary": int(self.total_budget * self.summary_percent),
            "dialogue": int(self.total_budget * self.dialogue_percent)
        }

class ContextOptimizer:
    """Optimizes context budget allocation using tree search"""
    
    def __init__(self):
        self.test_conversations = self._load_test_conversations()
        self.optimal_budgets: Dict[str, ContextBudgetConfiguration] = {}
        
    def _load_test_conversations(self) -> List[Dict]:
        """Load test conversations with quality metrics"""
        return [
            {
                "type": "memory_heavy",
                "conversation": [...],  # Multi-turn conversation
                "expected_quality": 0.9
            },
            {
                "type": "dialogue_heavy", 
                "conversation": [...],
                "expected_quality": 0.85
            }
        ]
        
    def generate_budget_variation(self, base: ContextBudgetConfiguration) -> ContextBudgetConfiguration:
        """Generate budget allocation variations"""
        import numpy as np
        
        new_config = ContextBudgetConfiguration(base.total_budget)
        
        # Vary allocations with constraints
        new_config.memory_percent = np.clip(
            base.memory_percent + np.random.normal(0, 0.05),
            0.15, 0.35  # Between 15% and 35%
        )
        new_config.summary_percent = np.clip(
            base.summary_percent + np.random.normal(0, 0.02),
            0.05, 0.15
        )
        new_config.system_percent = np.clip(
            base.system_percent + np.random.normal(0, 0.02),
            0.10, 0.15
        )
        
        # Dialogue gets the remainder
        new_config.dialogue_percent = 1.0 - (
            new_config.memory_percent + 
            new_config.summary_percent + 
            new_config.system_percent
        )
        
        return new_config
        
    def evaluate_budget(self, config: ContextBudgetConfiguration, conversation_type: str) -> float:
        """Evaluate budget configuration quality"""
        relevant_convos = [c for c in self.test_conversations if c["type"] == conversation_type]
        
        total_quality = 0
        for convo in relevant_convos:
            # Simulate context packing with this budget
            packed_context = self._pack_context(convo["conversation"], config)
            
            # Measure quality (would use LLM evaluation in practice)
            quality = self._measure_response_quality(packed_context)
            total_quality += quality
            
        return total_quality / len(relevant_convos) if relevant_convos else 0
        
    def optimize_for_conversation_type(self, conv_type: str, iterations: int = 500):
        """Find optimal budget for conversation type"""
        
        # Start with current configuration
        best_config = ContextBudgetConfiguration()
        best_config.memory_percent = 0.25  # Current setting
        best_score = self.evaluate_budget(best_config, conv_type)
        
        for _ in range(iterations):
            candidate = self.generate_budget_variation(best_config)
            score = self.evaluate_budget(candidate, conv_type)
            
            if score > best_score:
                best_config = candidate
                best_score = score
                
        self.optimal_budgets[conv_type] = best_config
        return best_config
        
    def generate_adaptive_budget_code(self) -> str:
        """Generate code for adaptive budget allocation"""
        code = []
        code.append("# Adaptive context budget allocation from tree search")
        code.append("def get_optimal_budget(conversation_metrics: Dict) -> Dict[str, float]:")
        code.append('    """Return optimal budget allocation based on conversation type"""')
        code.append("    ")
        code.append("    # Analyze conversation characteristics")
        code.append("    memory_density = conversation_metrics.get('memory_density', 0.5)")
        code.append("    turn_count = conversation_metrics.get('turn_count', 0)")
        code.append("    ")
        
        for conv_type, config in self.optimal_budgets.items():
            if conv_type == "memory_heavy":
                code.append("    if memory_density > 0.7:")
            elif conv_type == "dialogue_heavy":
                code.append("    elif turn_count > 10:")
                
            code.append("        return {")
            code.append(f'            "memory_percent": {config.memory_percent:.2f},')
            code.append(f'            "summary_percent": {config.summary_percent:.2f},')
            code.append(f'            "system_percent": {config.system_percent:.2f},')
            code.append(f'            "dialogue_percent": {config.dialogue_percent:.2f}')
            code.append("        }")
            
        code.append("    else:")
        code.append("        # Default balanced allocation")
        code.append("        return {")
        code.append('            "memory_percent": 0.25,')
        code.append('            "summary_percent": 0.10,')
        code.append('            "system_percent": 0.12,')
        code.append('            "dialogue_percent": 0.53')
        code.append("        }")
        
        return "\n".join(code)
```

### Expected Improvements
- **Memory-Heavy Conversations**: Optimize for fact-rich discussions
- **Dialogue-Heavy Conversations**: Balance recent context vs memory
- **Adaptive Allocation**: Adjust budget based on conversation characteristics
- **Quality Optimization**: Maximize LLM response quality within token limits

## Phase 4: End-to-End Pipeline Optimization (Days 8-10)

### Goal
Optimize the complete extraction → retrieval → context pipeline using tree search.

### Implementation: `server/tree_search/pipeline_evolution.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio

@dataclass
class PipelineConfiguration:
    """Complete pipeline configuration"""
    extraction_patterns: Dict[str, Any]
    retrieval_weights: Dict[str, Tuple[float, float, float, float]]
    context_budgets: Dict[str, Dict[str, float]]
    performance_metrics: Dict[str, float] = None
    
class PipelineEvolution:
    """Evolves complete pipeline configuration"""
    
    def __init__(self):
        self.extraction_optimizer = ExtractionPatternEvolver("tests/test_entity_resolution_simple.py")
        self.weight_optimizer = WeightOptimizer()
        self.context_optimizer = ContextOptimizer()
        self.best_configuration = None
        
    async def evolve_pipeline(self, iterations: int = 1000):
        """Evolve complete pipeline configuration"""
        
        for iteration in range(iterations):
            # Phase 1: Evolve extraction patterns
            if iteration < 300:
                extraction_patterns = await self._evolve_extraction()
                
            # Phase 2: Optimize weights with discovered patterns
            elif iteration < 600:
                retrieval_weights = await self._evolve_weights()
                
            # Phase 3: Optimize context with improved retrieval
            elif iteration < 900:
                context_budgets = await self._evolve_context()
                
            # Phase 4: Fine-tune complete pipeline
            else:
                configuration = PipelineConfiguration(
                    extraction_patterns=extraction_patterns,
                    retrieval_weights=retrieval_weights,
                    context_budgets=context_budgets
                )
                
                # Evaluate end-to-end
                metrics = await self._evaluate_pipeline(configuration)
                
                if self._is_better(metrics, self.best_configuration):
                    self.best_configuration = configuration
                    self.best_configuration.performance_metrics = metrics
                    
            # Periodic analysis
            if iteration % 10 == 0:
                self._analyze_progress(iteration)
                
        return self.best_configuration
        
    async def _evaluate_pipeline(self, config: PipelineConfiguration) -> Dict[str, float]:
        """Evaluate complete pipeline performance"""
        metrics = {
            "extraction_accuracy": 0.0,
            "retrieval_precision": 0.0,
            "context_quality": 0.0,
            "response_accuracy": 0.0,
            "pipeline_latency": 0.0
        }
        
        # Run comprehensive test suite
        test_results = await self._run_comprehensive_tests(config)
        
        # Calculate metrics
        metrics["extraction_accuracy"] = test_results["extraction"]["accuracy"]
        metrics["retrieval_precision"] = test_results["retrieval"]["precision"]
        metrics["context_quality"] = test_results["context"]["quality"]
        metrics["response_accuracy"] = test_results["response"]["accuracy"]
        metrics["pipeline_latency"] = test_results["performance"]["latency"]
        
        return metrics
        
    def generate_integration_code(self) -> str:
        """Generate code to integrate optimized pipeline"""
        if not self.best_configuration:
            return "# No optimized configuration available"
            
        code = []
        code.append("# Optimized pipeline configuration from tree search")
        code.append("# Performance metrics:")
        
        for metric, value in self.best_configuration.performance_metrics.items():
            code.append(f"#   {metric}: {value:.2f}")
            
        code.append("")
        code.append("OPTIMIZED_PIPELINE_CONFIG = {")
        code.append("    'extraction_patterns': {")
        
        for pattern_type, pattern in self.best_configuration.extraction_patterns.items():
            code.append(f"        '{pattern_type}': {pattern},")
            
        code.append("    },")
        code.append("    'retrieval_weights': {")
        
        for query_type, weights in self.best_configuration.retrieval_weights.items():
            code.append(f"        '{query_type}': {weights},")
            
        code.append("    },")
        code.append("    'context_budgets': {")
        
        for conv_type, budget in self.best_configuration.context_budgets.items():
            code.append(f"        '{conv_type}': {budget},")
            
        code.append("    }")
        code.append("}")
        
        return "\n".join(code)
```

## Implementation Architecture

### File Structure
```
server/
├── tree_search/
│   ├── __init__.py
│   ├── extraction_optimizer.py      # MCTS for extraction patterns
│   ├── weight_optimizer.py          # Weight configuration discovery
│   ├── context_optimizer.py         # Context budget optimization
│   ├── pipeline_evolution.py        # End-to-end optimization
│   ├── evaluation_metrics.py        # Shared evaluation framework
│   └── integration_generator.py     # Code generation for HotMem
```

### Integration Points

#### 1. memory_hotpath.py Integration
```python
# Import discovered patterns
from tree_search.extraction_optimizer import DISCOVERED_PATTERNS

# Use in extraction
def _extract(self, text: str):
    # Try discovered patterns first
    for pattern_type, pattern_config in DISCOVERED_PATTERNS.items():
        if pattern_config['accuracy'] > 0.8:
            triples = self._apply_pattern(pattern_config, text)
            if triples:
                return triples
    
    # Fallback to existing extraction
    return self._extract_with_ud(text)
```

#### 2. Adaptive Weight Integration
```python
# Import query-adaptive weights
from tree_search.weight_optimizer import get_optimal_weights

# Use in retrieval
def _retrieve_context(self, query, entities, turn_id, intent):
    # Get optimal weights for this query
    alpha, beta, gamma, delta = get_optimal_weights(query)
    
    # Apply weights in scoring
    # ... rest of retrieval logic
```

#### 3. Context Budget Integration
```python
# Import adaptive budget
from tree_search.context_optimizer import get_optimal_budget

# Use in context orchestrator
def pack_context(self, messages, budget_tokens):
    # Get optimal allocation
    metrics = self._analyze_conversation(messages)
    budget = get_optimal_budget(metrics)
    
    # Apply optimized allocation
    target_memory = int(budget_tokens * budget["memory_percent"])
    # ... rest of packing logic
```

## Testing Strategy

### Use Existing Test Infrastructure
1. **test_entity_resolution_simple.py** - Validate extraction improvements
2. **test_comprehensive_retrieval.py** - Test retrieval enhancements
3. **analyze_failures.py** - Identify remaining issues
4. **ROADMAP_TO_90_PERCENT_ACCURACY.md** - Track progress

### Performance Monitoring
```python
# Add to bot.py for real-time monitoring
class TreeSearchMonitor:
    def __init__(self):
        self.extraction_accuracy = []
        self.retrieval_precision = []
        self.response_quality = []
        
    def log_extraction(self, text, extracted, expected):
        accuracy = self._calculate_accuracy(extracted, expected)
        self.extraction_accuracy.append(accuracy)
        
    def get_metrics(self):
        return {
            "extraction": np.mean(self.extraction_accuracy),
            "retrieval": np.mean(self.retrieval_precision),
            "quality": np.mean(self.response_quality)
        }
```

## Success Metrics & Timeline

### Performance Targets
| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Extraction Accuracy | 67% | 90%+ | MCTS pattern discovery |
| Retrieval Precision | 75% | 95%+ | Adaptive weights |
| Context Quality | Good | Excellent | Optimized budgets |
| Pipeline Latency | <200ms | <200ms | Maintained |
| Relationship Queries | 0% | 80%+ | Enhanced patterns |

### Implementation Timeline

#### Week 1 (Days 1-5)
- **Days 1-3**: Extraction pattern discovery with MCTS
- **Days 4-5**: Weight optimization for query types
- **Milestone**: 85%+ extraction accuracy achieved

#### Week 2 (Days 6-10)
- **Days 6-7**: Context budget optimization
- **Days 8-10**: End-to-end pipeline evolution
- **Milestone**: 90%+ overall accuracy achieved

### Expected Outcomes

#### Immediate Benefits
1. **Rich Context**: 10+ relevant bullets per query (up from 3)
2. **Relationship Understanding**: Successfully extract husband/wife/family relationships
3. **Query-Specific Optimization**: Adaptive system that handles different query types optimally
4. **Self-Improvement**: System learns from failures and continuously improves

#### Long-Term Impact
1. **Self-Evolving System**: Continuously discovers better patterns
2. **Reduced Manual Tuning**: Tree search finds configurations humans miss
3. **Scalable Approach**: Can be applied to other optimization challenges
4. **Data-Driven Development**: Decisions based on empirical results

## Key Innovations

### 1. Hybrid Intelligence
Combines human-designed architecture (HotMem) with AI-discovered optimizations (tree search).

### 2. Continuous Learning
System improves through:
- Failure analysis and pattern evolution
- Hybridization of successful approaches
- Stagnation detection and forced exploration
- Periodic reflection and strategy adjustment

### 3. Comprehensive Optimization
Optimizes entire pipeline, not just individual components:
- Extraction → Retrieval → Context → Response
- Each phase informs the next
- End-to-end metrics guide evolution

## Risk Mitigation

### Performance Risks
- **Mitigation**: Maintain <200ms latency constraint in all optimizations
- **Fallback**: Keep original HotMem as fallback for failed optimizations

### Complexity Risks
- **Mitigation**: Modular design allows incremental adoption
- **Testing**: Comprehensive test suite validates each change

### Integration Risks
- **Mitigation**: Generate drop-in code that integrates cleanly
- **Rollback**: Version control allows easy reversion

## Conclusion

By applying tree search techniques to LocalCat's HotMem system, we can systematically discover optimal configurations that achieve the 90% accuracy target. The combination of:

1. **Your fuzzy matching breakthrough** (retrieval solved)
2. **Tree search optimization** (extraction enhanced)
3. **Continuous learning** (self-improvement)

Creates a **self-evolving memory system** that gets smarter with every conversation, achieving both immediate performance goals and long-term scalability.

The tree search approach is particularly powerful because LocalCat already has:
- Modular architecture (easy integration)
- Comprehensive tests (clear success metrics)
- Performance monitoring (real-time feedback)
- GLiNER + classifier models (strong foundation)

This creates the perfect environment for tree search optimization to thrive and deliver breakthrough results.