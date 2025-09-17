"""
Text-based pipeline testing framework for localcat.

This framework provides comprehensive testing of the core pipeline components
excluding STT/TTS, focusing on memory management, context handling, intent
processing, and overall data flow integrity.
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from components.memory.hotmemory_facade import HotMemoryFacade
from components.context.context_orchestrator import pack_context
from components.context.memory_config import get_global_config
from components.session.session_store import SessionStore
from components.memory.enhanced_rule_classifier_v2 import EnhancedRuleClassifierV2
from components.extraction.enhanced_level3_extractor import QualityExtractor
from components.ai.dspy_modules import GraphBuilder
from services.summarizer import Summarizer
from services.nlp_cache import NLPCache

@dataclass
class TestMetrics:
    """Comprehensive metrics for pipeline testing."""
    latency_ms: float
    memory_operations: int
    tokens_processed: int
    context_size: int
    retrieval_accuracy: float
    extraction_quality: float
    intent_accuracy: float
    memory_efficiency: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TestCase:
    """Test case configuration."""
    name: str
    input_text: str
    expected_intent: str
    expected_entities: List[str]
    expected_relations: List[str]
    context_budget: int = 4000
    session_id: Optional[str] = None

@dataclass
class TestResult:
    """Test execution result."""
    test_case: TestCase
    metrics: TestMetrics
    success: bool
    output_text: str
    memories_created: List[Dict[str, Any]]
    context_injected: List[str]
    error: Optional[str] = None

class TextPipelineTester:
    """Main testing class for text-based pipeline evaluation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline tester."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # Initialize pipeline components
        self.memory_facade = HotMemoryFacade()
        self.session_store = SessionStore()
        self.intent_classifier = EnhancedRuleClassifierV2()
        self.extractor = QualityExtractor()
        self.graph_extractor = GraphBuilder()
        self.summarizer = Summarizer()
        self.nlp_cache = NLPCache()
        self.config = get_global_config()

        # Test data storage
        self.results: List[TestResult] = []
        self.benchmarks: Dict[str, List[float]] = {}

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def create_test_session(self) -> str:
        """Create a new test session."""
        session_id = f"test_session_{int(time.time())}"
        await self.session_store.create_session(session_id)
        return session_id

    async def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case through the pipeline."""
        start_time = time.time()
        memory_ops = 0
        tokens_processed = 0

        try:
            # Create session if not provided
            if not test_case.session_id:
                test_case.session_id = await self.create_test_session()

            # Step 1: Intent Classification
            intent_start = time.time()
            intent_result = await self.intent_classifier.classify_intent(
                test_case.input_text,
                test_case.session_id
            )
            intent_latency = (time.time() - intent_start) * 1000
            memory_ops += 1

            # Step 2: Entity and Relation Extraction
            extraction_start = time.time()
            extraction_result = await self.extractor.extract_turn(
                test_case.input_text,
                test_case.session_id,
                intent_result
            )
            extraction_latency = (time.time() - extraction_start) * 1000
            memory_ops += len(extraction_result.get('entities', []))

            # Step 3: Memory Processing
            memory_start = time.time()
            memory_result = await self.memory_facade.process_turn(
                test_case.input_text,
                test_case.session_id,
                intent_result
            )
            memory_latency = (time.time() - memory_start) * 1000
            memory_ops += memory_result.get('operations_count', 0)

            # Step 4: Context Building
            context_start = time.time()
            # Get memory bullets from memory result
            memory_bullets = memory_result.get('memory_bullets', [])
            summary_text = memory_result.get('summary_text', None)

            # Build context using functional API
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            packed_messages, context_stats = pack_context(
                messages=messages,
                memory_bullets=memory_bullets,
                summary_text=summary_text,
                budget_tokens=test_case.context_budget,
                progressive_mode=True
            )

            context_result = {
                'context': packed_messages,
                'memory_bullets': memory_bullets,
                'token_count': context_stats.get('tokens_total', 0),
                'stats': context_stats
            }
            context_latency = (time.time() - context_start) * 1000

            # Step 5: LLM Processing (simulated)
            llm_start = time.time()
            llm_response = await self.simulate_llm_processing(
                test_case.input_text,
                context_result
            )
            llm_latency = (time.time() - llm_start) * 1000
            tokens_processed = len(test_case.input_text.split()) + len(llm_response.split())

            # Calculate metrics
            total_latency = (time.time() - start_time) * 1000

            # Calculate accuracy scores
            intent_accuracy = self.calculate_intent_accuracy(
                intent_result, test_case.expected_intent
            )

            extraction_quality = self.calculate_extraction_quality(
                extraction_result, test_case.expected_entities, test_case.expected_relations
            )

            retrieval_accuracy = self.calculate_retrieval_accuracy(
                memory_result.get('retrieved_memories', [])
            )

            memory_efficiency = self.calculate_memory_efficiency(
                memory_result, context_result
            )

            metrics = TestMetrics(
                latency_ms=total_latency,
                memory_operations=memory_ops,
                tokens_processed=tokens_processed,
                context_size=context_result.get('token_count', 0),
                retrieval_accuracy=retrieval_accuracy,
                extraction_quality=extraction_quality,
                intent_accuracy=intent_accuracy,
                memory_efficiency=memory_efficiency
            )

            # Determine success
            success = (
                intent_accuracy > 0.8 and
                extraction_quality > 0.7 and
                retrieval_accuracy > 0.6 and
                total_latency < 5000  # 5 second threshold
            )

            return TestResult(
                test_case=test_case,
                metrics=metrics,
                success=success,
                output_text=llm_response,
                memories_created=memory_result.get('created_memories', []),
                context_injected=context_result.get('memory_bullets', [])
            )

        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return TestResult(
                test_case=test_case,
                metrics=TestMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                success=False,
                output_text="",
                memories_created=[],
                context_injected=[],
                error=str(e)
            )

    async def simulate_llm_processing(self, input_text: str, context: Dict[str, Any]) -> str:
        """Simulate LLM processing for testing purposes."""
        # In a real implementation, this would call the local LLM
        # For testing, we'll simulate based on context and input
        context_bullets = context.get('memory_bullets', [])

        if context_bullets:
            response = f"I understand you're saying '{input_text}'. "
            response += f"Based on our conversation, I remember that "
            response += ", ".join(context_bullets[:2]) + ". "
            response += "How can I help you further?"
        else:
            response = f"I hear you saying '{input_text}'. "
            response += "This seems to be the start of our conversation. "
            response += "Could you tell me more about what you'd like to discuss?"

        return response

    def calculate_intent_accuracy(self, actual: Dict[str, Any], expected: str) -> float:
        """Calculate intent classification accuracy."""
        actual_intent = actual.get('intent', 'unknown')
        return 1.0 if actual_intent == expected else 0.0

    def calculate_extraction_quality(
        self,
        extraction_result: Dict[str, Any],
        expected_entities: List[str],
        expected_relations: List[str]
    ) -> float:
        """Calculate extraction quality score."""
        actual_entities = [e.get('text', '') for e in extraction_result.get('entities', [])]
        actual_relations = [r.get('relation_type', '') for r in extraction_result.get('relations', [])]

        entity_score = len(set(actual_entities) & set(expected_entities)) / max(len(expected_entities), 1)
        relation_score = len(set(actual_relations) & set(expected_relations)) / max(len(expected_relations), 1)

        return (entity_score + relation_score) / 2

    def calculate_retrieval_accuracy(self, retrieved_memories: List[Dict[str, Any]]) -> float:
        """Calculate retrieval accuracy based on relevance scores."""
        if not retrieved_memories:
            return 0.0

        avg_relevance = np.mean([
            mem.get('relevance_score', 0) for mem in retrieved_memories
        ])

        return min(avg_relevance, 1.0)

    def calculate_memory_efficiency(
        self,
        memory_result: Dict[str, Any],
        context_result: Dict[str, Any]
    ) -> float:
        """Calculate memory efficiency score."""
        created_count = len(memory_result.get('created_memories', []))
        retrieved_count = len(context_result.get('memory_bullets', []))

        # Efficiency = useful memories retrieved / total memories created
        if created_count == 0:
            return 1.0

        return min(retrieved_count / created_count, 1.0)

    async def run_test_suite(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run a complete test suite."""
        self.logger.info(f"Starting test suite with {len(test_cases)} test cases")

        results = []
        for test_case in test_cases:
            self.logger.info(f"Running test: {test_case.name}")
            result = await self.run_single_test(test_case)
            results.append(result)

        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)

        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_cases),
            'passed_tests': sum(1 for r in results if r.success),
            'failed_tests': sum(1 for r in results if not r.success),
            'aggregate_metrics': aggregate_metrics,
            'individual_results': [self.serialize_result(r) for r in results]
        }

        self.results.extend(results)
        return report

    def calculate_aggregate_metrics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics from test results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                'avg_latency_ms': 0,
                'avg_memory_operations': 0,
                'avg_tokens_processed': 0,
                'avg_retrieval_accuracy': 0,
                'avg_extraction_quality': 0,
                'avg_intent_accuracy': 0,
                'avg_memory_efficiency': 0,
                'success_rate': 0
            }

        return {
            'avg_latency_ms': np.mean([r.metrics.latency_ms for r in successful_results]),
            'avg_memory_operations': np.mean([r.metrics.memory_operations for r in successful_results]),
            'avg_tokens_processed': np.mean([r.metrics.tokens_processed for r in successful_results]),
            'avg_retrieval_accuracy': np.mean([r.metrics.retrieval_accuracy for r in successful_results]),
            'avg_extraction_quality': np.mean([r.metrics.extraction_quality for r in successful_results]),
            'avg_intent_accuracy': np.mean([r.metrics.intent_accuracy for r in successful_results]),
            'avg_memory_efficiency': np.mean([r.metrics.memory_efficiency for r in successful_results]),
            'success_rate': len(successful_results) / len(results),
            'p95_latency_ms': np.percentile([r.metrics.latency_ms for r in successful_results], 95),
            'p99_latency_ms': np.percentile([r.metrics.latency_ms for r in successful_results], 99)
        }

    def serialize_result(self, result: TestResult) -> Dict[str, Any]:
        """Serialize test result for storage."""
        return {
            'test_case': {
                'name': result.test_case.name,
                'input_text': result.test_case.input_text,
                'expected_intent': result.test_case.expected_intent,
                'expected_entities': result.test_case.expected_entities,
                'expected_relations': result.test_case.expected_relations
            },
            'metrics': result.metrics.to_dict(),
            'success': result.success,
            'output_text': result.output_text,
            'memories_created': result.memories_created,
            'context_injected': result.context_injected,
            'error': result.error
        }

    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save test report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Test report saved to {filepath}")

# Standard test cases for pipeline evaluation
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
    TestCase(
        name="Reaction",
        input_text="That's interesting! Tell me more about it.",
        expected_intent="REACTION",
        expected_entities=[],
        expected_relations=[]
    ),
    TestCase(
        name="Correction",
        input_text="Actually, I meant to say I work at Google, not Microsoft",
        expected_intent="CORRECTION",
        expected_entities=["Google", "Microsoft"],
        expected_relations=["works_at"]
    ),
    TestCase(
        name="Complex Memory",
        input_text="My daughter Emma just started college at Stanford University to study computer science",
        expected_intent="FACT",
        expected_entities=["Emma", "Stanford University", "computer science"],
        expected_relations=["started_college_at", "studies"]
    )
]