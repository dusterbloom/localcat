"""
Minimal pipeline testing framework for localcat.

This provides a simplified version focused on core functionality
without complex dependencies.
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from components.memory.hotmemory_facade import HotMemoryFacade
from components.context.context_orchestrator import pack_context
from components.context.memory_config import get_global_config
from components.session.session_store import SessionStore
from components.memory.enhanced_rule_classifier_v2 import EnhancedRuleClassifierV2

@dataclass
class SimpleMetrics:
    """Simplified metrics for pipeline testing."""
    latency_ms: float
    memory_operations: int
    tokens_processed: int
    context_size: int
    success: bool

@dataclass
class SimpleTestCase:
    """Simple test case configuration."""
    name: str
    input_text: str
    expected_intent: str

@dataclass
class SimpleTestResult:
    """Simple test result."""
    test_case: SimpleTestCase
    metrics: SimpleMetrics
    output_text: str
    memories_created: int
    error: Optional[str] = None

class MinimalPipelineTester:
    """Minimal pipeline tester for core functionality."""

    def __init__(self):
        """Initialize the minimal pipeline tester."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # Initialize core components
        self.memory_facade = HotMemoryFacade()
        self.session_store = SessionStore()
        self.intent_classifier = EnhancedRuleClassifierV2()
        self.config = get_global_config()

        # Test data storage
        self.results: List[SimpleTestResult] = []

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

    async def run_single_test(self, test_case: SimpleTestCase) -> SimpleTestResult:
        """Run a single test case through the pipeline."""
        start_time = time.time()
        memory_ops = 0
        tokens_processed = 0

        try:
            # Create session
            session_id = await self.create_test_session()

            # Step 1: Intent Classification
            intent_start = time.time()
            intent_result = await self.intent_classifier.classify_intent(
                test_case.input_text,
                session_id
            )
            intent_latency = (time.time() - intent_start) * 1000
            memory_ops += 1

            # Step 2: Memory Processing
            memory_start = time.time()
            memory_result = await self.memory_facade.process_turn(
                test_case.input_text,
                session_id,
                intent_result
            )
            memory_latency = (time.time() - memory_start) * 1000
            memory_ops += memory_result.get('operations_count', 0)

            # Count created memories
            memories_created = len(memory_result.get('created_memories', []))

            # Step 3: Context Building (simplified)
            context_start = time.time()
            memory_bullets = memory_result.get('memory_bullets', [])
            summary_text = memory_result.get('summary_text', None)

            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            packed_messages, context_stats = pack_context(
                messages=messages,
                memory_bullets=memory_bullets,
                summary_text=summary_text,
                budget_tokens=1000,
                progressive_mode=True
            )
            context_latency = (time.time() - context_start) * 1000

            # Calculate metrics
            total_latency = (time.time() - start_time) * 1000
            tokens_processed = len(test_case.input_text.split())

            # Check success
            intent_correct = intent_result.get('intent') == test_case.expected_intent
            success = intent_correct and total_latency < 5000

            metrics = SimpleMetrics(
                latency_ms=total_latency,
                memory_operations=memory_ops,
                tokens_processed=tokens_processed,
                context_size=context_stats.get('tokens_total', 0),
                success=success
            )

            # Generate simple response
            if memory_bullets:
                response = f"I understand you're saying '{test_case.input_text}'. "
                response += f"Based on our conversation, I remember that "
                response += ", ".join(memory_bullets[:2]) + "."
            else:
                response = f"I hear you saying '{test_case.input_text}'."

            return SimpleTestResult(
                test_case=test_case,
                metrics=metrics,
                output_text=response,
                memories_created=memories_created,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return SimpleTestResult(
                test_case=test_case,
                metrics=SimpleMetrics(0, 0, 0, 0, False),
                output_text="",
                memories_created=0,
                error=str(e)
            )

    async def run_test_suite(self, test_cases: List[SimpleTestCase]) -> Dict[str, Any]:
        """Run a complete test suite."""
        self.logger.info(f"Starting minimal test suite with {len(test_cases)} test cases")

        results = []
        for test_case in test_cases:
            self.logger.info(f"Running test: {test_case.name}")
            result = await self.run_single_test(test_case)
            results.append(result)

        # Calculate aggregate metrics
        successful_results = [r for r in results if r.metrics.success]
        failed_results = [r for r in results if not r.metrics.success]

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_cases),
            'passed_tests': len(successful_results),
            'failed_tests': len(failed_results),
            'success_rate': len(successful_results) / len(test_cases) if test_cases else 0,
            'average_latency': sum(r.metrics.latency_ms for r in successful_results) / len(successful_results) if successful_results else 0,
            'average_memory_ops': sum(r.metrics.memory_operations for r in successful_results) / len(successful_results) if successful_results else 0,
            'total_memories_created': sum(r.memories_created for r in results),
            'individual_results': [self.serialize_result(r) for r in results]
        }

        self.results.extend(results)
        return report

    def serialize_result(self, result: SimpleTestResult) -> Dict[str, Any]:
        """Serialize test result for storage."""
        return {
            'test_case': {
                'name': result.test_case.name,
                'input_text': result.test_case.input_text,
                'expected_intent': result.test_case.expected_intent
            },
            'metrics': asdict(result.metrics),
            'output_text': result.output_text,
            'memories_created': result.memories_created,
            'error': result.error
        }

    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save test report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Test report saved to {filepath}")

# Simple test cases
SIMPLE_TEST_CASES = [
    SimpleTestCase(
        name="Simple Fact",
        input_text="My name is John and I live in New York",
        expected_intent="FACT"
    ),
    SimpleTestCase(
        name="Question",
        input_text="What's the weather like today?",
        expected_intent="PURE_QUESTION"
    ),
    SimpleTestCase(
        name="Reaction",
        input_text="That's interesting!",
        expected_intent="REACTION"
    ),
    SimpleTestCase(
        name="Greeting",
        input_text="Hello there!",
        expected_intent="GREETING"
    ),
    SimpleTestCase(
        name="Correction",
        input_text="Actually, I meant to say something else",
        expected_intent="CORRECTION"
    )
]