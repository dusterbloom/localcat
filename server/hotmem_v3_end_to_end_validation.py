"""
HotMem v3 End-to-End Validation System
Comprehensive testing and validation of the complete HotMem v3 system

This system provides:
1. Integration testing across all components
2. Performance benchmarking
3. Accuracy validation
4. Real-world scenario testing
5. Continuous monitoring and health checks
"""

import json
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import statistics

# Import HotMem v3 components
try:
    from hotmem_v3_streaming_extraction import StreamingExtractor, StreamingChunk
    from hotmem_v3_production_integration import HotMemIntegration
    from hotmem_v3_active_learning import ActiveLearningSystem
    from hotmem_v3_dual_graph_architecture import DualGraphArchitecture
except ImportError as e:
    logging.warning(f"Could not import HotMem components: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: float
    error_message: Optional[str] = None

@dataclass
class BenchmarkResult:
    """Result of performance benchmark"""
    benchmark_name: str
    metrics: Dict[str, float]
    execution_time: float
    timestamp: float
    passed_thresholds: bool

@dataclass
class TestCase:
    """Test case definition"""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    thresholds: Dict[str, float]
    category: str

class HotMemValidator:
    """Main validation system for HotMem v3"""
    
    def __init__(self, model_path: Optional[str] = None, test_data_dir: str = "./test_data"):
        self.model_path = model_path
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.integration = None
        self.active_learning = None
        self.dual_graph = None
        self.streaming_extractor = None
        
        # Test results storage
        self.validation_results: List[ValidationResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.test_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = time.time()
        self.test_count = 0
        self.passed_tests = 0
        
        # Initialize components if model path provided
        if model_path:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize HotMem v3 components"""
        
        try:
            # Initialize integration
            self.integration = HotMemIntegration(model_path=self.model_path)
            
            # Initialize active learning
            self.active_learning = ActiveLearningSystem(model_path=self.model_path)
            
            # Initialize dual graph architecture
            self.dual_graph = DualGraphArchitecture()
            
            # Initialize streaming extractor
            self.streaming_extractor = StreamingExtractor(model_path=self.model_path)
            
            logger.info("✅ All HotMem v3 components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_integration_tests(self) -> List[ValidationResult]:
        """Run integration tests across all components"""
        
        logger.info("🧪 Running integration tests...")
        
        test_cases = [
            TestCase(
                name="basic_extraction",
                description="Test basic entity and relation extraction",
                input_data={
                    "text": "Steve Jobs founded Apple in Cupertino",
                    "expected_entities": ["Steve Jobs", "Apple", "Cupertino"],
                    "expected_relations": [{"subject": "Steve Jobs", "predicate": "founded", "object": "Apple"}]
                },
                expected_output={
                    "min_entities": 2,
                    "min_relations": 1,
                    "min_confidence": 0.5
                },
                thresholds={
                    "entity_recall": 0.7,
                    "relation_recall": 0.6,
                    "confidence_threshold": 0.5
                },
                category="extraction"
            ),
            TestCase(
                name="streaming_extraction",
                description="Test real-time streaming extraction",
                input_data={
                    "chunks": [
                        "Hi, I'm Sarah",
                        "and I work at Google",
                        "in the AI department"
                    ],
                    "final_entities": ["Sarah", "Google", "AI department"],
                    "final_relations": [{"subject": "Sarah", "predicate": "works_at", "object": "Google"}]
                },
                expected_output={
                    "final_entity_count": 3,
                    "final_relation_count": 1
                },
                thresholds={
                    "streaming_accuracy": 0.6,
                    "processing_time_threshold": 1.0
                },
                category="streaming"
            ),
            TestCase(
                name="dual_graph_operation",
                description="Test dual graph architecture functionality",
                input_data={
                    "conversation": "I work at Microsoft as a software engineer",
                    "factual_knowledge": "Microsoft is a technology company"
                },
                expected_output={
                    "working_memory_entities": 2,
                    "long_term_memory_entities": 1,
                    "cross_graph_queries": True
                },
                thresholds={
                    "memory_separation": 0.8,
                    "query_response_time": 0.5
                },
                category="architecture"
            ),
            TestCase(
                name="active_learning_cycle",
                description="Test active learning feedback loop",
                input_data={
                    "extractions": [
                        {"text": "Apple is in California", "confidence": 0.9, "correct": True},
                        {"text": "Microsoft in Seattle", "confidence": 0.4, "correct": False}
                    ],
                    "corrections": [
                        {"original": "Microsoft in Seattle", "corrected": "Microsoft is in Seattle"}
                    ]
                },
                expected_output={
                    "patterns_detected": True,
                    "learning_examples_generated": True
                },
                thresholds={
                    "pattern_detection_accuracy": 0.7,
                    "correction_processing_time": 0.3
                },
                category="learning"
            )
        ]
        
        results = []
        
        for test_case in test_cases:
            result = self._run_test_case(test_case)
            results.append(result)
            self.validation_results.append(result)
            
            if result.passed:
                self.passed_tests += 1
            
            self.test_count += 1
            
            logger.info(f"  {test_case.name}: {'✅ PASS' if result.passed else '❌ FAIL'} "
                       f"(score: {result.score:.2f}, time: {result.execution_time:.3f}s)")
        
        return results
    
    def _run_test_case(self, test_case: TestCase) -> ValidationResult:
        """Run a single test case"""
        
        start_time = time.time()
        
        try:
            if test_case.category == "extraction":
                result = self._test_basic_extraction(test_case)
            elif test_case.category == "streaming":
                result = self._test_streaming_extraction(test_case)
            elif test_case.category == "architecture":
                result = self._test_dual_graph(test_case)
            elif test_case.category == "learning":
                result = self._test_active_learning(test_case)
            else:
                result = ValidationResult(
                    test_name=test_case.name,
                    passed=False,
                    score=0.0,
                    details={"error": "Unknown test category"},
                    execution_time=time.time() - start_time,
                    timestamp=time.time(),
                    error_message="Unknown test category"
                )
        
        except Exception as e:
            result = ValidationResult(
                test_name=test_case.name,
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=time.time(),
                error_message=str(e)
            )
        
        return result
    
    def _test_basic_extraction(self, test_case: TestCase) -> ValidationResult:
        """Test basic extraction functionality"""
        
        if not self.integration:
            return ValidationResult(
                test_name=test_case.name,
                passed=False,
                score=0.0,
                details={"error": "Integration not available"},
                execution_time=0.0,
                timestamp=time.time(),
                error_message="Integration not available"
            )
        
        text = test_case.input_data["text"]
        
        # Process extraction
        start_extract = time.time()
        self.integration.process_transcription(text, is_final=True)
        extract_time = time.time() - start_extract
        
        # Get results
        graph = self.integration.get_knowledge_graph()
        
        # Calculate metrics
        expected_entities = set(test_case.input_data["expected_entities"])
        actual_entities = set(graph["entities"])
        
        entity_recall = len(expected_entities.intersection(actual_entities)) / len(expected_entities) if expected_entities else 0
        
        expected_relations = test_case.input_data["expected_relations"]
        actual_relations = graph["relations"]
        
        # Simple relation matching (could be more sophisticated)
        relation_matches = 0
        for expected_rel in expected_relations:
            for actual_rel in actual_relations:
                if (actual_rel.get("subject") == expected_rel["subject"] and
                    actual_rel.get("object") == expected_rel["object"]):
                    relation_matches += 1
                    break
        
        relation_recall = relation_matches / len(expected_relations) if expected_relations else 0
        
        # Check thresholds
        entity_threshold = test_case.thresholds.get("entity_recall", 0.7)
        relation_threshold = test_case.thresholds.get("relation_recall", 0.6)
        confidence_threshold = test_case.thresholds.get("confidence_threshold", 0.5)
        
        passed = (entity_recall >= entity_threshold and 
                 relation_recall >= relation_threshold)
        
        score = (entity_recall + relation_recall) / 2
        
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            score=score,
            details={
                "entity_recall": entity_recall,
                "relation_recall": relation_recall,
                "expected_entities": list(expected_entities),
                "actual_entities": list(actual_entities),
                "expected_relations": expected_relations,
                "actual_relations": actual_relations,
                "extraction_time": extract_time
            },
            execution_time=extract_time,
            timestamp=time.time()
        )
    
    def _test_streaming_extraction(self, test_case: TestCase) -> ValidationResult:
        """Test streaming extraction functionality"""
        
        if not self.streaming_extractor:
            return ValidationResult(
                test_name=test_case.name,
                passed=False,
                score=0.0,
                details={"error": "Streaming extractor not available"},
                execution_time=0.0,
                timestamp=time.time(),
                error_message="Streaming extractor not available"
            )
        
        chunks = test_case.input_data["chunks"]
        expected_entities = set(test_case.input_data["final_entities"])
        expected_relations = test_case.input_data["final_relations"]
        
        # Process chunks
        start_time = time.time()
        
        for i, chunk_text in enumerate(chunks):
            chunk = StreamingChunk(
                text=chunk_text,
                timestamp=time.time(),
                chunk_id=i,
                is_final=(i == len(chunks) - 1)
            )
            
            result = self.streaming_extractor.process_chunk(chunk)
        
        total_time = time.time() - start_time
        
        # Get final graph
        final_graph = self.streaming_extractor.get_current_graph()
        
        # Calculate metrics
        actual_entities = set(final_graph["entities"])
        entity_recall = len(expected_entities.intersection(actual_entities)) / len(expected_entities) if expected_entities else 0
        
        actual_relations = final_graph["relations"]
        relation_matches = 0
        for expected_rel in expected_relations:
            for actual_rel in actual_relations:
                if (actual_rel.get("subject") == expected_rel["subject"] and
                    actual_rel.get("object") == expected_rel["object"]):
                    relation_matches += 1
                    break
        
        relation_recall = relation_matches / len(expected_relations) if expected_relations else 0
        
        # Check thresholds
        processing_threshold = test_case.thresholds.get("processing_time_threshold", 1.0)
        accuracy_threshold = test_case.thresholds.get("streaming_accuracy", 0.6)
        
        passed = (total_time <= processing_threshold and 
                 (entity_recall + relation_recall) / 2 >= accuracy_threshold)
        
        score = min(1.0, (entity_recall + relation_recall) / 2)
        
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            score=score,
            details={
                "entity_recall": entity_recall,
                "relation_recall": relation_recall,
                "total_processing_time": total_time,
                "avg_chunk_time": total_time / len(chunks),
                "expected_entities": list(expected_entities),
                "actual_entities": list(actual_entities)
            },
            execution_time=total_time,
            timestamp=time.time()
        )
    
    def _test_dual_graph(self, test_case: TestCase) -> ValidationResult:
        """Test dual graph architecture"""
        
        if not self.dual_graph:
            return ValidationResult(
                test_name=test_case.name,
                passed=False,
                score=0.0,
                details={"error": "Dual graph not available"},
                execution_time=0.0,
                timestamp=time.time(),
                error_message="Dual graph not available"
            )
        
        conversation = test_case.input_data["conversation"]
        factual_knowledge = test_case.input_data["factual_knowledge"]
        
        # Add conversation to working memory
        self.dual_graph.add_extraction(
            text=conversation,
            entities=[],  # Would extract from text in real scenario
            relations=[],
            confidence=0.8,
            extraction_type="conversation"
        )
        
        # Add factual knowledge to long-term memory
        self.dual_graph.add_extraction(
            text=factual_knowledge,
            entities=[],
            relations=[],
            confidence=0.95,
            extraction_type="factual"
        )
        
        # Test queries
        start_query = time.time()
        query_results = self.dual_graph.query_knowledge("Microsoft", "entities")
        query_time = time.time() - start_query
        
        # Check separation
        stats = self.dual_graph.get_system_stats()
        working_entities = stats["working_memory"]["entity_count"]
        long_term_entities = stats["long_term_memory"]["entity_count"]
        
        # Check thresholds
        query_threshold = test_case.thresholds.get("query_response_time", 0.5)
        memory_threshold = test_case.thresholds.get("memory_separation", 0.8)
        
        passed = (query_time <= query_threshold and 
                 working_entities > 0 and long_term_entities > 0)
        
        score = min(1.0, 1.0 - (query_time / query_threshold))
        
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            score=score,
            details={
                "query_time": query_time,
                "working_memory_entities": working_entities,
                "long_term_memory_entities": long_term_entities,
                "query_results": query_results
            },
            execution_time=query_time,
            timestamp=time.time()
        )
    
    def _test_active_learning(self, test_case: TestCase) -> ValidationResult:
        """Test active learning functionality"""
        
        if not self.active_learning:
            return ValidationResult(
                test_name=test_case.name,
                passed=False,
                score=0.0,
                details={"error": "Active learning not available"},
                execution_time=0.0,
                timestamp=time.time(),
                error_message="Active learning not available"
            )
        
        extractions = test_case.input_data["extractions"]
        corrections = test_case.input_data["corrections"]
        
        # Add extractions
        for extraction in extractions:
            self.active_learning.add_extraction_result(
                text=extraction["text"],
                extraction={"entities": [], "relations": []},
                confidence=extraction["confidence"],
                is_correct=extraction["correct"]
            )
        
        # Add corrections
        start_correction = time.time()
        for correction in corrections:
            self.active_learning.add_user_correction(
                original_text=correction["original"],
                original_extraction={"entities": [], "relations": []},
                corrected_extraction={"entities": [], "relations": []},
                confidence=0.5,
                error_type="user_correction"
            )
        correction_time = time.time() - start_correction
        
        # Check for patterns
        summary = self.active_learning.get_learning_summary()
        
        # Check thresholds
        time_threshold = test_case.thresholds.get("correction_processing_time", 0.3)
        pattern_threshold = test_case.thresholds.get("pattern_detection_accuracy", 0.7)
        
        passed = (correction_time <= time_threshold and 
                 summary["significant_patterns"] >= 0)
        
        score = min(1.0, 1.0 - (correction_time / time_threshold))
        
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            score=score,
            details={
                "correction_processing_time": correction_time,
                "significant_patterns": summary["significant_patterns"],
                "total_corrections": summary["total_corrections"],
                "learning_summary": summary
            },
            execution_time=correction_time,
            timestamp=time.time()
        )
    
    def run_performance_benchmarks(self) -> List[BenchmarkResult]:
        """Run performance benchmarks"""
        
        logger.info("⚡ Running performance benchmarks...")
        
        benchmarks = []
        
        # Extraction speed benchmark
        extraction_result = self._benchmark_extraction_speed()
        benchmarks.append(extraction_result)
        
        # Memory usage benchmark
        memory_result = self._benchmark_memory_usage()
        benchmarks.append(memory_result)
        
        # Streaming latency benchmark
        streaming_result = self._benchmark_streaming_latency()
        benchmarks.append(streaming_result)
        
        # Query response benchmark
        query_result = self._benchmark_query_response()
        benchmarks.append(query_result)
        
        self.benchmark_results.extend(benchmarks)
        
        return benchmarks
    
    def _benchmark_extraction_speed(self) -> BenchmarkResult:
        """Benchmark extraction speed"""
        
        test_texts = [
            "Steve Jobs founded Apple in Cupertino",
            "Microsoft is a technology company led by Satya Nadella",
            "Google develops artificial intelligence and machine learning systems",
            "Tesla manufactures electric vehicles and clean energy solutions",
            "Amazon provides e-commerce and cloud computing services"
        ] * 20  # 100 texts total
        
        if not self.integration:
            return BenchmarkResult(
                benchmark_name="extraction_speed",
                metrics={"error": "Integration not available"},
                execution_time=0.0,
                timestamp=time.time(),
                passed_thresholds=False
            )
        
        start_time = time.time()
        
        for text in test_texts:
            self.integration.process_transcription(text, is_final=True)
        
        total_time = time.time() - start_time
        
        metrics = {
            "total_texts": len(test_texts),
            "total_time": total_time,
            "texts_per_second": len(test_texts) / total_time,
            "avg_time_per_text": total_time / len(test_texts)
        }
        
        thresholds = {
            "min_texts_per_second": 10.0,
            "max_avg_time_per_text": 0.1
        }
        
        passed = (metrics["texts_per_second"] >= thresholds["min_texts_per_second"] and
                 metrics["avg_time_per_text"] <= thresholds["max_avg_time_per_text"])
        
        return BenchmarkResult(
            benchmark_name="extraction_speed",
            metrics=metrics,
            execution_time=total_time,
            timestamp=time.time(),
            passed_thresholds=passed
        )
    
    def _benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load some data
        if self.dual_graph:
            for i in range(100):
                self.dual_graph.add_extraction(
                    text=f"Test text {i} about entity {i}",
                    entities=[f"Entity{i}"],
                    relations=[{"subject": f"Entity{i}", "predicate": "related_to", "object": "Test"}],
                    confidence=0.8,
                    extraction_type="conversation"
                )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        metrics = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_per_entity_mb": memory_increase / 100 if self.dual_graph else 0
        }
        
        thresholds = {
            "max_memory_increase_mb": 100.0,
            "max_memory_per_entity_mb": 1.0
        }
        
        passed = (metrics["memory_increase_mb"] <= thresholds["max_memory_increase_mb"] and
                 metrics["memory_per_entity_mb"] <= thresholds["max_memory_per_entity_mb"])
        
        return BenchmarkResult(
            benchmark_name="memory_usage",
            metrics=metrics,
            execution_time=0.0,
            timestamp=time.time(),
            passed_thresholds=passed
        )
    
    def _benchmark_streaming_latency(self) -> BenchmarkResult:
        """Benchmark streaming latency"""
        
        if not self.streaming_extractor:
            return BenchmarkResult(
                benchmark_name="streaming_latency",
                metrics={"error": "Streaming extractor not available"},
                execution_time=0.0,
                timestamp=time.time(),
                passed_thresholds=False
            )
        
        test_chunks = [
            "This is the first chunk",
            "and this is the second chunk",
            "finally this is the third chunk"
        ] * 10
        
        latencies = []
        
        for chunk_text in test_chunks:
            start_time = time.time()
            
            chunk = StreamingChunk(
                text=chunk_text,
                timestamp=time.time(),
                chunk_id=len(latencies),
                is_final=False
            )
            
            result = self.streaming_extractor.process_chunk(chunk)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        metrics = {
            "total_chunks": len(test_chunks),
            "avg_latency_ms": statistics.mean(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) > 20 else max(latencies) * 1000
        }
        
        thresholds = {
            "max_avg_latency_ms": 100.0,
            "max_p95_latency_ms": 200.0
        }
        
        passed = (metrics["avg_latency_ms"] <= thresholds["max_avg_latency_ms"] and
                 metrics["p95_latency_ms"] <= thresholds["max_p95_latency_ms"])
        
        return BenchmarkResult(
            benchmark_name="streaming_latency",
            metrics=metrics,
            execution_time=sum(latencies),
            timestamp=time.time(),
            passed_thresholds=passed
        )
    
    def _benchmark_query_response(self) -> BenchmarkResult:
        """Benchmark query response time"""
        
        if not self.dual_graph:
            return BenchmarkResult(
                benchmark_name="query_response",
                metrics={"error": "Dual graph not available"},
                execution_time=0.0,
                timestamp=time.time(),
                passed_thresholds=False
            )
        
        # Add some test data
        for i in range(50):
            self.dual_graph.add_extraction(
                text=f"Entity{i} relates to Entity{i+1}",
                entities=[f"Entity{i}", f"Entity{i+1}"],
                relations=[{"subject": f"Entity{i}", "predicate": "relates_to", "object": f"Entity{i+1}"}],
                confidence=0.8,
                extraction_type="conversation"
            )
        
        test_queries = ["Entity0", "Entity25", "Entity49"] * 10
        
        response_times = []
        
        for query in test_queries:
            start_time = time.time()
            results = self.dual_graph.query_knowledge(query, "entities")
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        metrics = {
            "total_queries": len(test_queries),
            "avg_response_time_ms": statistics.mean(response_times) * 1000,
            "max_response_time_ms": max(response_times) * 1000,
            "min_response_time_ms": min(response_times) * 1000,
            "p95_response_time_ms": statistics.quantiles(response_times, n=20)[18] * 1000 if len(response_times) > 20 else max(response_times) * 1000
        }
        
        thresholds = {
            "max_avg_response_time_ms": 50.0,
            "max_p95_response_time_ms": 100.0
        }
        
        passed = (metrics["avg_response_time_ms"] <= thresholds["max_avg_response_time_ms"] and
                 metrics["p95_response_time_ms"] <= thresholds["max_p95_response_time_ms"])
        
        return BenchmarkResult(
            benchmark_name="query_response",
            metrics=metrics,
            execution_time=sum(response_times),
            timestamp=time.time(),
            passed_thresholds=passed
        )
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Calculate overall statistics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        avg_score = statistics.mean([r.score for r in self.validation_results]) if self.validation_results else 0
        
        # Benchmark statistics
        total_benchmarks = len(self.benchmark_results)
        passed_benchmarks = sum(1 for b in self.benchmark_results if b.passed_thresholds)
        benchmark_pass_rate = passed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0
        
        # Category breakdown
        categories = {}
        for result in self.validation_results:
            # Extract category from test name or details
            category = "unknown"
            if "extraction" in result.test_name.lower():
                category = "extraction"
            elif "streaming" in result.test_name.lower():
                category = "streaming"
            elif "graph" in result.test_name.lower():
                category = "architecture"
            elif "learning" in result.test_name.lower():
                category = "learning"
            
            if category not in categories:
                categories[category] = {"passed": 0, "total": 0, "scores": []}
            
            categories[category]["total"] += 1
            categories[category]["passed"] += 1 if result.passed else 0
            categories[category]["scores"].append(result.score)
        
        # Calculate category averages
        for category in categories:
            scores = categories[category]["scores"]
            categories[category]["pass_rate"] = categories[category]["passed"] / categories[category]["total"]
            categories[category]["avg_score"] = statistics.mean(scores) if scores else 0
            categories[category].pop("scores")  # Remove raw scores
        
        report = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": pass_rate,
                "average_score": avg_score,
                "validation_timestamp": time.time()
            },
            "benchmark_summary": {
                "total_benchmarks": total_benchmarks,
                "passed_benchmarks": passed_benchmarks,
                "benchmark_pass_rate": benchmark_pass_rate
            },
            "category_breakdown": categories,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "details": r.details,
                    "error_message": r.error_message
                }
                for r in self.validation_results
            ],
            "benchmark_results": [
                {
                    "benchmark_name": b.benchmark_name,
                    "passed_thresholds": b.passed_thresholds,
                    "metrics": b.metrics,
                    "execution_time": b.execution_time
                }
                for b in self.benchmark_results
            ],
            "system_info": {
                "model_path": self.model_path,
                "components_initialized": {
                    "integration": self.integration is not None,
                    "active_learning": self.active_learning is not None,
                    "dual_graph": self.dual_graph is not None,
                    "streaming_extractor": self.streaming_extractor is not None
                },
                "validation_duration": time.time() - self.start_time
            }
        }
        
        return report
    
    def save_validation_report(self, filepath: str):
        """Save validation report to file"""
        
        report = self.generate_validation_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {filepath}")
        
        # Also save summary to console
        print(f"\n{'='*60}")
        print("🧪 HOT MEM V3 VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Total Tests: {report['validation_summary']['total_tests']}")
        print(f"Passed Tests: {report['validation_summary']['passed_tests']}")
        print(f"Pass Rate: {report['validation_summary']['pass_rate']:.1%}")
        print(f"Average Score: {report['validation_summary']['average_score']:.2f}")
        print(f"Benchmark Pass Rate: {report['benchmark_summary']['benchmark_pass_rate']:.1%}")
        
        print(f"\nCategory Breakdown:")
        for category, stats in report['category_breakdown'].items():
            print(f"  {category}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%}) - avg score: {stats['avg_score']:.2f}")
        
        print(f"\nValidation completed in {report['system_info']['validation_duration']:.2f} seconds")

def main():
    """Run the complete validation suite"""
    
    print("🚀 HotMem v3 End-to-End Validation")
    print("=" * 60)
    
    # Initialize validator (without model for demo)
    validator = HotMemValidator(model_path=None)
    
    # Run integration tests
    print("\n🧪 Running Integration Tests...")
    integration_results = validator.run_integration_tests()
    
    # Run performance benchmarks
    print("\n⚡ Running Performance Benchmarks...")
    benchmark_results = validator.run_performance_benchmarks()
    
    # Generate and save report
    print("\n📊 Generating Validation Report...")
    validator.save_validation_report("hotmem_v3_validation_report.json")
    
    print(f"\n✅ Validation Complete!")
    print(f"Results saved to hotmem_v3_validation_report.json")

if __name__ == "__main__":
    main()