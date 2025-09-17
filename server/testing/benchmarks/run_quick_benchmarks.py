#!/usr/bin/env python3
"""
Quick performance benchmarks for LocalCat pipeline components.
Uses current .env configuration and existing testing infrastructure.
"""

import asyncio
import time
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import json

# Add server to path
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.memory_store import MemoryStore, Paths
from components.memory.enhanced_rule_classifier_v2 import EnhancedRuleClassifierV2
from components.extraction.enhanced_level3_extractor import QualityExtractor
from components.context.context_orchestrator import pack_context
from components.context.memory_config import get_global_config

class QuickBenchmarks:
    """Quick performance benchmarks using current configuration."""

    def __init__(self):
        self.config = get_global_config()
        self.results = {}

    async def benchmark_intent_classification(self):
        """Benchmark intent classification performance."""
        logger.info("🎯 Benchmarking intent classification...")

        classifier = EnhancedRuleClassifierV2()
        test_cases = [
            "What's the weather like today?",
            "My name is John and I live in New York",
            "That's interesting!",
            "Hello there!",
            "Actually, I meant to say something else"
        ]

        times = []
        for text in test_cases:
            start = time.perf_counter()
            result = classifier.classify(text)  # Correct API method
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        self.results['intent_classification'] = {
            'avg_latency_ms': avg_time,
            'test_cases': len(test_cases),
            'target_latency_ms': 50  # From your .env HOTMEM_RETRIEVAL_TIMEOUT_MS
        }

        logger.info(f"✅ Intent classification: {avg_time:.2f}ms avg")
        return avg_time < 50

    async def benchmark_memory_extraction(self):
        """Benchmark memory extraction performance."""
        logger.info("🧠 Benchmarking memory extraction...")

        extractor = QualityExtractor()
        test_texts = [
            "My wife is at Google since 2020. She works there as a manager.",
            "John lives in Seattle and works at Microsoft. He has been there since 2018.",
            "Emma is 5 years old and attends preschool in the morning."
        ]

        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("⚠️  spaCy model not found, skipping extraction benchmark")
            self.results['memory_extraction'] = {'skipped': True, 'reason': 'spaCy model not found'}
            return True

        times = []
        relations_count = []

        for text in test_texts:
            doc = nlp(text)
            start = time.perf_counter()
            kg = extractor.extract_quality_kg(doc)
            end = time.perf_counter()
            times.append((end - start) * 1000)
            relations_count.append(len(kg.get('relations', [])))

        avg_time = sum(times) / len(times)
        avg_relations = sum(relations_count) / len(relations_count)

        self.results['memory_extraction'] = {
            'avg_latency_ms': avg_time,
            'avg_relations': avg_relations,
            'test_cases': len(test_texts),
            'target_latency_ms': 200  # From your .env HOTMEM_EXTRACTION_TIMEOUT_MS
        }

        logger.info(f"✅ Memory extraction: {avg_time:.2f}ms avg, {avg_relations:.1f} relations avg")
        return avg_time < 200

    async def benchmark_context_building(self):
        """Benchmark context building performance."""
        logger.info("🏗️  Benchmarking context building...")

        # Test different context sizes
        test_scenarios = [
            {"memory_bullets": [], "summary_text": None, "name": "empty"},
            {"memory_bullets": ["User lives in San Francisco", "User works at Google"], "summary_text": None, "name": "small"},
            {"memory_bullets": ["User lives in San Francisco", "User works at Google", "User has a dog named Max", "User enjoys hiking", "User graduated from Stanford"], "summary_text": "User discussed their background and preferences", "name": "medium"}
        ]

        times = []

        for scenario in test_scenarios:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]

            start = time.perf_counter()
            packed_messages, stats = pack_context(
                messages=messages,
                memory_bullets=scenario["memory_bullets"],
                summary_text=scenario["summary_text"],
                budget_tokens=4096,  # From your .env CONTEXT_BUDGET_TOKENS
                progressive_mode=True  # From your .env CONTEXT_PROGRESSIVE_MODE
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        self.results['context_building'] = {
            'avg_latency_ms': avg_time,
            'test_scenarios': len(test_scenarios),
            'target_latency_ms': 50  # Quick target for context building
        }

        logger.info(f"✅ Context building: {avg_time:.2f}ms avg")
        return avg_time < 50

    async def benchmark_complex_retrieval(self):
        """Benchmark complex retrieval operations."""
        logger.info("🔍 Benchmarking complex retrieval...")

        # Create temporary database with realistic data
        temp_dir = tempfile.mkdtemp()
        sqlite_path = os.path.join(temp_dir, "test.db")
        lmdb_dir = sqlite_path + ".lmdb"

        try:
            paths = Paths()
            paths.sqlite_path = sqlite_path
            paths.lmdb_dir = lmdb_dir

            memory_store = MemoryStore(paths)
            classifier = EnhancedRuleClassifierV2()
            facade = HotMemoryFacade(memory_store)

            # First, build up a complex memory graph
            setup_conversation = [
                "My name is Sarah Chen and I work at Google as a senior software engineer.",
                "I live in San Francisco with my husband Michael and our two kids.",
                "Michael works at Apple as a designer and we met at Stanford University.",
                "We have a golden retriever named Max and a cat named Luna.",
                "I graduated from Stanford in 2018 with a degree in Computer Science.",
                "Michael studied graphic design at RISD and graduated in 2017.",
                "We bought our house in the Mission District in 2020.",
                "My parents live in New York and Michael's parents live in Seattle.",
                "We enjoy hiking in Marin County on weekends.",
                "Last summer we visited Japan for two weeks and loved Tokyo."
            ]

            session_id = "retrieval_benchmark"

            # Build the memory graph
            logger.info("  Building memory graph...")
            for turn_id, message in enumerate(setup_conversation):
                intent_result = classifier.classify(message)
                facade.process_turn(message, session_id, turn_id)

            # Now test complex retrieval queries
            retrieval_queries = [
                "What do you know about my family?",
                "Tell me about my educational background.",
                "Where do I live and work?",
                "What pets do I have?",
                "Tell me about my recent travels."
            ]

            retrieval_times = []

            for turn_id_offset, query in enumerate(retrieval_queries):
                start = time.perf_counter()

                intent_result = classifier.classify(query)
                memory_result = facade.process_turn(query, session_id, len(setup_conversation) + turn_id_offset)

                end = time.perf_counter()
                retrieval_time = (end - start) * 1000
                retrieval_times.append(retrieval_time)

                memories_retrieved = len(memory_result.bullets)
                logger.info(f"  Query: '{query[:30]}...' - {retrieval_time:.2f}ms - {memories_retrieved} memories")

            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)

            self.results['complex_retrieval'] = {
                'avg_latency_ms': avg_retrieval_time,
                'queries_tested': len(retrieval_queries),
                'individual_times_ms': retrieval_times,
                'target_latency_ms': 100  # Complex retrieval target
            }

            logger.info(f"✅ Complex retrieval: {avg_retrieval_time:.2f}ms avg")

            # Cleanup
            os.unlink(sqlite_path)
            import shutil
            shutil.rmtree(lmdb_dir)

            return avg_retrieval_time < 100

        except Exception as e:
            logger.error(f"❌ Complex retrieval benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['complex_retrieval'] = {'error': str(e)}
            return False

    async def benchmark_full_pipeline(self):
        """Benchmark full pipeline with realistic data."""
        logger.info("🚀 Benchmarking full pipeline...")

        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        sqlite_path = os.path.join(temp_dir, "test.db")
        lmdb_dir = sqlite_path + ".lmdb"

        try:
            # Initialize components
            paths = Paths()
            paths.sqlite_path = sqlite_path
            paths.lmdb_dir = lmdb_dir

            memory_store = MemoryStore(paths)  # No initialize method needed
            classifier = EnhancedRuleClassifierV2()
            facade = HotMemoryFacade(memory_store)

            # Test pipeline with realistic conversation
            test_conversation = [
                "Hi there!",
                "My name is Sarah and I work at Google as a software engineer.",
                "I live in San Francisco and have a dog named Max.",
                "What do you think about working in tech?",
                "Can you tell me about my background?",  # This should trigger retrieval
                "Actually, I meant to say I work at Meta, not Google."  # This should trigger correction
            ]

            session_id = "benchmark_session"
            total_times = []

            for i, message in enumerate(test_conversation):
                start = time.perf_counter()

                # Step 1: Intent classification (real operation)
                intent_result = classifier.classify(message)

                # Step 2: ACTUAL memory processing with HotMemoryFacade
                # This tests extraction, storage, and potentially retrieval
                memory_result = facade.process_turn(message, session_id, i)

                end = time.perf_counter()
                total_times.append((end - start) * 1000)

                logger.info(f"  Message {i+1}: {(end - start) * 1000:.2f}ms - Intent: {intent_result.intent}")

            avg_time = sum(total_times) / len(total_times)

            self.results['full_pipeline'] = {
                'avg_latency_ms': avg_time,
                'messages_processed': len(test_conversation),
                'target_latency_ms': 300,  # From your .env HOTMEM_TOTAL_BUDGET_MS
                'individual_times_ms': total_times
            }

            logger.info(f"✅ Full pipeline: {avg_time:.2f}ms avg")

            # Cleanup
            os.unlink(sqlite_path)
            import shutil
            shutil.rmtree(lmdb_dir)

            return avg_time < 300

        except Exception as e:
            logger.error(f"❌ Full pipeline benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['full_pipeline'] = {'error': str(e)}
            return False

    async def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        logger.info("🏁 Starting comprehensive benchmark suite...")
        logger.info(f"📊 Configuration loaded from .env")

        benchmarks = [
            ("Intent Classification", self.benchmark_intent_classification),
            ("Memory Extraction", self.benchmark_memory_extraction),
            ("Context Building", self.benchmark_context_building),
            ("Complex Retrieval", self.benchmark_complex_retrieval),
            ("Full Pipeline", self.benchmark_full_pipeline)
        ]

        results = {}
        for name, benchmark_func in benchmarks:
            try:
                success = await benchmark_func()
                results[name] = {"success": success, **self.results.get(name.split()[1].lower(), {})}
            except Exception as e:
                logger.error(f"❌ {name} benchmark failed: {e}")
                results[name] = {"success": False, "error": str(e)}

        # Generate comprehensive report
        report = {
            "timestamp": time.time(),
            "configuration": "Loaded from .env",
            "results": results,
            "summary": self.generate_summary(results)
        }

        # Save report
        with open("benchmark_results.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("📄 Benchmark report saved to benchmark_results.json")

        # Print summary
        self.print_summary(results)

        return report

    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("success", False))

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "recommendations": self.generate_recommendations(results)
        }

    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        for test_name, result in results.items():
            if not result.get("success", False):
                if "latency" in result:
                    recommendations.append(f"Optimize {test_name} - current latency exceeds target")
                elif "error" in result:
                    recommendations.append(f"Fix {test_name} - {result['error']}")

        if not recommendations:
            recommendations.append("All benchmarks within target thresholds!")

        return recommendations

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("🏆 BENCHMARK RESULTS SUMMARY")
        print("="*60)

        for test_name, result in results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            latency = result.get("avg_latency_ms", "N/A")
            target = result.get("target_latency_ms", "N/A")

            print(f"{test_name:20} {status:8} {latency:>8}ms (target: {target}ms)")

        print("="*60)

async def main():
    """Main entry point."""
    print("🚀 LocalCat Quick Benchmarks")
    print("=" * 50)

    benchmarks = QuickBenchmarks()
    report = await benchmarks.run_all_benchmarks()

    return report

if __name__ == "__main__":
    asyncio.run(main())