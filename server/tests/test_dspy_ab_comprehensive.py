#!/usr/bin/env python3.12
"""
Comprehensive A/B Test: DSPy vs Rule-Based System
===============================================

Compare current LocalCat rule-based system with DSPy + Osaurs across:
- Intent Classification Accuracy & Speed
- Memory Extraction Quality & Performance
- Retrieval Effectiveness & Context Building
- End-to-End Pipeline Performance

Uses llama-3.2-3b-instruct-4bit (only available model in Osaurs)
"""

import os
import sys
import time
import json
import statistics
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add server to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Import current pipeline components
try:
    from components.memory.memory_intent import get_intent_classifier, IntentType
    from components.extraction.enhanced_level3_extractor import QualityExtractor
    from components.memory.hotmemory_facade import HotMemoryFacade
    from components.memory.memory_store import MemoryStore, Paths
    from components.context.context_orchestrator import pack_context
    from components.context.memory_config import get_global_config
    CURRENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Current pipeline components not available: {e}")
    CURRENT_AVAILABLE = False

# Import DSPy components
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

@dataclass
class TestCase:
    """Single test case for A/B comparison"""
    text: str
    category: str  # 'intent', 'extraction', 'retrieval', 'complex_reasoning'
    difficulty: str  # 'easy', 'medium', 'hard'
    expected_intent: Optional[str] = None
    expected_entities: Optional[List[str]] = None
    expected_triples: Optional[List[Tuple[str, str, str]]] = None
    context_facts: Optional[List[Tuple[str, str, str]]] = None

@dataclass
class ComponentResult:
    """Result from component execution"""
    time_ms: float
    accuracy_score: float
    result_data: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class SystemResult:
    """Complete system comparison result"""
    current_result: ComponentResult
    dspy_result: ComponentResult
    winner: str  # 'current', 'dspy', 'tie'
    performance_improvement: float  # percentage improvement (positive = dspy faster)
    accuracy_improvement: float  # percentage improvement (positive = dspy more accurate)

class DSPyABTestFramework:
    """Comprehensive A/B test framework"""

    def __init__(self):
        self.current_classifier = None
        self.current_extractor = None
        self.dspy_configured = False
        self.dspy_lm = None
        self.results = []

        # Model configuration (using only available model)
        self.model_name = "lmstudio-community/gemma-3-270m-it-MLX-8bit"

        # Comprehensive test cases
        self.test_cases = [
            # Intent Classification Tests
            TestCase(
                text="Hello there!",
                category="intent",
                difficulty="easy",
                expected_intent="REACTION",
                expected_entities=[],
                expected_triples=[]
            ),
            TestCase(
                text="What is the weather like today?",
                category="intent",
                difficulty="easy",
                expected_intent="PURE_QUESTION",
                expected_entities=[],
                expected_triples=[]
            ),
            TestCase(
                text="I work at Google as a software engineer",
                category="intent",
                difficulty="medium",
                expected_intent="FACT_STATEMENT",
                expected_entities=["Google", "software engineer"],
                expected_triples=[("I", "work_at", "Google"), ("I", "am", "software engineer")]
            ),
            TestCase(
                text="No, actually I meant Microsoft",
                category="intent",
                difficulty="medium",
                expected_intent="CORRECTION",
                expected_entities=[],
                expected_triples=[]
            ),

            # Memory Extraction Tests
            TestCase(
                text="My dog Potola is 5 years old",
                category="extraction",
                difficulty="medium",
                expected_intent="FACT_STATEMENT",
                expected_entities=["dog Potola", "5 years old"],
                expected_triples=[("dog Potola", "is", "5 years old")]
            ),
            TestCase(
                text="Sarah works at Google as a software engineer",
                category="extraction",
                difficulty="medium",
                expected_intent="FACT_STATEMENT",
                expected_entities=["Sarah", "Google", "software engineer"],
                expected_triples=[("Sarah", "works_at", "Google"), ("Sarah", "is", "software engineer")]
            ),
            TestCase(
                text="Michael Chen is a cardiologist at Seattle General Hospital",
                category="extraction",
                difficulty="medium",
                expected_intent="FACT_STATEMENT",
                expected_entities=["Michael Chen", "cardiologist", "Seattle General Hospital"],
                expected_triples=[("Michael Chen", "is", "cardiologist"), ("Michael Chen", "works_at", "Seattle General Hospital")]
            ),

            # Complex Reasoning Tests (require context)
            TestCase(
                text="What hospital does Emma's father work at?",
                category="complex_reasoning",
                difficulty="hard",
                expected_intent="PURE_QUESTION",
                expected_entities=["Emma", "father", "hospital"],
                expected_triples=[],
                context_facts=[
                    ("Sarah", "has_child", "Emma"),
                    ("Sarah", "married_to", "Michael_Chen"),
                    ("Michael_Chen", "is", "cardiologist"),
                    ("Michael_Chen", "works_at", "Seattle_General_Hospital")
                ]
            ),
            TestCase(
                text="Who is married to the cardiologist?",
                category="complex_reasoning",
                difficulty="hard",
                expected_intent="PURE_QUESTION",
                expected_entities=["cardiologist", "married"],
                expected_triples=[],
                context_facts=[
                    ("Sarah", "married_to", "Michael_Chen"),
                    ("Michael_Chen", "is", "cardiologist")
                ]
            ),
            TestCase(
                text="What city does the software engineer who has a child live in?",
                category="complex_reasoning",
                difficulty="hard",
                expected_intent="PURE_QUESTION",
                expected_entities=["software engineer", "child", "city"],
                expected_triples=[],
                context_facts=[
                    ("Sarah", "is", "software_engineer"),
                    ("Sarah", "has_child", "Emma"),
                    ("Sarah", "lives_in", "Seattle")
                ]
            )
        ]

    def setup_current_system(self):
        """Setup current rule-based system"""
        if not CURRENT_AVAILABLE:
            print("❌ Current pipeline components not available")
            return False

        print("🔧 Setting up current rule-based system...")
        try:
            self.current_classifier = get_intent_classifier()
            self.current_extractor = QualityExtractor()
            print("✅ Current system ready")
            return True
        except Exception as e:
            print(f"❌ Failed to setup current system: {e}")
            return False

    def setup_dspy_system(self):
        """Setup DSPy with Osaurs backend"""
        if not DSPY_AVAILABLE:
            print("❌ DSPy not available")
            return False

        print("🔧 Setting up DSPy with Osaurs...")

        # Check if Osaurs is running
        try:
            import requests
            response = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
            if response.status_code != 200:
                print("❌ Osaurs not running")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Osaurs: {e}")
            return False

        # Configure DSPy
        try:
            self.dspy_lm = dspy.LM(
                model=f"openai/{self.model_name}",
                api_base="http://127.0.0.1:8000/v1",
                api_key="dummy",
                max_tokens=256,
                temperature=0.1,
                top_p=0.9
            )
            dspy.settings.configure(lm=self.dspy_lm)
            self.dspy_configured = True
            print("✅ DSPy configured successfully")
            return True
        except Exception as e:
            print(f"❌ DSPy configuration failed: {e}")
            return False

    def test_current_intent_classification(self, test_case: TestCase) -> ComponentResult:
        """Test current intent classification system"""
        start_time = time.time()

        try:
            result = self.current_classifier.analyze(test_case.text)
            elapsed_time = (time.time() - start_time) * 1000

            # Calculate accuracy
            accuracy = 0.0
            if result and hasattr(result, 'intent'):
                intent_match = result.intent.value == test_case.expected_intent
                accuracy = 1.0 if intent_match else 0.0

            return ComponentResult(
                time_ms=elapsed_time,
                accuracy_score=accuracy,
                result_data={
                    "intent": result.intent.value if result else None,
                    "confidence": result.confidence if result else 0.0,
                    "should_extract": result.should_extract_facts if result else False,
                    "should_retrieve": result.requires_retrieval if result else False
                }
            )
        except Exception as e:
            return ComponentResult(
                time_ms=0,
                accuracy_score=0.0,
                result_data={},
                error=str(e)
            )

    def test_dspy_intent_classification(self, test_case: TestCase) -> ComponentResult:
        """Test DSPy intent classification"""
        if not self.dspy_configured:
            return ComponentResult(0, 0.0, {}, "DSPy not configured")

        start_time = time.time()

        try:
            # Create DSPy intent classification signature
            class IntentSignature(dspy.Signature):
                """Classify user intent"""
                text: str = dspy.InputField(desc="User input text")
                intent: str = dspy.OutputField(desc="Classified intent (REACTION/PURE_QUESTION/FACT_STATEMENT/CORRECTION)")
                confidence: float = dspy.OutputField(desc="Confidence score 0-1")
                should_extract: bool = dspy.OutputField(desc="Should extract facts")
                should_retrieve: bool = dspy.OutputField(desc="Should retrieve information")

            class IntentClassifier(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.classify = dspy.ChainOfThought(IntentSignature)

                def forward(self, text: str):
                    return self.classify(text=text)

            classifier = IntentClassifier()
            result = classifier(text=test_case.text)
            elapsed_time = (time.time() - start_time) * 1000

            # Parse result
            intent = getattr(result, 'intent', 'UNKNOWN')
            confidence = getattr(result, 'confidence', 0.0)
            should_extract = getattr(result, 'should_extract', False)
            should_retrieve = getattr(result, 'should_retrieve', False)

            # Calculate accuracy
            accuracy = 1.0 if intent == test_case.expected_intent else 0.0

            return ComponentResult(
                time_ms=elapsed_time,
                accuracy_score=accuracy,
                result_data={
                    "intent": intent,
                    "confidence": confidence,
                    "should_extract": should_extract,
                    "should_retrieve": should_retrieve
                }
            )
        except Exception as e:
            return ComponentResult(
                time_ms=0,
                accuracy_score=0.0,
                result_data={},
                error=str(e)
            )

    def test_current_memory_extraction(self, test_case: TestCase) -> ComponentResult:
        """Test current memory extraction system"""
        start_time = time.time()

        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(test_case.text)

            result = self.current_extractor.extract_quality_kg(doc)
            elapsed_time = (time.time() - start_time) * 1000

            # Calculate accuracy based on expected entities and triples
            extracted_entities = result.get('entities', [])
            extracted_relations = result.get('relations', [])

            # Entity accuracy
            expected_entities = test_case.expected_entities or []
            entity_overlap = len(set(expected_entities) & set(extracted_entities)) / max(len(expected_entities), 1)

            # Triple accuracy
            expected_triples = test_case.expected_triples or []
            triple_matches = 0
            for expected_s, expected_r, expected_o in expected_triples:
                for rel in extracted_relations:
                    if (rel.get('subject') == expected_s and
                        rel.get('relation') == expected_r and
                        rel.get('object') == expected_o):
                        triple_matches += 1
                        break

            triple_accuracy = triple_matches / max(len(expected_triples), 1)
            accuracy = (entity_overlap + triple_accuracy) / 2

            return ComponentResult(
                time_ms=elapsed_time,
                accuracy_score=accuracy,
                result_data={
                    "entities": extracted_entities,
                    "relations": extracted_relations,
                    "relation_count": len(extracted_relations)
                }
            )
        except Exception as e:
            return ComponentResult(
                time_ms=0,
                accuracy_score=0.0,
                result_data={},
                error=str(e)
            )

    def test_dspy_memory_extraction(self, test_case: TestCase) -> ComponentResult:
        """Test DSPy memory extraction"""
        if not self.dspy_configured:
            return ComponentResult(0, 0.0, {}, "DSPy not configured")

        start_time = time.time()

        try:
            # Create DSPy extraction signature
            class ExtractionSignature(dspy.Signature):
                """Extract entities and relations from text"""
                text: str = dspy.InputField(desc="Input text to analyze")
                entities: List[str] = dspy.OutputField(desc="Extracted entities")
                relations: List[str] = dspy.OutputField(desc="Extracted relations as 'subject relation object'")

            class Extractor(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.extract = dspy.ChainOfThought(ExtractionSignature)

                def forward(self, text: str):
                    return self.extract(text=text)

            extractor = Extractor()
            result = extractor(text=test_case.text)
            elapsed_time = (time.time() - start_time) * 1000

            # Parse results
            entities = getattr(result, 'entities', [])
            relations = getattr(result, 'relations', [])

            # Parse relations into triples
            parsed_triples = []
            for rel_str in relations:
                parts = rel_str.split()
                if len(parts) >= 3:
                    parsed_triples.append((parts[0], parts[1], ' '.join(parts[2:])))

            # Calculate accuracy
            expected_entities = test_case.expected_entities or []
            entity_overlap = len(set(expected_entities) & set(entities)) / max(len(expected_entities), 1)

            expected_triples = test_case.expected_triples or []
            triple_matches = 0
            for expected_s, expected_r, expected_o in expected_triples:
                for parsed_s, parsed_r, parsed_o in parsed_triples:
                    if (parsed_s == expected_s and parsed_r == expected_r and parsed_o == expected_o):
                        triple_matches += 1
                        break

            triple_accuracy = triple_matches / max(len(expected_triples), 1)
            accuracy = (entity_overlap + triple_accuracy) / 2

            return ComponentResult(
                time_ms=elapsed_time,
                accuracy_score=accuracy,
                result_data={
                    "entities": entities,
                    "relations": parsed_triples,
                    "relation_count": len(parsed_triples)
                }
            )
        except Exception as e:
            return ComponentResult(
                time_ms=0,
                accuracy_score=0.0,
                result_data={},
                error=str(e)
            )

    def run_ab_test(self) -> Dict[str, Any]:
        """Run comprehensive A/B test"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE A/B TEST: DSPy vs Rule-Based System")
        print(f"Model: {self.model_name}")
        print(f"Test Cases: {len(self.test_cases)}")
        print(f"{'='*80}")

        # Setup systems
        current_available = self.setup_current_system()
        dspy_available = self.setup_dspy_system()

        if not current_available and not dspy_available:
            print("❌ Neither system is available")
            return {}

        results = {
            "model": self.model_name,
            "test_cases": [],
            "summary": {
                "current_stats": {},
                "dspy_stats": {},
                "category_stats": {},
                "recommendations": []
            }
        }

        category_results = {
            "intent": {"current": [], "dspy": []},
            "extraction": {"current": [], "dspy": []},
            "complex_reasoning": {"current": [], "dspy": []}
        }

        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n--- Test Case {i}/{len(self.test_cases)} ---")
            print(f"Text: \"{test_case.text}\"")
            print(f"Category: {test_case.category}, Difficulty: {test_case.difficulty}")

            # Test current system
            current_result = None
            if current_available:
                if test_case.category == "intent":
                    current_result = self.test_current_intent_classification(test_case)
                elif test_case.category == "extraction":
                    current_result = self.test_current_memory_extraction(test_case)
                else:
                    current_result = ComponentResult(0, 0.0, {}, "Not supported by current system")
            else:
                current_result = ComponentResult(0, 0.0, {}, "Current system not available")

            # Test DSPy system
            dspy_result = None
            if dspy_available:
                if test_case.category == "intent":
                    dspy_result = self.test_dspy_intent_classification(test_case)
                elif test_case.category == "extraction":
                    dspy_result = self.test_dspy_memory_extraction(test_case)
                elif test_case.category == "complex_reasoning":
                    dspy_result = self.test_dspy_complex_reasoning(test_case)
                else:
                    dspy_result = ComponentResult(0, 0.0, {}, "Not supported")
            else:
                dspy_result = ComponentResult(0, 0.0, {}, "DSPy not available")

            # Determine winner
            winner = "tie"
            perf_improvement = 0.0
            acc_improvement = 0.0

            if current_result and dspy_result and not current_result.error and not dspy_result.error:
                if current_result.time_ms > 0 and dspy_result.time_ms > 0:
                    perf_improvement = (current_result.time_ms - dspy_result.time_ms) / current_result.time_ms * 100
                if current_result.accuracy_score >= 0 and dspy_result.accuracy_score >= 0:
                    acc_improvement = (dspy_result.accuracy_score - current_result.accuracy_score) * 100

                if acc_improvement > 10:  # 10% threshold
                    winner = "dspy"
                elif acc_improvement < -10:
                    winner = "current"
                else:
                    # Tie-breaker: prefer faster system
                    if perf_improvement > 0:
                        winner = "dspy"
                    else:
                        winner = "current"

            print(f"  Current: {current_result.time_ms:.1f}ms, Accuracy: {current_result.accuracy_score:.2f}")
            print(f"  DSPy:    {dspy_result.time_ms:.1f}ms, Accuracy: {dspy_result.accuracy_score:.2f}")
            print(f"  Winner:  {winner.upper()}")

            # Store results
            case_result = {
                "test_case": {
                    "text": test_case.text,
                    "category": test_case.category,
                    "difficulty": test_case.difficulty,
                    "expected_intent": test_case.expected_intent,
                    "expected_entities": test_case.expected_entities,
                    "expected_triples": test_case.expected_triples
                },
                "current": {
                    "time_ms": current_result.time_ms,
                    "accuracy": current_result.accuracy_score,
                    "result": current_result.result_data,
                    "error": current_result.error
                },
                "dspy": {
                    "time_ms": dspy_result.time_ms,
                    "accuracy": dspy_result.accuracy_score,
                    "result": dspy_result.result_data,
                    "error": dspy_result.error
                },
                "comparison": {
                    "winner": winner,
                    "performance_improvement_percent": perf_improvement,
                    "accuracy_improvement_percent": acc_improvement
                }
            }
            results["test_cases"].append(case_result)

            # Store category stats
            if test_case.category in category_results:
                if current_result and not current_result.error:
                    category_results[test_case.category]["current"].append(current_result)
                if dspy_result and not dspy_result.error:
                    category_results[test_case.category]["dspy"].append(dspy_result)

        # Calculate summary statistics
        self._calculate_summary_stats(results, category_results)

        # Print summary
        self._print_summary(results)

        return results

    def test_dspy_complex_reasoning(self, test_case: TestCase) -> ComponentResult:
        """Test DSPy complex reasoning (current system can't do this)"""
        if not self.dspy_configured:
            return ComponentResult(0, 0.0, {}, "DSPy not configured")

        start_time = time.time()

        try:
            # Create context from facts
            context = "Knowledge Graph Facts:\n"
            for fact in test_case.context_facts or []:
                context += f"- {fact[0]} {fact[1]} {fact[2]}\n"

            # Create reasoning signature
            class ReasoningSignature(dspy.Signature):
                """Complex multi-hop reasoning"""
                context: str = dspy.InputField(desc="Knowledge graph context")
                query: str = dspy.InputField(desc="Complex query to answer")
                reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
                answer: str = dspy.OutputField(desc="Final answer")

            class Reasoner(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.reason = dspy.ChainOfThought(ReasoningSignature)

                def forward(self, context: str, query: str):
                    return self.reason(context=context, query=query)

            reasoner = Reasoner()
            result = reasoner(context=context, query=test_case.text)
            elapsed_time = (time.time() - start_time) * 1000

            # Simple accuracy check: answer contains expected entities
            answer = getattr(result, 'answer', '')
            expected_entities = test_case.expected_entities or []
            entities_found = sum(1 for entity in expected_entities if entity.lower() in answer.lower())
            accuracy = entities_found / max(len(expected_entities), 1)

            return ComponentResult(
                time_ms=elapsed_time,
                accuracy_score=accuracy,
                result_data={
                    "answer": answer,
                    "reasoning": getattr(result, 'reasoning', ''),
                    "entities_found": entities_found
                }
            )
        except Exception as e:
            return ComponentResult(
                time_ms=0,
                accuracy_score=0.0,
                result_data={},
                error=str(e)
            )

    def _calculate_summary_stats(self, results: Dict, category_results: Dict):
        """Calculate summary statistics"""
        # Overall stats
        current_times = []
        current_accs = []
        dspy_times = []
        dspy_accs = []

        for case in results["test_cases"]:
            if case["current"]["time_ms"] > 0 and not case["current"]["error"]:
                current_times.append(case["current"]["time_ms"])
                current_accs.append(case["current"]["accuracy"])
            if case["dspy"]["time_ms"] > 0 and not case["dspy"]["error"]:
                dspy_times.append(case["dspy"]["time_ms"])
                dspy_accs.append(case["dspy"]["accuracy"])

        if current_times:
            results["summary"]["current_stats"] = {
                "avg_time_ms": statistics.mean(current_times),
                "median_time_ms": statistics.median(current_times),
                "avg_accuracy": statistics.mean(current_accs),
                "success_rate": len(current_times) / len(results["test_cases"])
            }

        if dspy_times:
            results["summary"]["dspy_stats"] = {
                "avg_time_ms": statistics.mean(dspy_times),
                "median_time_ms": statistics.median(dspy_times),
                "avg_accuracy": statistics.mean(dspy_accs),
                "success_rate": len(dspy_times) / len(results["test_cases"])
            }

        # Category stats
        for category, data in category_results.items():
            if data["current"] or data["dspy"]:
                cat_current_times = [r.time_ms for r in data["current"]]
                cat_dspy_times = [r.time_ms for r in data["dspy"]]
                cat_current_accs = [r.accuracy_score for r in data["current"]]
                cat_dspy_accs = [r.accuracy_score for r in data["dspy"]]

                results["summary"]["category_stats"][category] = {
                    "current": {
                        "avg_time_ms": statistics.mean(cat_current_times) if cat_current_times else 0,
                        "avg_accuracy": statistics.mean(cat_current_accs) if cat_current_accs else 0,
                        "test_count": len(cat_current_times)
                    },
                    "dspy": {
                        "avg_time_ms": statistics.mean(cat_dspy_times) if cat_dspy_times else 0,
                        "avg_accuracy": statistics.mean(cat_dspy_accs) if cat_dspy_accs else 0,
                        "test_count": len(cat_dspy_times)
                    }
                }

    def _print_summary(self, results: Dict):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE A/B TEST RESULTS")
        print(f"{'='*80}")

        summary = results["summary"]

        # Overall comparison
        if "current_stats" in summary and "dspy_stats" in summary:
            curr = summary["current_stats"]
            dsp = summary["dspy_stats"]

            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"   Current System: {curr['avg_time_ms']:.1f}ms avg, {curr['avg_accuracy']:.2f} accuracy")
            print(f"   DSPy System:    {dsp['avg_time_ms']:.1f}ms avg, {dsp['avg_accuracy']:.2f} accuracy")

            speed_diff = (curr['avg_time_ms'] - dsp['avg_time_ms']) / curr['avg_time_ms'] * 100
            acc_diff = (dsp['avg_accuracy'] - curr['avg_accuracy']) * 100

            print(f"   Speed Difference: {speed_diff:+.1f}% ({'DSPy faster' if speed_diff > 0 else 'Current faster'})")
            print(f"   Accuracy Difference: {acc_diff:+.1f}% ({'DSPy more accurate' if acc_diff > 0 else 'Current more accurate'})")

        # Category breakdown
        if "category_stats" in summary:
            print(f"\n📈 CATEGORY BREAKDOWN:")
            for category, stats in summary["category_stats"].items():
                curr_cat = stats["current"]
                dsp_cat = stats["dspy"]

                print(f"\n   {category.upper()}:")
                if curr_cat["test_count"] > 0:
                    print(f"     Current: {curr_cat['avg_time_ms']:.1f}ms, {curr_cat['avg_accuracy']:.2f} acc ({curr_cat['test_count']} tests)")
                if dsp_cat["test_count"] > 0:
                    print(f"     DSPy:    {dsp_cat['avg_time_ms']:.1f}ms, {dsp_cat['avg_accuracy']:.2f} acc ({dsp_cat['test_count']} tests)")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        self._generate_recommendations(results)

    def _generate_recommendations(self, results: Dict):
        """Generate deployment recommendations"""
        recommendations = []

        # Analyze results
        intent_winner = self._analyze_category_winner(results, "intent")
        extraction_winner = self._analyze_category_winner(results, "extraction")
        reasoning_winner = self._analyze_category_winner(results, "complex_reasoning")

        if intent_winner == "current":
            recommendations.append("• Use current system for intent classification (faster and more accurate)")
        elif intent_winner == "dspy":
            recommendations.append("• Consider DSPy for intent classification (better accuracy)")

        if extraction_winner == "current":
            recommendations.append("• Use current system for memory extraction (proven reliability)")
        elif extraction_winner == "dspy":
            recommendations.append("• Consider DSPy for memory extraction (better quality)")

        if reasoning_winner == "dspy":
            recommendations.append("• Deploy DSPy for complex reasoning (current system cannot handle)")
            recommendations.append("• Hybrid approach: current for simple tasks, DSPy for complex reasoning")

        # Performance considerations
        if "dspy_stats" in results["summary"]:
            dspy_perf = results["summary"]["dspy_stats"]
            if dspy_perf["avg_time_ms"] > 100:
                recommendations.append("• DSPy performance needs optimization for real-time use")
            elif dspy_perf["avg_time_ms"] < 50:
                recommendations.append("• DSPy performance is acceptable for production use")

        print("\n".join(recommendations))

    def _analyze_category_winner(self, results: Dict, category: str) -> str:
        """Analyze winner for a specific category"""
        category_cases = [case for case in results["test_cases"] if case["test_case"]["category"] == category]

        if not category_cases:
            return "tie"

        current_wins = 0
        dspy_wins = 0

        for case in category_cases:
            winner = case["comparison"]["winner"]
            if winner == "current":
                current_wins += 1
            elif winner == "dspy":
                dspy_wins += 1

        if current_wins > dspy_wins:
            return "current"
        elif dspy_wins > current_wins:
            return "dspy"
        else:
            return "tie"

    def save_results(self, results: Dict):
        """Save test results to file"""
        filename = f"dspy_ab_comprehensive_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main test function"""
    print("🚀 Starting Comprehensive A/B Test: DSPy vs Rule-Based System")

    # Run the test
    framework = DSPyABTestFramework()
    results = framework.run_ab_test()

    if results:
        framework.save_results(results)

        print(f"\n{'='*80}")
        print("TEST COMPLETE")
        print(f"{'='*80}")
        print("Results saved and recommendations provided above.")
    else:
        print("❌ Test failed to complete")

if __name__ == "__main__":
    main()