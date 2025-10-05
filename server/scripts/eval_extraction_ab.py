#!/usr/bin/env python3
"""
A/B Testing Framework for Graph Extraction Methods

Compares multiple extraction approaches:
- YAML baseline
- YAML with GraphJudge
- Hybrid SpaCy+LLM (recovered)
- YAML with SLM refinement
- DSPy enhanced

Usage:
  # Test single method
  python server/scripts/eval_extraction_ab.py \
      --method yaml \
      --dataset server/tests/data/yaml_eval_l1_en_medium.json

  # Compare all methods
  python server/scripts/eval_extraction_ab.py \
      --methods all \
      --dataset all \
      --output results/ab_test_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from loguru import logger

# Core imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
from core.memory.extractors.yaml_extractor import YAMLExtractor
from core.memory.eval_graph import prf1


@dataclass
class ComplexityScore:
    """Sentence complexity assessment"""
    score: float
    clause_count: int
    entity_count: int
    conjunction_count: int
    depth: int
    length: int
    has_passive: bool
    has_relative: bool


@dataclass
class ExtractionResult:
    """Result from single extraction"""
    method: str
    example_id: str
    text: str
    predictions: List[Tuple[str, str, str]]
    gold: List[Tuple[str, str, str]]
    latency_ms: float
    complexity: float
    metrics: Dict[str, float]
    error: Optional[str] = None
    timeout: bool = False


class ComplexityAnalyzer:
    """Assess sentence complexity for routing decisions"""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            from core.memory.nlp_manager import get_nlp_model
            self._nlp = get_nlp_model("en")
        return self._nlp

    def assess(self, text: str, doc=None) -> ComplexityScore:
        """Return detailed complexity assessment"""
        if doc is None:
            doc = self.nlp(text)

        # Count linguistic features
        clause_count = len([t for t in doc if t.dep_ in
                           ["ccomp", "xcomp", "advcl", "acl", "relcl"]])
        conjunction_count = len([t for t in doc if t.dep_ == "conj"])
        entity_count = len(doc.ents)

        # Parse tree depth
        depths = []
        for token in doc:
            depth = 0
            current = token
            while current.head != current and depth < 10:
                depth += 1
                current = current.head
            depths.append(depth)
        max_depth = max(depths) if depths else 0

        # Check for passive voice
        has_passive = any(t.dep_ == "auxpass" for t in doc)

        # Check for relative clauses
        has_relative = any(t.dep_ == "relcl" for t in doc)

        # Calculate weighted score
        score = 0.0
        score += min(clause_count * 0.15, 0.3)
        score += min(conjunction_count * 0.1, 0.2)
        score += min(max_depth / 10, 0.2)
        score += min(len(doc) / 50, 0.15)
        score += 0.1 if has_passive else 0
        score += min(entity_count * 0.02, 0.05)

        return ComplexityScore(
            score=min(score, 1.0),
            clause_count=clause_count,
            entity_count=entity_count,
            conjunction_count=conjunction_count,
            depth=max_depth,
            length=len(doc),
            has_passive=has_passive,
            has_relative=has_relative
        )


class BaseExtractor:
    """Base interface for all extractors"""

    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.__class__.__name__


class YAMLBaselineExtractor(BaseExtractor):
    """Pure YAML extraction"""

    def __init__(self, yaml_path: str):
        self.extractor = YAMLExtractor(yaml_path)

    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        _, triples, _, doc = self.extractor.extract(text, lang)
        triples = self.extractor.refine(text, triples, doc)
        return triples

    def get_name(self) -> str:
        return "yaml"


class YAMLWithJudgeExtractor(BaseExtractor):
    """YAML with GraphJudge filtering"""

    def __init__(self, yaml_path: str):
        # Enable judge via environment
        os.environ["YAML_GRAPH_JUDGE"] = "on"
        if os.path.exists("models/graph_judge.json"):
            os.environ["YAML_GRAPH_JUDGE_MODEL"] = "models/graph_judge.json"
        self.extractor = YAMLExtractor(yaml_path)

    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        _, triples, _, doc = self.extractor.extract(text, lang)
        triples = self.extractor.refine(text, triples, doc)  # Judge applied in refine
        return triples

    def get_name(self) -> str:
        return "yaml_judge"


class HybridSpacyLLMExtractor(BaseExtractor):
    """Recovered hybrid extractor with complexity routing"""

    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.complexity_threshold = float(os.getenv("HOTMEM_COMPLEXITY_THRESHOLD", "0.6"))
        self._initialized = False
        # Support for different LLM models
        self.llm_model = os.getenv("HOTMEM_LLM_ASSISTED_MODEL", "openai/gpt-oss-20b")
        self.llm_base_url = os.getenv("HOTMEM_LLM_ASSISTED_BASE_URL", "http://127.0.0.1:1234/v1")

    def _lazy_init(self):
        """Lazy initialization to avoid import issues"""
        if not self._initialized:
            try:
                # Try to import recovered extractor
                from core.memory.extractors.recovered_hybrid import HybridRelationExtractor
                self.extractor = HybridRelationExtractor()
                # Configure LLM settings if available
                if hasattr(self.extractor, 'configure_llm'):
                    self.extractor.configure_llm(
                        model=self.llm_model,
                        base_url=self.llm_base_url
                    )
                logger.info(f"Hybrid extractor using model: {self.llm_model}")
                self._initialized = True
            except ImportError:
                logger.warning("Hybrid extractor not found, using YAML fallback")
                from core.memory.extractors.yaml_extractor import YAMLExtractor
                yaml_path = "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
                self.extractor = YAMLExtractor(yaml_path)
                self._initialized = True

    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        self._lazy_init()

        # Assess complexity
        complexity = self.complexity_analyzer.assess(text)

        if hasattr(self.extractor, 'extract'):
            if hasattr(self.extractor, 'extract') and callable(self.extractor.extract):
                # Call appropriate method based on extractor type
                result = self.extractor.extract(text, lang)
                if isinstance(result, list):
                    return result
                elif isinstance(result, tuple) and len(result) >= 2:
                    # YAML extractor returns (entities, triples, ...)
                    return result[1] if len(result[1]) > 0 else []

        return []

    def get_name(self) -> str:
        return "hybrid_spacy"


class YAMLWithSLMExtractor(BaseExtractor):
    """YAML with SLM refinement using MLX"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.slm_enabled = os.getenv("SLM_REFINEMENT_ENABLED", "false").lower() == "true"
        self._extractor = None

    def _lazy_init(self):
        """Lazy initialization of SLM extractor"""
        if self._extractor is None:
            if self.slm_enabled:
                try:
                    from core.memory.extractors.hybrid_slm import YAMLWithSLMRefinement
                    # Use qwen2.5-coder-0.5b-instruct for better code/structure understanding
                    slm_model = os.getenv("SLM_MODEL_PATH", "qwen2.5-coder-0.5b-instruct")
                    self._extractor = YAMLWithSLMRefinement(
                        yaml_path=self.yaml_path,
                        slm_model=slm_model,
                        max_refinement_ms=int(os.getenv("SLM_MAX_REFINEMENT_MS", "150"))
                    )
                    logger.info(f"Initialized SLM extractor with model: {slm_model}")
                except ImportError as e:
                    logger.warning(f"SLM extractor not available: {e}, using YAML fallback")
                    from core.memory.extractors.yaml_extractor import YAMLExtractor
                    self._extractor = YAMLExtractor(self.yaml_path)
            else:
                from core.memory.extractors.yaml_extractor import YAMLExtractor
                self._extractor = YAMLExtractor(self.yaml_path)

    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        self._lazy_init()

        if hasattr(self._extractor, 'extract'):
            result = self._extractor.extract(text, lang)
            if isinstance(result, tuple) and len(result) >= 2:
                # YAMLWithSLMRefinement returns (entities, triples, neg_count, doc)
                _, triples, _, doc = result
                if hasattr(self._extractor, 'yaml_extractor'):
                    # Apply refinement if it's the SLM extractor
                    triples = self._extractor.yaml_extractor.refine(text, triples, doc)
                return triples
            elif isinstance(result, list):
                return result

        return []

    def get_name(self) -> str:
        return "yaml_slm"


class DSPyExtractor(BaseExtractor):
    """DSPy-based extraction"""

    def __init__(self):
        self._extractor = None

    def _lazy_init(self):
        if self._extractor is None:
            try:
                from archive.experimental.experiments.memory_system.extraction.dspy_extractor import (
                    DSPyEdgeExtractor
                )
                self._extractor = DSPyEdgeExtractor(
                    model=os.getenv("DSPY_MODEL", "openai/gpt-4o-mini"),
                    base_url=os.getenv("DSPY_BASE_URL"),
                    max_tokens=256
                )
            except Exception as e:
                logger.warning(f"DSPy extractor init failed: {e}")

    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        self._lazy_init()
        if self._extractor:
            # DSPy expects existing edges, so start with empty
            return self._extractor.extract_missing_edges(text, [])
        return []

    def get_name(self) -> str:
        return "dspy"


class ExtractionABTestHarness:
    """A/B testing framework for extraction methods"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.complexity_analyzer = ComplexityAnalyzer()
        self.extractors = self._initialize_extractors()

    def _initialize_extractors(self) -> Dict[str, BaseExtractor]:
        """Initialize all extraction methods"""
        extractors = {}

        # Always include YAML baseline
        extractors["yaml"] = YAMLBaselineExtractor(self.yaml_path)

        # Conditionally add other extractors
        if os.getenv("ENABLE_YAML_JUDGE", "true").lower() == "true":
            extractors["yaml_judge"] = YAMLWithJudgeExtractor(self.yaml_path)

        if os.getenv("ENABLE_HYBRID", "true").lower() == "true":
            extractors["hybrid_spacy"] = HybridSpacyLLMExtractor()

        if os.getenv("ENABLE_SLM", "true").lower() == "true":
            extractors["yaml_slm"] = YAMLWithSLMExtractor(self.yaml_path)

        if os.getenv("ENABLE_DSPY", "false").lower() == "true":
            extractors["dspy"] = DSPyExtractor()

        logger.info(f"Initialized extractors: {list(extractors.keys())}")
        return extractors

    async def test_extractor(
        self,
        method_name: str,
        extractor: BaseExtractor,
        example: Dict[str, Any]
    ) -> ExtractionResult:
        """Test single extractor with timeout"""

        text = example["text"]
        gold = [(s, r, d) for s, r, d in example["gold"]]

        # Assess complexity
        complexity = self.complexity_analyzer.assess(text)

        try:
            # Time the extraction
            start = time.perf_counter()

            # Extract with timeout
            predictions = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    extractor.extract,
                    text,
                    example.get("lang", "en")
                ),
                timeout=2.0  # 2 second timeout
            )

            latency = (time.perf_counter() - start) * 1000

            # Calculate metrics
            metrics = prf1(predictions, gold)

            return ExtractionResult(
                method=method_name,
                example_id=example.get("id", "unknown"),
                text=text,
                predictions=predictions,
                gold=gold,
                latency_ms=latency,
                complexity=complexity.score,
                metrics=metrics
            )

        except asyncio.TimeoutError:
            return ExtractionResult(
                method=method_name,
                example_id=example.get("id", "unknown"),
                text=text,
                predictions=[],
                gold=gold,
                latency_ms=2000,
                complexity=complexity.score,
                metrics={"precision": 0, "recall": 0, "f1": 0},
                timeout=True
            )
        except Exception as e:
            logger.error(f"Extractor {method_name} failed: {e}")
            return ExtractionResult(
                method=method_name,
                example_id=example.get("id", "unknown"),
                text=text,
                predictions=[],
                gold=gold,
                latency_ms=0,
                complexity=complexity.score,
                metrics={"precision": 0, "recall": 0, "f1": 0},
                error=str(e)
            )

    def _compute_bin_stats(self, bin_results: List[ExtractionResult]) -> Dict:
        """Compute statistics for a complexity bin"""
        if not bin_results:
            return {"count": 0, "f1_mean": 0, "f1_std": 0, "latency_mean": 0, "latency_p95": 0}

        f1_scores = [r.metrics["f1"] for r in bin_results if not r.timeout and not r.error]
        latencies = [r.latency_ms for r in bin_results if not r.timeout and not r.error]

        return {
            "count": len(bin_results),
            "f1_mean": np.mean(f1_scores) if f1_scores else 0,
            "f1_std": np.std(f1_scores) if f1_scores else 0,
            "latency_mean": np.mean(latencies) if latencies else 0,
            "latency_p95": np.percentile(latencies, 95) if latencies else 0,
        }

    async def run_comparison(
        self,
        dataset: List[Dict],
        methods: Optional[List[str]] = None
    ) -> Dict[str, List[ExtractionResult]]:
        """Run comparison across multiple methods"""

        if methods is None:
            methods = list(self.extractors.keys())

        results = defaultdict(list)

        for example in dataset:
            logger.debug(f"Testing example: {example.get('id', 'unknown')}")

            # Test each method
            tasks = []
            for method in methods:
                if method in self.extractors:
                    tasks.append(self.test_extractor(
                        method,
                        self.extractors[method],
                        example
                    ))

            # Run in parallel
            method_results = await asyncio.gather(*tasks)

            # Aggregate
            for result in method_results:
                results[result.method].append(result)

        return dict(results)

    def analyze_results(self, results: Dict[str, List[ExtractionResult]]) -> Dict:
        """Analyze and summarize results"""

        summary = {}

        for method, method_results in results.items():
            # Skip if no results
            if not method_results:
                continue

            # Aggregate metrics
            f1_scores = [r.metrics["f1"] for r in method_results if not r.error]
            precision_scores = [r.metrics["precision"] for r in method_results if not r.error]
            recall_scores = [r.metrics["recall"] for r in method_results if not r.error]
            latencies = [r.latency_ms for r in method_results if not r.timeout]

            # Group by complexity
            simple = [r for r in method_results if r.complexity < 0.4]
            medium = [r for r in method_results if 0.4 <= r.complexity < 0.7]
            complex = [r for r in method_results if r.complexity >= 0.7]

            summary[method] = {
                "overall": {
                    "f1_mean": np.mean(f1_scores) if f1_scores else 0,
                    "f1_std": np.std(f1_scores) if f1_scores else 0,
                    "precision_mean": np.mean(precision_scores) if precision_scores else 0,
                    "recall_mean": np.mean(recall_scores) if recall_scores else 0,
                    "latency_mean": np.mean(latencies) if latencies else 0,
                    "latency_p95": np.percentile(latencies, 95) if latencies else 0,
                    "timeout_rate": sum(1 for r in method_results if r.timeout) / len(method_results),
                    "error_rate": sum(1 for r in method_results if r.error) / len(method_results)
                },
                "by_complexity": {
                    "simple": {
                        "f1_mean": np.mean([r.metrics["f1"] for r in simple]) if simple else 0,
                        "count": len(simple)
                    },
                    "medium": {
                        "f1_mean": np.mean([r.metrics["f1"] for r in medium]) if medium else 0,
                        "count": len(medium)
                    },
                    "complex": {
                        "f1_mean": np.mean([r.metrics["f1"] for r in complex]) if complex else 0,
                        "count": len(complex)
                    }
                },
                "complexity_bins": {
                    "0.0-0.2": self._compute_bin_stats([r for r in method_results if 0.0 <= r.complexity < 0.2]),
                    "0.2-0.4": self._compute_bin_stats([r for r in method_results if 0.2 <= r.complexity < 0.4]),
                    "0.4-0.5": self._compute_bin_stats([r for r in method_results if 0.4 <= r.complexity < 0.5]),
                    "0.5-0.6": self._compute_bin_stats([r for r in method_results if 0.5 <= r.complexity < 0.6]),
                    "0.6-0.7": self._compute_bin_stats([r for r in method_results if 0.6 <= r.complexity < 0.7]),
                    "0.7-1.0": self._compute_bin_stats([r for r in method_results if 0.7 <= r.complexity <= 1.0]),
                }
            }

        return summary


def load_dataset(path: Path) -> List[Dict]:
    """Load evaluation dataset"""
    data = json.loads(path.read_text())
    # Handle both flat list and categorized format
    if isinstance(data, dict):
        # Categorized format
        all_examples = []
        for category, examples in data.items():
            all_examples.extend(examples)
        return all_examples
    return data


def print_results(summary: Dict):
    """Print formatted results"""

    print("\n" + "=" * 80)
    print("A/B TEST RESULTS: Graph Extraction Methods")
    print("=" * 80)

    # Overall performance
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"{'Method':<15} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Latency (ms)':>15} {'Timeouts':>10}")
    print("-" * 40)

    for method, stats in summary.items():
        overall = stats["overall"]
        print(f"{method:<15} "
              f"{overall['f1_mean']:>8.3f} "
              f"{overall['precision_mean']:>8.3f} "
              f"{overall['recall_mean']:>8.3f} "
              f"{overall['latency_mean']:>7.1f} (p95: {overall['latency_p95']:.1f}) "
              f"{overall['timeout_rate']:>9.1%}")

    # By complexity
    print("\n📈 PERFORMANCE BY COMPLEXITY")
    print("-" * 40)
    print(f"{'Method':<15} {'Simple F1':>10} {'Medium F1':>10} {'Complex F1':>10}")
    print("-" * 40)

    for method, stats in summary.items():
        by_comp = stats["by_complexity"]
        print(f"{method:<15} "
              f"{by_comp['simple']['f1_mean']:>10.3f} "
              f"{by_comp['medium']['f1_mean']:>10.3f} "
              f"{by_comp['complex']['f1_mean']:>10.3f}")

    # Detailed complexity bins
    print("\n📊 DETAILED COMPLEXITY BINS (F1 / Latency ms)")
    print("-" * 60)
    print(f"{'Method':<15} {'0.0-0.2':>9} {'0.2-0.4':>9} {'0.4-0.5':>9} {'0.5-0.6':>9} {'0.6-0.7':>9} {'0.7-1.0':>9}")
    print("-" * 60)

    for method, stats in summary.items():
        if "complexity_bins" in stats:
            bins = stats["complexity_bins"]
            bin_strs = []
            for bin_range in ["0.0-0.2", "0.2-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-1.0"]:
                if bin_range in bins and bins[bin_range]["count"] > 0:
                    f1 = bins[bin_range]["f1_mean"]
                    lat = bins[bin_range]["latency_mean"]
                    bin_strs.append(f"{f1:.2f}/{lat:.0f}")
                else:
                    bin_strs.append("-")
            print(f"{method:<15} " + " ".join(f"{s:>9}" for s in bin_strs))

    # Best complexity threshold analysis
    print("\n🎯 COMPLEXITY THRESHOLD ANALYSIS")
    print("-" * 40)
    for method, stats in summary.items():
        if "complexity_bins" in stats:
            bins = stats["complexity_bins"]
            # Find the bin with best F1/latency tradeoff
            best_threshold = None
            best_score = 0
            for threshold in [0.4, 0.5, 0.6, 0.7]:
                below = [b for k, b in bins.items() if float(k.split('-')[1]) <= threshold]
                above = [b for k, b in bins.items() if float(k.split('-')[0]) >= threshold]
                if below and above:
                    # Score based on F1 gain vs latency cost
                    avg_f1_below = np.mean([b["f1_mean"] for b in below if b["count"] > 0])
                    avg_f1_above = np.mean([b["f1_mean"] for b in above if b["count"] > 0])
                    avg_lat_below = np.mean([b["latency_mean"] for b in below if b["count"] > 0])
                    avg_lat_above = np.mean([b["latency_mean"] for b in above if b["count"] > 0])
                    if avg_lat_above > avg_lat_below:
                        score = (avg_f1_above - avg_f1_below) / (avg_lat_above / avg_lat_below)
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
            if best_threshold:
                print(f"{method}: Best threshold={best_threshold:.1f} (score={best_score:.3f})")

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)

    # Find best for each scenario
    best_speed = min(summary.items(), key=lambda x: x[1]["overall"]["latency_mean"])
    best_accuracy = max(summary.items(), key=lambda x: x[1]["overall"]["f1_mean"])

    # Find best balance (F1 * speed factor)
    best_balanced = max(
        summary.items(),
        key=lambda x: x[1]["overall"]["f1_mean"] * (1 - min(x[1]["overall"]["latency_mean"] / 1000, 0.5))
    )

    print(f"🚀 Fastest: {best_speed[0]} ({best_speed[1]['overall']['latency_mean']:.1f}ms)")
    print(f"🎯 Most Accurate: {best_accuracy[0]} (F1: {best_accuracy[1]['overall']['f1_mean']:.3f})")
    print(f"⚖️  Best Balance: {best_balanced[0]}")

    # Voice agent specific recommendation
    voice_suitable = [
        m for m, s in summary.items()
        if s["overall"]["latency_p95"] < 200 and s["overall"]["f1_mean"] > 0.6
    ]
    if voice_suitable:
        print(f"🎤 For Voice (<200ms p95): {', '.join(voice_suitable)}")

    print("=" * 80)


async def main():
    ap = argparse.ArgumentParser(description="A/B test extraction methods")
    ap.add_argument("--dataset", required=True, help="Path to test dataset")
    ap.add_argument("--yaml", default="server/archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml",
                    help="Path to YAML index")
    ap.add_argument("--methods", nargs="+", default=["yaml", "yaml_judge"],
                    help="Methods to test (default: yaml yaml_judge)")
    ap.add_argument("--output", help="Output JSON file for results")
    ap.add_argument("--lang", default="en", help="Language code")
    args = ap.parse_args()

    # Load dataset
    dataset = load_dataset(Path(args.dataset))
    logger.info(f"Loaded {len(dataset)} examples from {args.dataset}")

    # Initialize harness
    harness = ExtractionABTestHarness(args.yaml)

    # Run comparison
    if "all" in args.methods:
        methods = None  # Use all available
    else:
        methods = args.methods

    results = await harness.run_comparison(dataset, methods)

    # Analyze
    summary = harness.analyze_results(results)

    # Print results
    print_results(summary)

    # Save if requested
    if args.output:
        output_data = {
            "summary": summary,
            "config": {
                "dataset": args.dataset,
                "yaml": args.yaml,
                "methods": methods or list(harness.extractors.keys()),
                "timestamp": time.time()
            }
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())