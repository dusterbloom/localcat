#!/usr/bin/env python3
"""
Staged Runtime Policy for ASI1 Graph Extraction

Implements the recommended staged extraction policy based on complexity:
- < 0.45: YAML + distilled Judge only (fast path)
- 0.45-0.65: YAML + distilled Judge → SLM repair on gray-zone triples
- > 0.65: Hybrid SpaCy+LLM for batch/offline or tightly budgeted calls

Usage:
    python scripts/staged_extraction_policy.py --text "John works at Google"
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from core.memory.extractors.yaml_extractor import YAMLExtractor
from scripts.eval_extraction_ab import ComplexityAnalyzer
from core.memory.judge import GraphJudge


class StagedExtractionPolicy:
    """Implements staged extraction based on complexity"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.complexity_analyzer = ComplexityAnalyzer()

        # Thresholds from analysis
        self.FAST_THRESHOLD = 0.45
        self.MEDIUM_THRESHOLD = 0.65

        # Initialize extractors lazily
        self._yaml_extractor = None
        self._slm_extractor = None
        self._hybrid_extractor = None

        # Configure based on environment
        self._configure_from_env()

    def _configure_from_env(self):
        """Load configuration from environment"""
        # Override thresholds if specified
        self.FAST_THRESHOLD = float(os.getenv("STAGE_FAST_THRESHOLD", "0.45"))
        self.MEDIUM_THRESHOLD = float(os.getenv("STAGE_MEDIUM_THRESHOLD", "0.65"))

        # Judge settings
        os.environ["YAML_GRAPH_JUDGE"] = "on"
        if os.path.exists("models/graph_judge.json"):
            os.environ["YAML_GRAPH_JUDGE_MODEL"] = "models/graph_judge.json"

        # Gray-zone logging
        os.environ.setdefault("YAML_GRAPH_JUDGE_GRAY_BAND", "0.10")
        os.environ["YAML_GRAPH_JUDGE_GRAYZONE_LOG"] = "data/judge_grayzone.jsonl"
        try:
            self.gray_band = float(os.getenv("YAML_GRAPH_JUDGE_GRAY_BAND", "0.10"))
        except Exception:
            self.gray_band = 0.10
        self._judge = GraphJudge.from_env()

    @property
    def yaml_extractor(self):
        """Lazy load YAML extractor"""
        if self._yaml_extractor is None:
            self._yaml_extractor = YAMLExtractor(self.yaml_path)
        return self._yaml_extractor

    @property
    def slm_extractor(self):
        """Lazy load SLM extractor"""
        if self._slm_extractor is None:
            # Check if SLM is available
            if os.getenv("SLM_REFINEMENT_ENABLED", "false").lower() == "true":
                try:
                    from core.memory.extractors.hybrid_slm import YAMLWithSLMRefinement
                    self._slm_extractor = YAMLWithSLMRefinement(self.yaml_path)
                except ImportError:
                    logger.warning("SLM refinement not available, falling back to YAML")
                    self._slm_extractor = self.yaml_extractor
            else:
                self._slm_extractor = self.yaml_extractor
        return self._slm_extractor

    @property
    def hybrid_extractor(self):
        """Lazy load hybrid extractor"""
        if self._hybrid_extractor is None:
            # Check if LM Studio is available
            lm_studio_url = os.getenv("LLM_JUDGE_BASE_URL", "http://127.0.0.1:1234/v1")
            try:
                import requests
                response = requests.get(f"{lm_studio_url}/models", timeout=1)
                if response.status_code == 200:
                    from core.memory.extractors.recovered_hybrid import HybridRelationExtractor
                    self._hybrid_extractor = HybridRelationExtractor()
                else:
                    raise ConnectionError("LM Studio not available")
            except Exception as e:
                logger.warning(f"Hybrid extraction not available: {e}, falling back to SLM")
                self._hybrid_extractor = self.slm_extractor
        return self._hybrid_extractor

    def extract(self, text: str, lang: str = "en") -> Tuple[List[Tuple[str, str, str]], dict]:
        """
        Extract triples using staged policy based on complexity

        Returns:
            (triples, metadata) where metadata includes:
            - complexity: float
            - stage: str (fast/medium/complex)
            - latency_ms: float
            - method: str
        """
        # Assess complexity
        complexity = self.complexity_analyzer.assess(text)

        start_time = time.perf_counter()

        # Stage 1: Fast path (< 0.45 complexity)
        if complexity.score < self.FAST_THRESHOLD:
            logger.debug(f"Fast path: complexity={complexity.score:.2f}")
            entities, triples, neg, doc = self.yaml_extractor.extract(text, lang)
            triples = self.yaml_extractor.refine(text, triples, doc)
            method = "yaml_judge"
            stage = "fast"

        # Stage 2: Medium path (0.45-0.65 complexity)
        elif complexity.score < self.MEDIUM_THRESHOLD:
            logger.debug(f"Medium path: complexity={complexity.score:.2f}")

            # Start with YAML+Judge
            entities, triples, neg, doc = self.yaml_extractor.extract(text, lang)
            triples = self.yaml_extractor.refine(text, triples, doc)

            # Apply SLM refinement only if gray-zone triples are present
            method = "yaml_judge"
            if self.slm_extractor != self.yaml_extractor and self._judge.enabled():
                # Score triples and detect gray-zone around threshold
                try:
                    thr = self._judge.cfg.threshold
                except Exception:
                    thr = 0.5
                gray = []
                for s, r, d in triples:
                    try:
                        sc = self._judge.score(s, r, d, doc)
                    except Exception:
                        sc = 0.0
                    if abs(sc - thr) <= self.gray_band:
                        gray.append((s, r, d))
                if gray:
                    triples = self.slm_extractor.refine_triples(triples, text, lang)
                    method = "yaml_judge_slm"
            stage = "medium"

        # Stage 3: Complex path (> 0.65 complexity)
        else:
            logger.debug(f"Complex path: complexity={complexity.score:.2f}")

            # Use hybrid SpaCy+LLM for best accuracy
            if hasattr(self.hybrid_extractor, 'extract_with_complexity'):
                triples = self.hybrid_extractor.extract_with_complexity(text, complexity.score, lang)
            else:
                # Fallback if method not available
                result = self.hybrid_extractor.extract(text, lang)
                if isinstance(result, list):
                    triples = result
                elif isinstance(result, tuple) and len(result) >= 2:
                    triples = result[1]
                else:
                    triples = []

            method = "hybrid_llm"
            stage = "complex"

        latency_ms = (time.perf_counter() - start_time) * 1000

        metadata = {
            "complexity": complexity.score,
            "stage": stage,
            "method": method,
            "latency_ms": latency_ms,
            "clause_count": complexity.clause_count,
            "entity_count": complexity.entity_count,
            "has_passive": complexity.has_passive,
            "has_relative": complexity.has_relative,
        }

        logger.info(f"Extracted {len(triples)} triples using {method} "
                   f"(complexity={complexity.score:.2f}, latency={latency_ms:.1f}ms)")

        return triples, metadata


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Staged extraction policy for ASI1")
    parser.add_argument("--text", required=True, help="Text to extract from")
    parser.add_argument("--yaml", default="archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml",
                       help="Path to YAML spec")
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")

    # Initialize staged extractor
    extractor = StagedExtractionPolicy(args.yaml)

    # Extract triples
    triples, metadata = extractor.extract(args.text, args.lang)

    # Display results
    print(f"\n📊 Extraction Results")
    print("=" * 50)
    print(f"Text: {args.text}")
    print(f"Complexity: {metadata['complexity']:.2f} ({metadata['stage']})")
    print(f"Method: {metadata['method']}")
    print(f"Latency: {metadata['latency_ms']:.1f}ms")
    print(f"\n📝 Extracted Triples ({len(triples)}):")
    for i, (s, r, d) in enumerate(triples, 1):
        print(f"  {i}. ({s}, {r}, {d})")

    # Check gray-zone logging
    gray_log = os.getenv("YAML_GRAPH_JUDGE_GRAYZONE_LOG", "data/judge_grayzone.jsonl")
    if os.path.exists(gray_log):
        with open(gray_log) as f:
            gray_count = sum(1 for _ in f)
        if gray_count > 0:
            print(f"\n⚠️  {gray_count} gray-zone cases logged for retraining")


if __name__ == "__main__":
    main()
