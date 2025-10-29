"""
A/B Test Harness for Memory Metadata Formatting

Tests three format variants:
- Control (technical): [conf=0.83 rec=1.00]
- Variant A (emoji): ⭐⭐⭐🆕📌
- Variant B (minimal): +++ now

Measures metadata leaking and response quality.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock
from dataclasses import dataclass, asdict

from core.memory.retrieval import Retrieval, Candidate


@dataclass
class TestResult:
    """Result of a single test query."""
    scenario_id: int
    scenario_name: str
    format: str  # "technical", "emoji", "minimal"
    query: str
    bullet: str  # Generated memory bullet
    metadata_leaked: bool  # Did metadata appear in bullet?
    leaked_indicators: List[str]  # Which indicators leaked
    bullet_length: int  # Length in characters
    timestamp: float


class MetadataFormatTester:
    """Test harness for A/B testing memory metadata formats."""

    def __init__(self, test_queries_path: str = None):
        """Initialize tester with test queries."""
        if test_queries_path is None:
            test_queries_path = Path(__file__).parent / "test_queries.json"

        with open(test_queries_path) as f:
            data = json.load(f)
            self.scenarios = data["test_scenarios"]

    def create_mock_host(self) -> Mock:
        """Create a mock host for Retrieval instance."""
        host = Mock()
        host.entity_index = {}
        host.recency_buffer = []
        host.store = Mock()
        host.store.get_edge_usage = Mock(return_value=(0, 0))
        host.store.increment_edge_usage = Mock()
        host.store.get_turn_prosody = Mock(return_value=(0.5, {}))
        host.current_user_id = "test_user"
        host.current_session_id = "test_session"
        return host

    def create_test_candidates(self, scenario: Dict[str, Any]) -> List[Tuple[float, Candidate, Dict[str, float]]]:
        """
        Create mock candidates for a test scenario.

        Returns scored_candidates list for _apply_token_budget_and_deduplication.
        """
        now_ms = int(time.time() * 1000)

        # Create candidates with different confidence and recency levels
        candidates = [
            (
                0.9,  # High score
                Candidate(
                    text="you favorite_color yellow",
                    source="graph",
                    score_hint=0.0,
                    ts=now_ms,  # Very recent
                    meta={"edge_id": "e1", "weight": 0.85, "pos": 3, "neg": 0}
                ),
                {"wsrc": 0.3, "wconf": 0.35, "wrec": 0.25, "wuse": 0.0}
            ),
            (
                0.7,  # Medium score
                Candidate(
                    text="So you don't know, do you know your favorite color?",
                    source="convo",
                    score_hint=0.6,
                    ts=now_ms - 3600000,  # 1 hour ago
                    meta={"bm25_score": 0.6, "turn_id": 5}
                ),
                {"wsrc": 0.4, "wconf": 0.2, "wrec": 0.15, "wpro": 0.075}
            ),
            (
                0.5,  # Low score
                Candidate(
                    text="alice lives in paris",
                    source="graph",
                    score_hint=0.0,
                    ts=now_ms - 86400000 * 7,  # 7 days ago
                    meta={"edge_id": "e2", "weight": 0.45, "pos": 1, "neg": 0}
                ),
                {"wsrc": 0.3, "wconf": 0.15, "wrec": 0.05, "wuse": 0.0}
            ),
        ]

        return candidates

    def test_format(self, format_name: str, scenario: Dict[str, Any]) -> TestResult:
        """
        Test a single format variant for a scenario.

        Args:
            format_name: "technical", "emoji", or "minimal"
            scenario: Test scenario dict

        Returns:
            TestResult with scoring
        """
        # Set environment variable for this test
        if format_name == "technical":
            os.environ["MEMORY_METADATA_FORMAT"] = "technical"
            os.environ["MEMORY_INJECTION_MODE"] = "headers"
        else:
            os.environ["MEMORY_METADATA_FORMAT"] = format_name
            os.environ["MEMORY_INJECTION_MODE"] = "bullets"

        # Create retrieval instance
        host = self.create_mock_host()
        retrieval = Retrieval(host)

        # Create test candidates
        scored_candidates = self.create_test_candidates(scenario)

        # Generate bullet using the format
        final_bullets, _ = retrieval._apply_token_budget_and_deduplication(
            scored_candidates,
            max_bullets=3,
            query=scenario["query"]
        )

        # Take first bullet for analysis
        bullet = final_bullets[0] if final_bullets else ""

        # Check for metadata leaking
        indicators = scenario["metadata_indicators"]
        leaked_indicators = [ind for ind in indicators if ind in bullet]
        metadata_leaked = len(leaked_indicators) > 0

        return TestResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            format=format_name,
            query=scenario["query"],
            bullet=bullet,
            metadata_leaked=metadata_leaked,
            leaked_indicators=leaked_indicators,
            bullet_length=len(bullet),
            timestamp=time.time()
        )

    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """
        Run all test scenarios for all format variants.

        Returns:
            Dict mapping format name to list of TestResults
        """
        formats = ["technical", "emoji", "minimal"]
        results = {fmt: [] for fmt in formats}

        print(f"\n{'='*80}")
        print("A/B TEST: Memory Metadata Formatting")
        print(f"{'='*80}\n")
        print(f"Running {len(self.scenarios)} scenarios × {len(formats)} formats = {len(self.scenarios) * len(formats)} tests\n")

        for scenario in self.scenarios:
            print(f"Scenario {scenario['id']}: {scenario['name']}")
            print(f"  Query: \"{scenario['query']}\"")

            for format_name in formats:
                result = self.test_format(format_name, scenario)
                results[format_name].append(result)

                # Print result
                leaked_status = "❌ LEAKED" if result.metadata_leaked else "✅ CLEAN"
                print(f"    [{format_name:10s}] {leaked_status:12s} | {result.bullet[:60]}...")
                if result.metadata_leaked:
                    print(f"                  Leaked: {result.leaked_indicators}")

            print()

        return results

    def analyze_results(self, results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """
        Analyze test results and generate recommendation.

        Args:
            results: Dict of format -> list of TestResults

        Returns:
            Analysis dict with scores and recommendation
        """
        analysis = {}

        for format_name, test_results in results.items():
            # Count metadata leaks
            leak_count = sum(1 for r in test_results if r.metadata_leaked)
            leak_rate = leak_count / len(test_results) if test_results else 0

            # Average bullet length
            avg_length = sum(r.bullet_length for r in test_results) / len(test_results) if test_results else 0

            # Collect all leaked indicators
            all_leaked = []
            for r in test_results:
                all_leaked.extend(r.leaked_indicators)

            analysis[format_name] = {
                "total_tests": len(test_results),
                "leak_count": leak_count,
                "leak_rate": leak_rate,
                "avg_bullet_length": avg_length,
                "leaked_indicators": list(set(all_leaked)),
                "pass": leak_count == 0  # Must have zero leaks to pass
            }

        # Determine winner
        passing_formats = [fmt for fmt, data in analysis.items() if data["pass"]]

        if not passing_formats:
            recommendation = {
                "winner": None,
                "reason": "All formats failed - metadata leaked in all variants"
            }
        elif len(passing_formats) == 1:
            recommendation = {
                "winner": passing_formats[0],
                "reason": f"Only format with zero metadata leaks"
            }
        else:
            # Multiple passing formats - choose most compact
            winner = min(passing_formats, key=lambda f: analysis[f]["avg_bullet_length"])
            recommendation = {
                "winner": winner,
                "reason": f"Zero leaks + most compact ({analysis[winner]['avg_bullet_length']:.0f} chars avg)"
            }

        analysis["recommendation"] = recommendation

        return analysis

    def print_report(self, analysis: Dict[str, Any]):
        """Print comprehensive test report."""
        print(f"\n{'='*80}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*80}\n")

        for format_name in ["technical", "emoji", "minimal"]:
            data = analysis[format_name]
            status = "✅ PASS" if data["pass"] else "❌ FAIL"

            print(f"{format_name.upper():12s} {status}")
            print(f"  Leak Rate:    {data['leak_rate']*100:5.1f}% ({data['leak_count']}/{data['total_tests']})")
            print(f"  Avg Length:   {data['avg_bullet_length']:5.0f} chars")

            if data["leaked_indicators"]:
                print(f"  Leaked:       {', '.join(data['leaked_indicators'])}")
            print()

        # Print recommendation
        rec = analysis["recommendation"]
        print(f"{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}\n")

        if rec["winner"]:
            print(f"🏆 WINNER: {rec['winner'].upper()}")
            print(f"   Reason: {rec['reason']}")
        else:
            print(f"⚠️  NO WINNER")
            print(f"   Reason: {rec['reason']}")

        print()

    def save_results(self, results: Dict[str, List[TestResult]], analysis: Dict[str, Any], output_path: str = None):
        """Save test results to JSON file."""
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"test_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = {}
        for format_name, test_results in results.items():
            serializable_results[format_name] = [asdict(r) for r in test_results]

        output_data = {
            "results": serializable_results,
            "analysis": analysis,
            "timestamp": time.time()
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"📝 Results saved to: {output_path}\n")


def main():
    """Run the A/B test suite."""
    tester = MetadataFormatTester()

    # Run all tests
    results = tester.run_all_tests()

    # Analyze results
    analysis = tester.analyze_results(results)

    # Print report
    tester.print_report(analysis)

    # Save results
    tester.save_results(results, analysis)


if __name__ == "__main__":
    main()
