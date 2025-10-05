#!/usr/bin/env python3
"""
Intent Classification Replay & Tuning System

Replays exact messages from production logs to test and tune intent classification.
Provides zero-maintenance tuning by analyzing misclassifications and suggesting fixes.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.intent.service import IntentService, get_intent_service
from core.intent.strategies import get_intent_strategies


@dataclass
class TestCase:
    """Individual test case from production logs"""
    id: int
    text: str
    classified_as: str
    confidence: float
    skip_memory: bool
    expected_skip: bool = None  # None means we need to determine this


@dataclass
class TuningResult:
    """Result of intent classification tuning"""
    case: TestCase
    current_classification: Dict[str, Any]
    is_correct: bool
    issue_type: str = ""  # "false_skip", "false_process", "confidence_low", "ok"
    suggested_fix: str = ""


class IntentReplayTuner:
    """
    Zero-maintenance intent classification tuner
    Analyzes production traffic and suggests configuration changes
    """

    def __init__(self, data_file: str = "intent_replay_data.json"):
        self.data_file = data_file
        self.test_cases: List[TestCase] = []
        self.service: IntentService = None
        self.strategies = get_intent_strategies()

        # Domain-specific keywords for better classification
        self.technical_keywords = {
            'payment', 'graph', 'liquidity', 'agent', 'agents', 'blockchain',
            'netting', 'multilateral', 'coordinate', 'settle', 'provider',
            'refund', 'cycle', 'balance', 'transaction', 'protocol'
        }

        self.factual_patterns = {
            'definition', 'is', 'means', 'basically', 'purpose', 'correct',
            'that is', 'the way', 'how it works', 'explanation'
        }

    def load_test_cases(self) -> None:
        """Load test cases from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            self.test_cases = []
            for item in data:
                case = TestCase(
                    id=item['id'],
                    text=item['text'],
                    classified_as=item['classified_as'],
                    confidence=item['confidence'],
                    skip_memory=item['skip_memory'],
                    expected_skip=item.get('expected_skip')
                )

                # Auto-determine expected behavior if not set
                if case.expected_skip is None:
                    case.expected_skip = self._should_skip_memory(case.text)

                self.test_cases.append(case)

            logger.info(f"Loaded {len(self.test_cases)} test cases from {self.data_file}")

        except FileNotFoundError:
            logger.error(f"Test data file {self.data_file} not found")
            raise
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            raise

    def _should_skip_memory(self, text: str) -> bool:
        """
        Determine if text should skip memory processing based on content analysis
        Uses heuristics to classify technical discussions vs casual chat
        """
        text_lower = text.lower().strip()

        # Very short texts or greetings/goodbyes should skip
        if len(text_lower) < 20:
            if any(word in text_lower for word in ['hi', 'hello', 'bye', 'goodbye', 'amazing', 'correct', 'yes', 'no']):
                return True

        # Technical discussions should NOT skip memory
        if any(keyword in text_lower for keyword in self.technical_keywords):
            return False

        # Factual statements should NOT skip memory
        if any(pattern in text_lower for pattern in self.factual_patterns):
            return False

        # Long complex sentences should NOT skip memory
        if len(text) > 80 and (',' in text or 'and' in text):
            return False

        # Default: casual chat, skip memory
        return True

    async def initialize_service(self) -> None:
        """Initialize intent classification service"""
        try:
            self.service = get_intent_service()

            # Ensure service is enabled for testing
            if not self.service.enabled:
                logger.warning("Intent service is disabled - enabling for testing")
                self.service.enabled = True
                self.service._initialize_components()

            logger.info("Intent service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize intent service: {e}")
            raise

    async def run_replay_analysis(self) -> List[TuningResult]:
        """Run replay analysis on all test cases"""
        if not self.service:
            await self.initialize_service()

        results: List[TuningResult] = []

        logger.info("Starting intent classification replay analysis...")

        for case in self.test_cases:
            try:
                # Get current classification
                current = await self.service.classify_intent(case.text)

                # Analyze the result
                result = self._analyze_classification(case, current)
                results.append(result)

                # Log the result
                status = "✅ OK" if result.is_correct else f"❌ {result.issue_type.upper()}"
                logger.info(f"Case {case.id:2d}: {status} - '{case.text[:50]}...'")

                if not result.is_correct:
                    logger.info(f"         Issue: {result.suggested_fix}")

            except Exception as e:
                logger.error(f"Failed to analyze case {case.id}: {e}")
                continue

        return results

    def _analyze_classification(self, case: TestCase, current: Dict[str, Any]) -> TuningResult:
        """Analyze a single classification result"""
        result = TuningResult(
            case=case,
            current_classification=current,
            is_correct=True  # Assume correct until proven otherwise
        )

        current_intent = current['intent']
        current_skip = current['skip_memory']
        expected_skip = case.expected_skip

        # Check if memory processing decision is correct
        if current_skip != expected_skip:
            result.is_correct = False

            if current_skip and not expected_skip:
                # Should process memory but skipping
                result.issue_type = "false_skip"
                result.suggested_fix = f"Technical content classified as '{current_intent}' should not skip memory"

            elif not current_skip and expected_skip:
                # Processing memory but should skip
                result.issue_type = "false_process"
                result.suggested_fix = f"Casual content classified as '{current_intent}' should skip memory"
        else:
            result.issue_type = "ok"
            result.suggested_fix = f"Classification '{current_intent}' → skip={current_skip} is correct"

        return result

    def generate_tuning_report(self, results: List[TuningResult]) -> Dict[str, Any]:
        """Generate comprehensive tuning report with actionable recommendations"""
        total_cases = len(results)
        correct_cases = sum(1 for r in results if r.is_correct)
        accuracy = (correct_cases / total_cases) * 100 if total_cases > 0 else 0

        # Categorize issues
        issue_counts = {}
        false_skips = []
        false_processes = []

        for result in results:
            if not result.is_correct:
                issue_type = result.issue_type
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

                if issue_type == "false_skip":
                    false_skips.append(result)
                elif issue_type == "false_process":
                    false_processes.append(result)

        # Generate specific recommendations
        recommendations = []

        if false_skips:
            recommendations.append({
                "type": "model_tuning",
                "priority": "high",
                "issue": f"{len(false_skips)} technical discussions incorrectly classified as casual chat",
                "suggestion": "Add technical domain keywords or retrain with domain-specific examples",
                "examples": [r.case.text[:60] + "..." for r in false_skips[:3]]
            })

        if false_processes:
            recommendations.append({
                "type": "strategy_adjustment",
                "priority": "medium",
                "issue": f"{len(false_processes)} casual messages incorrectly processed for memory",
                "suggestion": "Adjust intent strategies or confidence thresholds",
                "examples": [r.case.text[:60] + "..." for r in false_processes[:3]]
            })

        # Configuration suggestions
        config_suggestions = []

        if len(false_skips) > len(false_processes):
            config_suggestions.append(
                "Consider lowering INTENT_CONFIDENCE_THRESHOLD to catch more technical discussions"
            )

        if accuracy < 80:
            config_suggestions.append(
                "Consider switching to a domain-specific intent classification model"
            )

        return {
            "summary": {
                "total_cases": total_cases,
                "correct_cases": correct_cases,
                "accuracy_percent": round(accuracy, 1),
                "major_issue": "false_skip" if len(false_skips) > 2 else "acceptable"
            },
            "issue_breakdown": issue_counts,
            "recommendations": recommendations,
            "config_suggestions": config_suggestions,
            "detailed_results": [
                {
                    "case_id": r.case.id,
                    "text": r.case.text,
                    "expected_skip": r.case.expected_skip,
                    "actual_skip": r.current_classification['skip_memory'],
                    "intent": r.current_classification['intent'],
                    "confidence": r.current_classification['confidence'],
                    "issue": r.issue_type,
                    "suggestion": r.suggested_fix
                }
                for r in results
            ]
        }

    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        logger.info("Starting Intent Classification Replay & Tuning Analysis")
        logger.info("=" * 60)

        # Load test cases
        self.load_test_cases()

        # Initialize service
        await self.initialize_service()

        # Run analysis
        results = await self.run_replay_analysis()

        # Generate report
        report = self.generate_tuning_report(results)

        # Print summary
        self._print_summary_report(report)

        # Save detailed report
        report_file = "intent_classification_tuning_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Detailed report saved to {report_file}")

        return report

    def _print_summary_report(self, report: Dict[str, Any]) -> None:
        """Print human-readable summary report"""
        summary = report['summary']

        print(f"\n📊 INTENT CLASSIFICATION ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Total Cases: {summary['total_cases']}")
        print(f"Accuracy: {summary['accuracy_percent']}%")
        print(f"Status: {'🚨 NEEDS ATTENTION' if summary['major_issue'] != 'acceptable' else '✅ ACCEPTABLE'}")

        if report['recommendations']:
            print(f"\n🔧 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. [{rec['priority'].upper()}] {rec['issue']}")
                print(f"   💡 {rec['suggestion']}")

        if report['config_suggestions']:
            print(f"\n⚙️  CONFIGURATION CHANGES:")
            for i, suggestion in enumerate(report['config_suggestions'], 1):
                print(f"{i}. {suggestion}")

        print(f"\n📝 See intent_classification_tuning_report.json for detailed analysis")

    async def apply_automatic_fixes(self, report: Dict[str, Any]) -> None:
        """Apply automatic fixes where possible (zero-maintenance approach)"""
        logger.info("Applying automatic fixes...")

        # This would implement automatic configuration adjustments
        # For now, just provide the recommendations

        fixes_applied = []

        for rec in report['recommendations']:
            if rec['type'] == 'strategy_adjustment' and rec['priority'] == 'high':
                # Could automatically adjust confidence thresholds
                fixes_applied.append(f"Would adjust confidence threshold for {rec['issue']}")

        if fixes_applied:
            logger.info("Automatic fixes that could be applied:")
            for fix in fixes_applied:
                logger.info(f"  - {fix}")
        else:
            logger.info("No automatic fixes available - manual tuning required")


async def main():
    """Main entry point"""
    tuner = IntentReplayTuner()

    try:
        report = await tuner.run_full_analysis()

        # Optionally apply automatic fixes
        # await tuner.apply_automatic_fixes(report)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())