#!/usr/bin/env python3
"""
HotMem V4 Comprehensive Validation Test Runner

Runs all feature activation tests and generates comprehensive reports.
This validates the evidence-based improvements to the 26x speed breakthrough.
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from test_feature_impact import FeatureActivationTester
from baseline_performance import BaselinePerformanceMeasurer

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: str
    baseline_performance: Dict[str, Any]
    feature_comparisons: List[Dict[str, Any]]
    overall_recommendations: List[str]
    performance_summary: Dict[str, Any]
    success_criteria_met: Dict[str, bool]

class ComprehensiveValidator:
    """Runs complete HotMem V4 validation suite"""
    
    def __init__(self):
        self.results = {}
        
    async def run_complete_validation(self) -> ValidationReport:
        """Run comprehensive validation of all HotMem V4 improvements"""
        
        logger.info("🚀 Starting HotMem V4 Comprehensive Validation")
        logger.info("=" * 70)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Step 1: Establish baseline performance
        logger.info("📊 Step 1: Measuring baseline performance...")
        measurer = BaselinePerformanceMeasurer()
        baseline = await measurer.generate_baseline_report()
        baseline_dict = asdict(baseline)
        
        # Step 2: Run feature activation A/B tests
        logger.info("🧪 Step 2: Running feature activation A/B tests...")
        tester = FeatureActivationTester()
        
        # Test all features
        coref_baseline, coref_enhanced = await tester.test_coref_activation_impact()
        coref_comparison = tester.compare_results(coref_baseline, coref_enhanced)
        
        leann_baseline, leann_enhanced = await tester.test_leann_semantic_activation()
        leann_comparison = tester.compare_results(leann_baseline, leann_enhanced)
        
        decomp_baseline, decomp_enhanced = await tester.test_decomposition_activation()
        decomp_comparison = tester.compare_results(decomp_baseline, decomp_enhanced)
        
        dspy_result = await tester.test_dspy_integration_readiness()
        
        feature_comparisons = [coref_comparison, leann_comparison, decomp_comparison]
        
        # Step 3: Performance analysis
        logger.info("📈 Step 3: Analyzing performance improvements...")
        performance_summary = self._analyze_performance_improvements(
            baseline_dict, feature_comparisons, dspy_result
        )
        
        # Step 4: Generate recommendations
        logger.info("💡 Step 4: Generating recommendations...")
        recommendations = self._generate_recommendations(
            baseline_dict, feature_comparisons, dspy_result, performance_summary
        )
        
        # Step 5: Success criteria validation
        logger.info("🏆 Step 5: Validating success criteria...")
        success_criteria = self._validate_success_criteria(
            baseline_dict, feature_comparisons, performance_summary
        )
        
        # Generate comprehensive report
        report = ValidationReport(
            timestamp=timestamp,
            baseline_performance=baseline_dict,
            feature_comparisons=feature_comparisons,
            overall_recommendations=recommendations,
            performance_summary=performance_summary,
            success_criteria_met=success_criteria
        )
        
        # Print final report
        self._print_final_report(report)
        
        return report
    
    def _analyze_performance_improvements(self, baseline: Dict[str, Any], 
                                        comparisons: List[Dict[str, Any]], 
                                        dspy_result: Any) -> Dict[str, Any]:
        """Analyze overall performance improvements"""
        
        total_accuracy_improvement = sum(c.get('accuracy_improvement', 0) for c in comparisons)
        total_latency_impact = sum(c.get('latency_delta_ms', 0) for c in comparisons)
        
        features_to_enable = [c['feature'] for c in comparisons if c['recommendation'] == 'ENABLE']
        
        return {
            'total_accuracy_improvement': total_accuracy_improvement,
            'total_latency_impact_ms': total_latency_impact,
            'features_to_enable': features_to_enable,
            'baseline_classifier_ms': baseline.get('classifier_ms_avg', 0),
            'baseline_pipeline_ms': baseline.get('pipeline_ms_avg', 0),
            'projected_accuracy': baseline.get('extraction_accuracy', 0) + total_accuracy_improvement,
            'projected_latency_ms': baseline.get('pipeline_ms_avg', 0) + total_latency_impact,
            'dspy_ready': getattr(dspy_result, 'accuracy_score', 0) > 0,
            'speed_advantage_maintained': total_latency_impact < 50  # Keep under 50ms overhead
        }
    
    def _generate_recommendations(self, baseline: Dict[str, Any], 
                                comparisons: List[Dict[str, Any]], 
                                dspy_result: Any,
                                performance: Dict[str, Any]) -> List[str]:
        """Generate evidence-based recommendations"""
        
        recommendations = []
        
        # Feature activation recommendations
        for comparison in comparisons:
            if comparison['recommendation'] == 'ENABLE':
                feature = comparison['feature']
                accuracy_gain = comparison['accuracy_improvement']
                latency_cost = comparison['latency_delta_ms']
                
                if latency_cost < 25:  # Low latency cost
                    recommendations.append(
                        f"✅ ENABLE {feature}: +{accuracy_gain:.1%} accuracy with minimal latency impact (+{latency_cost:.1f}ms)"
                    )
                elif accuracy_gain > 0.10:  # High accuracy gain
                    recommendations.append(
                        f"⚖️ CONSIDER {feature}: +{accuracy_gain:.1%} accuracy but +{latency_cost:.1f}ms latency cost"
                    )
                else:
                    recommendations.append(
                        f"⚠️ OPTIONAL {feature}: Limited benefit (+{accuracy_gain:.1%} accuracy, +{latency_cost:.1f}ms)"
                    )
        
        # Performance recommendations
        if performance['speed_advantage_maintained']:
            recommendations.append("🚀 26x speed advantage maintained with feature improvements")
        else:
            recommendations.append("⚠️ Feature activation may impact 26x speed advantage - review latency budget")
        
        # DSPy integration
        if performance['dspy_ready']:
            recommendations.append("🤖 DSPy framework ready for production integration")
        else:
            recommendations.append("🔧 DSPy framework needs configuration before production use")
        
        # Configuration recommendations
        optimal_config = []
        for comparison in comparisons:
            if comparison['recommendation'] == 'ENABLE':
                feature = comparison['feature']
                if feature == 'COREF':
                    optimal_config.append("HOTMEM_USE_COREF=true")
                elif feature == 'LEANN':
                    optimal_config.append("HOTMEM_USE_LEANN=true") 
                elif feature == 'DECOMP':
                    optimal_config.append("HOTMEM_DECOMPOSE_CLAUSES=true")
        
        if performance['dspy_ready']:
            optimal_config.append("HOTMEM_USE_DSPY=true")
        
        if optimal_config:
            config_str = " ".join(optimal_config)
            recommendations.append(f"🔧 Optimal configuration: {config_str}")
        
        return recommendations
    
    def _validate_success_criteria(self, baseline: Dict[str, Any], 
                                 comparisons: List[Dict[str, Any]], 
                                 performance: Dict[str, Any]) -> Dict[str, bool]:
        """Validate against HotMem V4 success criteria"""
        
        criteria = {
            'maintain_54ms_classifier': baseline.get('classifier_ms_avg', 1000) <= 60,  # Allow 10% margin
            'overall_accuracy_80_percent': performance.get('projected_accuracy', 0) >= 0.80,
            'pipeline_under_100ms': performance.get('projected_latency_ms', 1000) <= 100,
            'feature_activation_successful': len(performance.get('features_to_enable', [])) >= 2,
            'dspy_integration_ready': performance.get('dspy_ready', False),
            'speed_advantage_preserved': performance.get('speed_advantage_maintained', False)
        }
        
        return criteria
    
    def _print_final_report(self, report: ValidationReport) -> None:
        """Print comprehensive final report"""
        
        logger.info("🏆 HOTMEM V4 COMPREHENSIVE VALIDATION REPORT")
        logger.info("=" * 70)
        logger.info(f"📅 Timestamp: {report.timestamp}")
        logger.info("")
        
        # Baseline performance
        baseline = report.baseline_performance
        logger.info("📊 BASELINE PERFORMANCE (26x Speed Breakthrough):")
        logger.info(f"   🚀 Classifier: {baseline.get('classifier_ms_avg', 0):.1f}ms avg (Target: 54ms)")
        logger.info(f"   ⚡ Pipeline: {baseline.get('pipeline_ms_avg', 0):.1f}ms avg (Target: <200ms)")
        logger.info(f"   🎯 Accuracy: {baseline.get('extraction_accuracy', 0):.1%} overall")
        logger.info("")
        
        # Feature improvements
        logger.info("🧪 FEATURE ACTIVATION RESULTS:")
        for comparison in report.feature_comparisons:
            feature = comparison['feature']
            recommendation = comparison['recommendation']
            accuracy = comparison['accuracy_improvement']
            latency = comparison['latency_delta_ms']
            
            status = "✅" if recommendation == "ENABLE" else "❌"
            logger.info(f"   {status} {feature}: {recommendation}")
            logger.info(f"      Accuracy: {accuracy:+.1%}, Latency: {latency:+.1f}ms")
        logger.info("")
        
        # Success criteria
        logger.info("🏆 SUCCESS CRITERIA VALIDATION:")
        for criterion, met in report.success_criteria_met.items():
            status = "✅" if met else "❌"
            logger.info(f"   {status} {criterion.replace('_', ' ').title()}: {'PASSED' if met else 'FAILED'}")
        logger.info("")
        
        # Recommendations
        logger.info("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.overall_recommendations, 1):
            logger.info(f"   {i}. {rec}")
        logger.info("")
        
        # Summary
        performance = report.performance_summary
        total_features_enabled = len(performance.get('features_to_enable', []))
        projected_accuracy = performance.get('projected_accuracy', 0)
        projected_latency = performance.get('projected_latency_ms', 0)
        
        logger.info("📈 OVERALL IMPACT SUMMARY:")
        logger.info(f"   🎯 Features to Enable: {total_features_enabled}/3")
        logger.info(f"   📊 Projected Accuracy: {projected_accuracy:.1%}")
        logger.info(f"   ⚡ Projected Latency: {projected_latency:.1f}ms") 
        logger.info(f"   🚀 Speed Advantage: {'MAINTAINED' if performance.get('speed_advantage_maintained') else 'AT RISK'}")
        logger.info("")
        
        # Final verdict
        criteria_passed = sum(1 for passed in report.success_criteria_met.values() if passed)
        total_criteria = len(report.success_criteria_met)
        
        if criteria_passed >= total_criteria * 0.8:  # 80% success rate
            logger.info("🎉 HOTMEM V4 VALIDATION: SUCCESS!")
            logger.info("   Evidence-based improvements ready for production activation.")
        else:
            logger.info("⚠️  HOTMEM V4 VALIDATION: NEEDS WORK")
            logger.info(f"   {criteria_passed}/{total_criteria} success criteria met.")
        
        logger.info("=" * 70)

async def main():
    """Run comprehensive HotMem V4 validation"""
    
    validator = ComprehensiveValidator()
    report = await validator.run_complete_validation()
    
    # Save report to file
    report_file = Path(__file__).parent / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    logger.info(f"📄 Full report saved to: {report_file}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())