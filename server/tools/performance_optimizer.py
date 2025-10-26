#!/usr/bin/env python3
"""
PERFORMANCE OPTIMIZER - Fix critical latency bottlenecks
Addresses SOTA classifier and extraction pipeline performance
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class PerformanceOptimizer:
    """Optimize pipeline performance with data-driven recommendations"""

    def __init__(self):
        self.recommendations = []
        self.config_changes = {}

    def analyze_sota_classifier_performance(self) -> Dict[str, Any]:
        """Analyze SOTA classifier performance and recommend fixes"""

        # Based on benchmarks:
        # CPU: 1,697ms avg (248x slower)
        # MPS: 451ms avg (66x slower)

        analysis = {
            "critical_issue": True,
            "cpu_latency_ms": 1697,
            "mps_latency_ms": 451,
            "rule_based_latency_ms": 6.85,
            "slowdown_factor_cpu": 248,
            "slowdown_factor_mps": 66,
            "net_impact_ms": 1655,
            "recommendations": []
        }

        # Generate specific recommendations
        if analysis["mps_latency_ms"] > 100:
            analysis["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Disable SOTA classifier entirely",
                "reason": f"Even on MPS, adds {analysis['mps_latency_ms']}ms per turn",
                "env_var": "HOTMEM_USE_SOTA_CLASSIFIER=false"
            })

        if analysis["cpu_latency_ms"] > 1000:
            analysis["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Force MPS device for SOTA if enabled",
                "reason": f"CPU is {analysis['slowdown_factor_cpu']}x slower than rule-based",
                "config": "device='mps' in classifier init"
            })

        analysis["recommendations"].append({
            "priority": "HIGH",
            "action": "Implement classifier caching",
            "reason": "Cache results for repeated/similar phrases",
            "implementation": "LRU cache with text similarity matching"
        })

        analysis["recommendations"].append({
            "priority": "MEDIUM",
            "action": "Use streaming/async classification",
            "reason": "Don't block pipeline while classifying",
            "implementation": "Background classification with fallback to rule-based"
        })

        return analysis

    def analyze_memory_extraction_performance(self) -> Dict[str, Any]:
        """Analyze memory extraction bottlenecks"""

        # Based on logs: p95=1824ms, mean=942ms for extraction
        analysis = {
            "current_p95_ms": 1824,
            "current_mean_ms": 942,
            "target_ms": 200,
            "slowdown_factor": 9.1,  # 1824/200
            "recommendations": []
        }

        analysis["recommendations"].extend([
            {
                "priority": "HIGH",
                "action": "Optimize spaCy model loading",
                "reason": "Model loading likely causing delays",
                "implementation": "Pre-load and cache models, use smaller models"
            },
            {
                "priority": "HIGH",
                "action": "Reduce extraction complexity",
                "reason": "Complex NLP processing taking too long",
                "config": "ENHANCED_LEVEL3_COMPLEXITY=low, disable heavy features"
            },
            {
                "priority": "MEDIUM",
                "action": "Implement extraction timeouts",
                "reason": "Prevent extraction from blocking pipeline",
                "implementation": "Timeout after 200ms, fallback to simple extraction"
            },
            {
                "priority": "MEDIUM",
                "action": "Use async/background extraction",
                "reason": "Don't block response generation",
                "implementation": "Extract in background, use for next turn"
            }
        ])

        return analysis

    def analyze_llm_performance(self, model_name: str, observed_latency_ms: float) -> Dict[str, Any]:
        """Analyze LLM performance and provide recommendations"""

        analysis = {
            "model": model_name,
            "observed_latency_ms": observed_latency_ms,
            "target_latency_ms": 250,  # Target from logs
            "slowdown_factor": observed_latency_ms / 250,
            "recommendations": []
        }

        # Model-specific analysis
        if "vl" in model_name.lower() or "vision" in model_name.lower():
            analysis["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Switch to text-only model for voice interactions",
                "reason": f"Vision model adds significant overhead: {observed_latency_ms:.1f}ms",
                "alternative": "llama-3.2-1b-instruct"
            })

        if "4b" in model_name:
            analysis["recommendations"].append({
                "priority": "HIGH",
                "action": "Consider smaller model for better latency",
                "reason": f"4B model too slow for real-time: {observed_latency_ms:.1f}ms",
                "alternative": "llama-3.2-1b-instruct or gemma-3n-e4b"
            })

        if observed_latency_ms > 10000:  # >10 seconds
            analysis["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Model initialization or loading issue detected",
                "reason": f"Latency suggests model reloading: {observed_latency_ms:.1f}ms",
                "fix": "Check model caching, pre-loading, and initialization"
            })

        return analysis

    def generate_immediate_fixes(self) -> Dict[str, str]:
        """Generate immediate environment variable fixes"""

        fixes = {
            # Disable SOTA classifier entirely
            "HOTMEM_USE_SOTA_CLASSIFIER": "false",

            # Optimize memory extraction
            "ENHANCED_LEVEL3_COMPLEXITY": "low",
            "HOTMEM_EXTRACTION_TIMEOUT": "200",  # 200ms timeout
            "HOTMEM_USE_SIMPLE_EXTRACTION": "true",

            # Reduce retrieval overhead
            "HOTMEM_MAX_BULLETS": "3",  # Fewer bullets = faster
            "HOTMEM_RETRIEVAL_TIMEOUT": "50",   # 50ms timeout

            # Optimize session processing
            "HOTMEM_SESSION_CONTEXT": "false",  # Disable if not critical
            "HOTMEM_MIN_EDGE_CONFIDENCE": "0.8",  # Higher threshold = fewer edges

            # Performance monitoring
            "HOTMEM_ENABLE_METRICS": "true",
            "HOTMEM_LOG_PERFORMANCE": "true"
        }

        return fixes

    def generate_performance_config(self) -> str:
        """Generate optimized .env configuration"""

        config_lines = [
            "# =================================================================",
            "# PERFORMANCE OPTIMIZED CONFIGURATION",
            "# Generated by Performance Optimizer",
            f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "# =================================================================",
            "",
            "# CRITICAL: Disable SOTA classifier (1655ms latency impact)",
            "HOTMEM_USE_SOTA_CLASSIFIER=false",
            "",
            "# Memory extraction optimizations",
            "ENHANCED_LEVEL3_COMPLEXITY=low",
            "HOTMEM_EXTRACTION_TIMEOUT=200",
            "HOTMEM_USE_SIMPLE_EXTRACTION=true",
            "",
            "# Retrieval optimizations",
            "HOTMEM_MAX_BULLETS=3",
            "HOTMEM_RETRIEVAL_TIMEOUT=50",
            "HOTMEM_MIN_EDGE_CONFIDENCE=0.8",
            "",
            "# Session optimizations",
            "HOTMEM_SESSION_CONTEXT=false",
            "",
            "# Performance monitoring",
            "HOTMEM_ENABLE_METRICS=true",
            "HOTMEM_LOG_PERFORMANCE=true",
            "",
            "# Target latency budgets (ms)",
            "HOTMEM_TARGET_TOTAL_MS=200",
            "HOTMEM_TARGET_EXTRACTION_MS=100",
            "HOTMEM_TARGET_RETRIEVAL_MS=50",
            "HOTMEM_TARGET_INTENT_MS=20",
            ""
        ]

        return "\n".join(config_lines)

    def print_performance_report(self):
        """Print comprehensive performance analysis"""

        print("\n" + "="*80)
        print("🚀 LOCALCAT PIPELINE PERFORMANCE ANALYSIS")
        print("="*80)

        # SOTA classifier analysis
        sota_analysis = self.analyze_sota_classifier_performance()
        print(f"\n🔥 CRITICAL: SOTA CLASSIFIER PERFORMANCE")
        print(f"   CPU Latency: {sota_analysis['cpu_latency_ms']}ms ({sota_analysis['slowdown_factor_cpu']}x slower)")
        print(f"   MPS Latency: {sota_analysis['mps_latency_ms']}ms ({sota_analysis['slowdown_factor_mps']}x slower)")
        print(f"   Net Impact: +{sota_analysis['net_impact_ms']}ms per turn")
        print(f"   Status: {'🚨 PIPELINE KILLER' if sota_analysis['critical_issue'] else '⚠️ Needs optimization'}")

        print(f"\n💡 SOTA CLASSIFIER RECOMMENDATIONS:")
        for rec in sota_analysis['recommendations']:
            priority_emoji = {"CRITICAL": "🔥", "HIGH": "⚠️", "MEDIUM": "💡"}[rec['priority']]
            print(f"   {priority_emoji} {rec['priority']}: {rec['action']}")
            print(f"      Reason: {rec['reason']}")
            if 'env_var' in rec:
                print(f"      Fix: {rec['env_var']}")
            if 'implementation' in rec:
                print(f"      Implementation: {rec['implementation']}")
            print()

        # Memory extraction analysis
        extraction_analysis = self.analyze_memory_extraction_performance()
        print(f"\n⚠️ MEMORY EXTRACTION PERFORMANCE")
        print(f"   Current p95: {extraction_analysis['current_p95_ms']}ms")
        print(f"   Current mean: {extraction_analysis['current_mean_ms']}ms")
        print(f"   Target: {extraction_analysis['target_ms']}ms")
        print(f"   Slowdown: {extraction_analysis['slowdown_factor']:.1f}x")

        print(f"\n💡 EXTRACTION RECOMMENDATIONS:")
        for rec in extraction_analysis['recommendations']:
            priority_emoji = {"CRITICAL": "🔥", "HIGH": "⚠️", "MEDIUM": "💡"}[rec['priority']]
            print(f"   {priority_emoji} {rec['priority']}: {rec['action']}")
            print(f"      Reason: {rec['reason']}")
            if 'implementation' in rec:
                print(f"      Implementation: {rec['implementation']}")
            print()

        # Immediate fixes
        print(f"\n🛠️ IMMEDIATE FIXES:")
        fixes = self.generate_immediate_fixes()
        for key, value in fixes.items():
            print(f"   export {key}={value}")

        print(f"\n🎯 EXPECTED IMPROVEMENTS:")
        print(f"   SOTA Classifier: -1655ms per turn (disabled)")
        print(f"   Memory Extraction: ~-800ms per turn (optimizations)")
        print(f"   Total Savings: ~2455ms per turn")
        print(f"   New Target: <500ms total HotMem latency")

        print("="*80)

def main():
    """Run performance analysis and generate fixes"""
    optimizer = PerformanceOptimizer()

    # Print analysis
    optimizer.print_performance_report()

    # Generate optimized config
    config = optimizer.generate_performance_config()

    # Write config file
    config_path = "/Users/peppi/Dev/localcat/server/.env.performance_optimized"
    with open(config_path, 'w') as f:
        f.write(config)

    print(f"\n📁 Generated optimized config: {config_path}")
    print(f"💡 To apply: cp {config_path} .env.extraction_testing")
    print(f"🔄 Then restart the server to apply changes")

if __name__ == "__main__":
    main()