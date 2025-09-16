#!/usr/bin/env python3
"""
Analyze memory operation timing using the built-in tracer
"""

import os
import sys
import time
from loguru import logger

# Add server to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Main function to analyze memory timing"""
    from components.memory.memory_timing_tracer import get_memory_tracer

    # Get the global tracer instance
    tracer = get_memory_tracer()

    # Print current report
    logger.info("=" * 80)
    logger.info("MEMORY TIMING ANALYSIS")
    logger.info("=" * 80)

    # Print the timing report
    tracer.print_report()

    # Also get problematic operations
    problematic = tracer.get_problematic_operations(percentile=90)

    if problematic:
        logger.warning("\n" + "=" * 80)
        logger.warning("⚠️ PROBLEMATIC OPERATIONS (>90th percentile)")
        logger.warning("=" * 80)
        for op in problematic:
            logger.warning(f"  {op.component}.{op.operation}: {op.duration_ms:.0f}ms")
            if op.details:
                logger.warning(f"    Details: {op.details}")

    # Get optimization recommendations
    recommendations = tracer.get_optimization_recommendations()

    if recommendations:
        logger.info("\n" + "=" * 80)
        logger.info("💡 OPTIMIZATION RECOMMENDATIONS")
        logger.info("=" * 80)
        for rec in recommendations:
            logger.info(f"  • {rec}")

if __name__ == "__main__":
    main()