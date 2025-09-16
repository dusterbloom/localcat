#!/usr/bin/env python3
"""
Test memory operation timing metrics
"""

import os
import sys
import time

# Add server path and activate environment
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')

def test_memory_timing():
    """Test the memory timing tracer with actual operations"""
    print("🗄️ Testing Memory Operation Timing Metrics")
    print("=" * 60)

    from components.memory.hotmemory_facade import HotMemoryFacade
    from components.memory.memory_store import MemoryStore, Paths
    from components.session.session_store import SessionStore
    from components.memory.memory_timing_tracer import get_memory_tracer
    import tempfile

    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize components
        paths = Paths(
            sqlite_path=os.path.join(temp_dir, "test_memory.db"),
            lmdb_dir=os.path.join(temp_dir, "test_graph.lmdb")
        )
        store = MemoryStore(paths)
        session_store = SessionStore(db_path=os.path.join(temp_dir, "test_sessions.db"))

        # Initialize facade with timing
        facade = HotMemoryFacade(store)
        tracer = get_memory_tracer()

        print("✅ Components initialized")

        # Test operations with timing
        test_cases = [
            ("Hello there!", "greeting"),
            ("My dog's name is Max", "fact_storage"),
            ("What is the weather like?", "retrieval"),
            ("Actually, his name is Buddy", "correction"),
            ("I went to Paris in 2020 and stayed for three months", "complex_fact"),
        ]

        print(f"\n🧪 Testing {len(test_cases)} memory operations:")

        for i, (text, test_type) in enumerate(test_cases, 1):
            session_id = f"test_session_{i}"
            turn_id = 1

            print(f"\n{i}. {test_type}: '{text[:40]}...'")

            # Process the turn
            start_time = time.perf_counter()
            result = facade.process_turn(text, session_id, turn_id)
            total_time = (time.perf_counter() - start_time) * 1000

            print(f"   Total: {total_time:.1f}ms | Bullets: {len(result.bullets)} | Stored: {len(result.triples)}")

            # Store assistant response
            facade.store_assistant_response(session_id, f"Response to: {text}", turn_id)

        # Print detailed timing report
        print(f"\n📊 MEMORY TIMING REPORT:")
        print("=" * 60)
        tracer.print_report(top_n=20)

        # Export timing data
        export_path = "/tmp/memory_timing_report.json"
        tracer.export_measurements(export_path)

        # Analyze performance vs backlog targets
        print(f"\n🎯 PERFORMANCE ANALYSIS:")
        summary = tracer.get_operation_summary()

        print(f"\nOperation Breakdown:")
        for op, data in summary["operation_breakdown"].items():
            target = get_operation_target(op)
            status = "✅" if data["avg_p95_ms"] <= target else "⚠️" if data["avg_p95_ms"] <= target * 2 else "🔥"
            print(f"   {status} {op:20} {data['avg_p95_ms']:6.1f}ms (target: {target}ms, {data['total_calls']} calls)")

        print(f"\nComponent Breakdown:")
        for comp, data in summary["component_breakdown"].items():
            print(f"   {comp:15} {data['avg_p95_ms']:6.1f}ms ({data['total_calls']} calls, {data['operations']} ops)")

        if summary["slow_operations"]:
            print(f"\n⚠️ Operations exceeding targets:")
            for slow in summary["slow_operations"]:
                print(f"   🔥 {slow['operation']:30} {slow['p95_ms']:6.1f}ms ({slow['slowdown_factor']:.1f}x slower)")

        # Recommendations from backlog
        if summary["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                print(f"   {rec}")

        print(f"\n📁 Detailed timing data exported to: {export_path}")

def get_operation_target(operation: str) -> int:
    """Get performance targets based on backlog requirements"""
    targets = {
        'session_write': 25,
        'assistant_write': 25,
        'fts_index': 30,
        'memory_retrieve': 100,
        'memory_write_batch': 50,
        'memory_flush': 200,
        'session_link': 20,
    }
    return targets.get(operation, 100)  # Default 100ms

def analyze_bottlenecks():
    """Analyze timing patterns for bottleneck identification"""
    from components.memory.memory_timing_tracer import get_memory_tracer

    tracer = get_memory_tracer()
    stats = tracer.get_stats()

    print(f"\n🔍 BOTTLENECK ANALYSIS:")
    print("=" * 40)

    bottlenecks = []
    for key, stat in stats.items():
        target = get_operation_target(stat.operation)
        if stat.p95_ms > target * 1.5:
            bottlenecks.append({
                'operation': key,
                'p95_ms': stat.p95_ms,
                'target_ms': target,
                'slowdown': stat.p95_ms / target,
                'frequency': stat.count
            })

    if bottlenecks:
        bottlenecks.sort(key=lambda x: x['slowdown'], reverse=True)
        print("Top bottlenecks (worst first):")
        for i, b in enumerate(bottlenecks[:5], 1):
            print(f"{i}. {b['operation']:30} {b['p95_ms']:6.1f}ms ({b['slowdown']:.1f}x slower, {b['frequency']} calls)")
    else:
        print("✅ No significant bottlenecks detected!")

if __name__ == "__main__":
    test_memory_timing()
    analyze_bottlenecks()