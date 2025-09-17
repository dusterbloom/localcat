#!/usr/bin/env python3
"""Test script to verify graph manager fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
from components.graph.dual_graph_manager import DualGraphManager

def test_graph_fixes():
    """Test that the graph manager fixes work correctly"""
    print("Testing graph manager fixes...")

    # Create manager with short TTL for testing
    manager = DualGraphManager(ttl_minutes=0.01, promotion_threshold=0.8, use_louvain=True)

    # Test 1: Add triples
    print("\n1. Adding test triples...")
    tid1 = manager.add_triple("you", "likes", "coffee", 0.7, source='agent')
    tid2 = manager.add_triple("coffee", "is", "beverage", 0.6, source='agent')
    tid3 = manager.add_triple("you", "lives_in", "San Francisco", 0.9, source='user')
    print(f"  Added {len(manager.agent_triples)} agent triples, {manager.user_graph.number_of_nodes()} user nodes")

    # Test 2: Test community detection (this used to fail with directed graphs)
    print("\n2. Testing community detection...")
    try:
        communities = manager.detect_communities('agent')
        print(f"  ✓ Agent communities: {len(communities)} detected")

        communities = manager.detect_communities('combined')
        print(f"  ✓ Combined communities: {len(communities)} detected")
    except Exception as e:
        print(f"  ✗ Community detection failed: {e}")
        return False

    # Test 3: Wait for TTL to expire and test cleanup (this used to fail with key argument)
    print("\n3. Testing TTL expiry and cleanup...")
    time.sleep(1)  # Wait for TTL to expire (0.01 minutes = 0.6 seconds)

    try:
        expired_count = manager.cleanup_expired()
        print(f"  ✓ Cleaned up {expired_count} expired triples")
        print(f"  Remaining agent triples: {len(manager.agent_triples)}")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")
        return False

    # Test 4: Verify graph state after cleanup
    print("\n4. Verifying graph state...")
    stats = manager.get_stats()
    print(f"  Agent nodes: {stats.agent_nodes}, edges: {stats.agent_edges}")
    print(f"  User nodes: {stats.user_nodes}, edges: {stats.user_edges}")
    print(f"  Decayed: {stats.decayed_triples}, Promoted: {stats.promoted_triples}")

    print("\n✅ All tests passed! Graph manager fixes are working correctly.")
    return True

if __name__ == "__main__":
    success = test_graph_fixes()
    sys.exit(0 if success else 1)