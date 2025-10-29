#!/usr/bin/env python3
"""
Verification script to test if the memory retrieval fix is working

Usage:
    python tests/diagnostics/verify_fix.py
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add server root to path
_HERE = Path(__file__).parent
_SERVER_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_SERVER_ROOT))

def verify_fix():
    """Verify that memory retrieval is now working"""

    print("="*70)
    print("MEMORY RETRIEVAL FIX VERIFICATION")
    print("="*70)

    # Check configuration
    user_id = os.getenv('USER_ID', 'NOT_SET')
    print(f"\n[1] Current USER_ID: {user_id}")

    if user_id == 'NOT_SET':
        print("⚠️  USER_ID not set in environment")
        print("   Make sure to source .env or set USER_ID manually")
        return False

    # Connect to database
    db_path = Path(_SERVER_ROOT.parent / "data" / "memory.db")
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        return False

    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # Check session ownership
    print(f"\n[2] Checking session ownership for user '{user_id}'")

    owned_sessions = cur.execute(
        "SELECT COUNT(DISTINCT session_id) FROM mention WHERE eid = ?",
        (user_id,)
    ).fetchone()[0]

    print(f"   Sessions owned by '{user_id}': {owned_sessions}")

    if owned_sessions == 0:
        print(f"   ❌ FAILED: No sessions found for user '{user_id}'")

        # Check what users do have sessions
        print("\n   Available users with sessions:")
        users_with_sessions = cur.execute("""
            SELECT DISTINCT eid, COUNT(DISTINCT session_id) as session_count
            FROM mention
            WHERE eid NOT LIKE 'session:%' AND eid NOT LIKE 'summary:%'
            GROUP BY eid
            ORDER BY session_count DESC
            LIMIT 5
        """).fetchall()

        for eid, count in users_with_sessions:
            print(f"     • {eid}: {count} sessions")

        print(f"\n   💡 Suggested fix:")
        if users_with_sessions:
            suggested_user = users_with_sessions[0][0]
            print(f"      Change USER_ID in .env to: {suggested_user}")
        return False

    # Check accessible edges
    print(f"\n[3] Checking accessible memory edges")

    # Get all edges
    total_edges = cur.execute("SELECT COUNT(*) FROM edge WHERE status = 1").fetchone()[0]
    print(f"   Total active edges in database: {total_edges}")

    # Simulate retrieval visibility check
    accessible_edges = cur.execute("""
        SELECT COUNT(DISTINCT e.id)
        FROM edge e
        JOIN edge_source es ON e.id = es.edge_id
        JOIN conversation_turn ct ON es.turn_id = ct.id
        WHERE ct.session_id IN (
            SELECT DISTINCT session_id FROM mention WHERE eid = ?
        )
        AND e.status = 1
    """, (user_id,)).fetchone()[0]

    print(f"   Edges accessible to '{user_id}': {accessible_edges}")

    accessibility_pct = (accessible_edges / total_edges * 100) if total_edges > 0 else 0
    print(f"   Accessibility: {accessibility_pct:.1f}%")

    if accessible_edges == 0:
        print(f"   ❌ FAILED: No edges are accessible to user '{user_id}'")
        return False
    elif accessibility_pct < 50:
        print(f"   ⚠️  WARNING: Only {accessibility_pct:.1f}% of edges are accessible")
        return False
    else:
        print(f"   ✅ GOOD: {accessibility_pct:.1f}% of edges are accessible")

    # Test sample retrieval
    print(f"\n[4] Testing sample retrieval")

    # Get a sample conversation turn
    sample_turn = cur.execute("""
        SELECT ct.text
        FROM conversation_turn ct
        WHERE ct.session_id IN (
            SELECT DISTINCT session_id FROM mention WHERE eid = ?
        )
        ORDER BY ct.ts DESC
        LIMIT 1
    """, (user_id,)).fetchone()

    if sample_turn:
        text = sample_turn[0]
        print(f"   Sample conversation: \"{text[:60]}...\"")
    else:
        print(f"   ⚠️  No conversation turns found")

    # Get sample edges
    sample_edges = cur.execute("""
        SELECT e.src, e.rel, e.dst, e.weight
        FROM edge e
        JOIN edge_source es ON e.id = es.edge_id
        JOIN conversation_turn ct ON es.turn_id = ct.id
        WHERE ct.session_id IN (
            SELECT DISTINCT session_id FROM mention WHERE eid = ?
        )
        AND e.status = 1
        ORDER BY e.updated_at DESC
        LIMIT 3
    """, (user_id,)).fetchall()

    if sample_edges:
        print(f"   Sample accessible edges:")
        for src, rel, dst, weight in sample_edges:
            print(f"     • {src} → {rel} → {dst} (weight={weight:.2f})")
    else:
        print(f"   ⚠️  No edges found")

    # Final verdict
    print("\n" + "="*70)
    if owned_sessions > 0 and accessible_edges > 0:
        print("✅ FIX VERIFIED: Memory retrieval should be working!")
        print(f"\n   User '{user_id}' can access:")
        print(f"   • {owned_sessions} sessions")
        print(f"   • {accessible_edges} memory edges ({accessibility_pct:.1f}% of total)")
        print(f"\n   Test in conversation:")
        print(f"   • Ask: \"What did we talk about before?\"")
        print(f"   • Ask: \"What do you remember about me?\"")
        print("="*70)
        return True
    else:
        print("❌ FIX NOT WORKING: Memory retrieval is still broken")
        print("\n   Please check:")
        print("   1. Is USER_ID in .env set to the correct user?")
        print("   2. Did you restart the server after changing .env?")
        print("   3. Does the database contain data for this user?")
        print("="*70)
        return False


if __name__ == "__main__":
    # Load .env if available
    try:
        from dotenv import load_dotenv
        env_path = Path(_SERVER_ROOT) / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded .env from: {env_path}")
    except ImportError:
        print("dotenv not available, using environment variables")

    success = verify_fix()
    sys.exit(0 if success else 1)
