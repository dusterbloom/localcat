#!/usr/bin/env python3
"""
Diagnostic script for cross-session memory retrieval issues

This script analyzes the memory database and configuration to identify
why memory retrieval might fail across sessions for the same user.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add server root to path
_HERE = Path(__file__).parent
_SERVER_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_SERVER_ROOT))


def diagnose_memory_system():
    """Run comprehensive diagnostics on the memory system"""

    print("="*70)
    print("MEMORY SYSTEM CROSS-SESSION RETRIEVAL DIAGNOSTICS")
    print("="*70)

    # 1. Check environment configuration
    print("\n[1] ENVIRONMENT CONFIGURATION")
    print("-" * 70)

    user_id = os.getenv('USER_ID', 'NOT_SET')
    agent_id = os.getenv('AGENT_ID', 'NOT_SET')
    memory_mode = os.getenv('MEMORY_MODE', 'NOT_SET')
    session_persistence = os.getenv('VOICE_AGENT_SESSION_PERSISTENCE', 'NOT_SET')
    memory_enabled = os.getenv('VOICE_AGENT_MEMORY_ENABLED', 'NOT_SET')

    print(f"USER_ID: {user_id}")
    print(f"AGENT_ID: {agent_id}")
    print(f"MEMORY_MODE: {memory_mode}")
    print(f"VOICE_AGENT_SESSION_PERSISTENCE: {session_persistence}")
    print(f"VOICE_AGENT_MEMORY_ENABLED: {memory_enabled}")

    # 2. Locate and connect to database
    print("\n[2] DATABASE CONNECTION")
    print("-" * 70)

    db_path = Path(_SERVER_ROOT.parent / "data" / "memory.db")
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        return

    print(f"✓ Database found at: {db_path}")
    print(f"  Size: {db_path.stat().st_size / 1024:.1f} KB")

    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # 3. Check database content
    print("\n[3] DATABASE CONTENT")
    print("-" * 70)

    edge_count = cur.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    turn_count = cur.execute("SELECT COUNT(*) FROM conversation_turn").fetchone()[0]
    mention_count = cur.execute("SELECT COUNT(*) FROM mention").fetchone()[0]

    print(f"Edges (facts): {edge_count}")
    print(f"Conversation turns: {turn_count}")
    print(f"Mentions: {mention_count}")

    # 4. Analyze session IDs
    print("\n[4] SESSION ID ANALYSIS")
    print("-" * 70)

    sessions = cur.execute(
        "SELECT DISTINCT session_id FROM conversation_turn ORDER BY session_id DESC LIMIT 10"
    ).fetchall()

    if sessions:
        print("Recent session IDs (last 10):")
        user_prefixes = set()
        for (sid,) in sessions:
            parts = sid.split('_')
            user_prefix = parts[0] if parts else 'unknown'
            user_prefixes.add(user_prefix)
            print(f"  • {sid} (user: {user_prefix})")

        print(f"\nUnique user prefixes found: {sorted(user_prefixes)}")

        # Check if current USER_ID matches any session prefix
        if user_id != 'NOT_SET':
            matching_sessions = [s for (s,) in sessions if s.startswith(user_id + '_')]
            if matching_sessions:
                print(f"✓ Current USER_ID '{user_id}' matches {len(matching_sessions)} sessions")
            else:
                print(f"⚠️  WARNING: Current USER_ID '{user_id}' doesn't match any sessions!")
                print(f"   All sessions use these prefixes: {sorted(user_prefixes)}")
    else:
        print("No sessions found in database")

    # 5. Check ownership mapping (mention table)
    print("\n[5] SESSION OWNERSHIP MAPPING")
    print("-" * 70)

    print("The system uses the 'mention' table to map sessions to users.")
    print("It looks for records where eid = user_id to determine ownership.\n")

    if user_id != 'NOT_SET':
        owned_sessions = cur.execute(
            "SELECT DISTINCT session_id FROM mention WHERE eid = ?",
            (user_id,)
        ).fetchall()

        print(f"Sessions owned by user '{user_id}': {len(owned_sessions)}")
        if owned_sessions:
            for (sid,) in owned_sessions[:5]:
                print(f"  • {sid}")
        else:
            print(f"  ⚠️  NO SESSIONS FOUND for user '{user_id}'")

            # Check if there are sessions for other users
            print("\n  Checking sessions for other potential users:")
            for prefix in sorted(user_prefixes):
                owned = cur.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM mention WHERE eid = ?",
                    (prefix,)
                ).fetchone()[0]
                if owned > 0:
                    print(f"    • User '{prefix}': {owned} sessions")

    # 6. Test retrieval visibility
    print("\n[6] RETRIEVAL VISIBILITY TEST")
    print("-" * 70)

    if user_id != 'NOT_SET' and sessions:
        test_session = sessions[0][0]  # Most recent session
        test_user_prefix = test_session.split('_')[0]

        print(f"Testing if user '{user_id}' can access session '{test_session[:50]}...'")

        # Get edges from this session
        edges_from_session = cur.execute("""
            SELECT DISTINCT e.id, e.src, e.rel, e.dst, e.weight
            FROM edge e
            JOIN edge_source es ON e.id = es.edge_id
            JOIN conversation_turn ct ON es.turn_id = ct.id
            WHERE ct.session_id = ?
            LIMIT 5
        """, (test_session,)).fetchall()

        if edges_from_session:
            print(f"  Found {len(edges_from_session)} edges from this session:")
            for edge_id, src, rel, dst, weight in edges_from_session:
                print(f"    • {src} -> {rel} -> {dst} (weight={weight:.2f})")

            # Check if these edges are visible to current user
            edge_ids = [e[0] for e in edges_from_session]

            # Simulate the ownership check from retrieval.py
            visible_count = 0
            for edge_id in edge_ids:
                # Get provenance for this edge
                prov = cur.execute("""
                    SELECT ct.session_id
                    FROM edge_source es
                    JOIN conversation_turn ct ON es.turn_id = ct.id
                    WHERE es.edge_id = ?
                """, (edge_id,)).fetchall()

                prov_sessions = [p[0] for p in prov]

                # Check if any provenance session is owned by current user
                for prov_session in prov_sessions:
                    is_owned = cur.execute(
                        "SELECT 1 FROM mention WHERE session_id = ? AND eid = ?",
                        (prov_session, user_id)
                    ).fetchone()

                    if is_owned:
                        visible_count += 1
                        break

            print(f"\n  Visibility result:")
            print(f"    • Total edges: {len(edge_ids)}")
            print(f"    • Visible to user '{user_id}': {visible_count}")
            print(f"    • Hidden: {len(edge_ids) - visible_count}")

            if visible_count == 0:
                print(f"\n  ❌ PROBLEM IDENTIFIED:")
                print(f"     User '{user_id}' cannot see edges from session '{test_session[:50]}...'")
                print(f"     This is because there are no mention records with eid='{user_id}'")
                print(f"     for this session.")

    # 7. Root cause summary
    print("\n[7] ROOT CAUSE ANALYSIS")
    print("="*70)

    if user_id != 'NOT_SET' and user_id not in user_prefixes:
        print("❌ CROSS-SESSION RETRIEVAL IS BROKEN")
        print(f"\nCause: USER_ID mismatch")
        print(f"  • Current USER_ID in .env: '{user_id}'")
        print(f"  • User prefixes in database: {sorted(user_prefixes)}")
        print(f"  • All historical data belongs to user(s): {', '.join(sorted(user_prefixes))}")
        print(f"\nImpact:")
        print(f"  • New sessions created by '{user_id}' cannot access historical data")
        print(f"  • Historical data from '{', '.join(sorted(user_prefixes))}' is invisible")
        print(f"\nSolution:")
        print(f"  1. Change USER_ID in .env from '{user_id}' to '{list(user_prefixes)[0]}'")
        print(f"  2. OR: Migrate database session_ids to use new user_id")
        print(f"  3. OR: Implement user-independent retrieval (remove ownership checks)")
    elif user_id == 'NOT_SET':
        print("⚠️  USER_ID is not set in environment")
        print("\nThe system may not properly scope memory to users.")
    else:
        print("✓ Configuration looks correct")
        print(f"\nUser '{user_id}' should be able to access historical data.")

    print("\n" + "="*70)
    print("END OF DIAGNOSTICS")
    print("="*70)


if __name__ == "__main__":
    # Create diagnostics directory if it doesn't exist
    Path(__file__).parent.mkdir(exist_ok=True)

    diagnose_memory_system()
