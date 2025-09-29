#!/usr/bin/env python3
"""
Migrate existing mention data to conversation_turn table
Run once after schema update
"""
import sqlite3
import sys
import os
from pathlib import Path

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.memory.memory_store import MemoryStore, Paths


def migrate_mentions_to_turns(db_path: str, dry_run: bool = False):
    """Migrate existing mention data to conversation_turn table"""

    print(f"Migrating provenance data in {db_path}")

    # Create store to use helper methods
    store = MemoryStore(Paths(sqlite_path=db_path, lmdb_dir=None))

    cur = store.sql.cursor()

    # Get unique conversation turns from mentions
    print("Extracting conversation turns from mentions...")
    turns = cur.execute("""
        SELECT DISTINCT session_id, turn_id, MIN(ts) as ts,
               GROUP_CONCAT(text, ' ') as combined_text
        FROM mention
        WHERE session_id IS NOT NULL AND turn_id IS NOT NULL
        GROUP BY session_id, turn_id
    """).fetchall()

    print(f"Found {len(turns)} unique conversation turns")

    if dry_run:
        print("\nDRY RUN - Sample of what would be migrated:")
        for i, (session_id, turn_num, ts, text) in enumerate(turns[:5]):
            tid = store.turn_id(session_id, turn_num)
            print(f"  Turn {i+1}: session={session_id}, turn={turn_num}, text_preview={text[:50]}...")
        print(f"\n... and {len(turns) - 5} more turns")
        print("\nNo changes made (dry run mode)")
        return 0

    # Insert into conversation_turn
    migrated = 0
    for session_id, turn_num, ts, text in turns:
        tid = store.turn_id(session_id, turn_num)
        cur.execute(
            "INSERT OR IGNORE INTO conversation_turn(id, text, session_id, turn_id, ts) "
            "VALUES(?, ?, ?, ?, ?)",
            (tid, text[:2000] if text else "", session_id, turn_num, ts or 0)
        )
        if cur.rowcount > 0:
            migrated += 1

    store.sql.commit()

    print(f"✅ Migrated {migrated} conversation turns")

    # Verify
    count = cur.execute("SELECT COUNT(*) FROM conversation_turn").fetchone()[0]
    print(f"Total conversation_turn rows: {count}")

    return migrated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate provenance data")
    parser.add_argument("--db", default="data/memory.db", help="Database path (relative to server/)")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = parser.parse_args()

    # Resolve path relative to server directory
    db_path = Path(__file__).parent.parent / args.db

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Please check the path or create a new database")
        sys.exit(1)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - no changes will be saved")
        print("=" * 60)

    try:
        result = migrate_mentions_to_turns(str(db_path), dry_run=args.dry_run)

        if args.dry_run:
            print("\n✅ Dry run complete - rerun without --dry-run to apply changes")
        elif result > 0:
            print(f"\n✅ Migration complete: {result} turns migrated")
        else:
            print("\n⚠️  No data to migrate (already migrated or no mentions)")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)