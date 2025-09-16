#!/usr/bin/env python
"""Clean corrupted/incomplete FTS entries while preserving good data."""

import sqlite3
import re
from pathlib import Path

def is_corrupted_entry(text):
    """Identify corrupted/incomplete entries."""
    if not text:
        return True

    # Patterns that indicate corruption
    corruption_patterns = [
        # Incomplete sentences (ends mid-word or with incomplete phrase)
        r'(?:^|\. )[A-Z][^.!?]*\s+(?:in|at|on|with|from|to|of|a|the|an)\s*$',
        # "You lives" grammatical error (should be "You live")
        r'\bYou lives\b',
        # Single word sentences or very short fragments
        r'^[A-Za-z]+\s*$',
        # Sentences that end with prepositions without objects
        r'\b(?:in|at|on|with|from|to|of|for|by)\s*$',
        # Truncated mid-sentence (no punctuation, ends with common words)
        r'\b(?:the|a|an|is|are|was|were|has|have|had|will|would|can|could|should|may|might)\s*$',
    ]

    text = text.strip()

    # Check if text matches corruption patterns
    for pattern in corruption_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Check for very short entries (less than 10 chars) that aren't complete thoughts
    if len(text) < 10 and not text.endswith(('.', '!', '?')):
        return True

    return False

def clean_fts_index(db_path):
    """Clean the FTS index by removing corrupted entries."""

    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # First, let's see what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {[t[0] for t in tables]}")

    # Check if we have an FTS table
    fts_tables = [t[0] for t in tables if 'fts' in t[0].lower()]

    if not fts_tables:
        print("No FTS tables found.")
        conn.close()
        return

    for fts_table in fts_tables:
        print(f"\nProcessing FTS table: {fts_table}")

        # Get all entries from the FTS table
        try:
            cursor.execute(f"SELECT rowid, * FROM {fts_table}")
            columns = [desc[0] for desc in cursor.description]
            entries = cursor.fetchall()

            print(f"Found {len(entries)} entries in {fts_table}")

            # Find text column (usually 'content' or 'text' or similar)
            text_col_idx = None
            for idx, col in enumerate(columns):
                if col.lower() in ['content', 'text', 'summary', 'data']:
                    text_col_idx = idx
                    break

            if text_col_idx is None:
                print(f"Could not find text column in {fts_table}")
                continue

            corrupted = []
            preserved = []

            # Analyze entries
            for entry in entries:
                rowid = entry[0]
                text = entry[text_col_idx] if text_col_idx < len(entry) else None

                if text and is_corrupted_entry(text):
                    corrupted.append((rowid, text))
                else:
                    preserved.append((rowid, text))

            print(f"\nAnalysis:")
            print(f"  - Corrupted entries to remove: {len(corrupted)}")
            print(f"  - Good entries to preserve: {len(preserved)}")

            # Show examples of what will be removed
            if corrupted:
                print(f"\nExamples of corrupted entries to be removed:")
                for rowid, text in corrupted[:5]:
                    print(f"  [{rowid}] '{text[:100]}...' " if len(text) > 100 else f"  [{rowid}] '{text}'")

            # Show examples of what will be preserved
            if preserved:
                print(f"\nExamples of good entries to be preserved:")
                for rowid, text in preserved[:5]:
                    if text:
                        print(f"  [{rowid}] '{text[:100]}...' " if len(text) > 100 else f"  [{rowid}] '{text}'")

            # Delete corrupted entries
            if corrupted:
                print(f"\nRemoving {len(corrupted)} corrupted entries...")
                for rowid, _ in corrupted:
                    cursor.execute(f"DELETE FROM {fts_table} WHERE rowid = ?", (rowid,))
                conn.commit()
                print(f"Successfully removed {len(corrupted)} corrupted entries from {fts_table}")

        except Exception as e:
            print(f"Error processing {fts_table}: {e}")
            continue

    # Also check the regular memory tables for corrupted data
    print("\n\nChecking regular tables for corrupted data...")

    # Check edges table if it exists
    if 'edges' in [t[0] for t in tables]:
        cursor.execute("SELECT rowid, subject, relation, object FROM edges")
        edges = cursor.fetchall()
        corrupted_edges = []

        for rowid, subj, rel, obj in edges:
            # Check for corrupted patterns in edges
            if (subj == 'You' and rel == 'lives' and obj == 'in a') or \
               (obj and obj.endswith(' a')) or \
               (obj and len(obj.strip()) < 2):
                corrupted_edges.append((rowid, subj, rel, obj))

        if corrupted_edges:
            print(f"\nFound {len(corrupted_edges)} corrupted edges:")
            for rowid, s, r, o in corrupted_edges[:5]:
                print(f"  [{rowid}] ({s}, {r}, {o})")

            print(f"Removing corrupted edges...")
            for rowid, _, _, _ in corrupted_edges:
                cursor.execute("DELETE FROM edges WHERE rowid = ?", (rowid,))
            conn.commit()
            print(f"Removed {len(corrupted_edges)} corrupted edges")

    # Vacuum to reclaim space
    print("\nVacuuming database to reclaim space...")
    conn.execute("VACUUM")

    conn.close()
    print("\nDatabase cleanup complete!")

if __name__ == "__main__":
    # Path to the memory database
    db_path = Path("/Users/peppi/Dev/localcat/server/components/data/memory.db")

    if not db_path.exists():
        print(f"Database not found at {db_path}")
    else:
        clean_fts_index(db_path)