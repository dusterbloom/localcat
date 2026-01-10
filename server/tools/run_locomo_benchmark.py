#!/usr/bin/env python3
"""
LoCoMo Benchmark for LocalCat Memory System
============================================

Computes F1 score comparable to SimpleMem (43.24%) and Mem0 (34.15%).

This benchmark tests the RETRIEVAL layer using SQLite FTS, independent of
the extraction layer (spacy). This is useful because:
1. Extraction is being replaced with LLM-based approach
2. Retrieval latency is the production bottleneck
3. FTS baseline shows where we need semantic retrieval

Usage:
    python tools/run_locomo_benchmark.py [--full]

    --full: Requires spacy, runs full HotMemory pipeline
    (default): Runs FTS-only benchmark, no heavy dependencies
"""

import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent
SERVER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SERVER_DIR))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    if not prediction or not ground_truth:
        return 0.0

    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(str(ground_truth).lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def load_locomo_dataset(data_path: Optional[str] = None) -> list:
    """Load LoCoMo-10 dataset."""
    if data_path is None:
        candidates = [
            SERVER_DIR.parent / "docs" / "locomo10.json",
            Path("/home/user/localcat/docs/locomo10.json"),
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break

    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError("Could not find locomo10.json dataset")

    with open(data_path) as f:
        return json.load(f)


def extract_turns_from_conversation(conv_data: dict) -> list[dict]:
    """Extract all turns from a LoCoMo conversation entry."""
    turns = []
    conversation = conv_data.get("conversation", {})

    # Get speaker names
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    # Extract all sessions (session_1, session_2, etc.)
    session_keys = sorted(
        [k for k in conversation.keys()
         if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0
    )

    for session_key in session_keys:
        session_turns = conversation.get(session_key, [])
        session_datetime = conversation.get(f"{session_key}_date_time", "")

        if not isinstance(session_turns, list):
            continue

        for turn in session_turns:
            if not isinstance(turn, dict):
                continue

            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")

            if text:
                turns.append({
                    "speaker": speaker,
                    "text": text,
                    "dia_id": dia_id,
                    "session": session_key,
                    "datetime": session_datetime,
                })

    return turns


class FTSMemoryStore:
    """
    Lightweight FTS-based memory store for benchmarking.
    Uses SQLite FTS5 for full-text search retrieval.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup_schema()

    def _setup_schema(self):
        """Create FTS5 tables for conversation storage."""
        self.conn.executescript("""
            -- Main conversation turns table
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                dia_id TEXT,
                session TEXT,
                datetime TEXT,
                conv_idx INTEGER
            );

            -- FTS5 index for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                text,
                speaker,
                content='turns',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                INSERT INTO turns_fts(rowid, text, speaker)
                VALUES (new.id, new.text, new.speaker);
            END;

            CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, text, speaker)
                VALUES ('delete', old.id, old.text, old.speaker);
            END;

            CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, text, speaker)
                VALUES ('delete', old.id, old.text, old.speaker);
                INSERT INTO turns_fts(rowid, text, speaker)
                VALUES (new.id, new.text, new.speaker);
            END;
        """)
        self.conn.commit()

    def add_turn(self, speaker: str, text: str, dia_id: str = "",
                 session: str = "", datetime: str = "", conv_idx: int = 0):
        """Add a conversation turn to the store."""
        self.conn.execute(
            """INSERT INTO turns (speaker, text, dia_id, session, datetime, conv_idx)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (speaker, text, dia_id, session, datetime, conv_idx)
        )

    def commit(self):
        """Commit pending changes."""
        self.conn.commit()

    def search(self, query: str, limit: int = 5) -> list[str]:
        """
        Search for relevant turns using FTS5.
        Returns list of matching text snippets.
        """
        # Escape special FTS characters and build query
        # Use MATCH with BM25 ranking
        safe_query = query.replace('"', '""')

        # Try phrase match first, then fall back to OR match
        results = []

        # Method 1: BM25 ranked search with individual terms
        terms = [t.strip() for t in safe_query.split() if t.strip()]
        if terms:
            # Build OR query for flexibility
            fts_query = " OR ".join(f'"{t}"' for t in terms[:10])  # Limit terms

            try:
                cursor = self.conn.execute(
                    """SELECT t.text, t.speaker, bm25(turns_fts) as rank
                       FROM turns_fts f
                       JOIN turns t ON f.rowid = t.id
                       WHERE turns_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, limit)
                )
                results = [f"{row['speaker']}: {row['text']}" for row in cursor]
            except sqlite3.OperationalError:
                # If FTS query fails, fall back to LIKE
                pass

        # Method 2: Fallback to LIKE if FTS didn't find anything
        if not results and terms:
            like_conditions = " OR ".join(
                "text LIKE ?" for _ in terms[:5]
            )
            like_params = [f"%{t}%" for t in terms[:5]]

            cursor = self.conn.execute(
                f"""SELECT speaker, text FROM turns
                    WHERE {like_conditions}
                    LIMIT ?""",
                like_params + [limit]
            )
            results = [f"{row['speaker']}: {row['text']}" for row in cursor]

        return results

    def get_stats(self) -> dict:
        """Get storage statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM turns")
        turn_count = cursor.fetchone()[0]

        cursor = self.conn.execute(
            "SELECT COUNT(DISTINCT conv_idx) FROM turns"
        )
        conv_count = cursor.fetchone()[0]

        return {
            "total_turns": turn_count,
            "conversations": conv_count,
        }


def run_fts_benchmark(data: list) -> dict:
    """
    Run LoCoMo benchmark using FTS-only retrieval.

    This tests the retrieval layer without NLP extraction.
    """
    print("\n" + "=" * 60)
    print("LOCOMO BENCHMARK - FTS RETRIEVAL")
    print("=" * 60)

    # Initialize store
    store = FTSMemoryStore(":memory:")

    # Load all conversations
    print("\n📥 Loading conversations...")
    total_turns = 0

    for conv_idx, conv_data in enumerate(data):
        turns = extract_turns_from_conversation(conv_data)

        for turn in turns:
            store.add_turn(
                speaker=turn["speaker"],
                text=turn["text"],
                dia_id=turn["dia_id"],
                session=turn["session"],
                datetime=turn["datetime"],
                conv_idx=conv_idx,
            )
            total_turns += 1

        if (conv_idx + 1) % 2 == 0:
            print(f"   Loaded {conv_idx + 1}/10 conversations...")

    store.commit()

    stats = store.get_stats()
    print(f"✅ Loaded {stats['total_turns']} turns from {stats['conversations']} conversations")

    # Run QA evaluation
    print("\n📊 Evaluating QA pairs...")

    results_by_category = defaultdict(list)
    all_f1_scores = []
    latencies = []

    total_qa = sum(len(conv.get("qa", [])) for conv in data)
    processed = 0

    category_names = {
        1: "Factual",
        2: "Temporal",
        3: "Inference",
        4: "Multi-hop",
        5: "Adversarial"
    }

    for conv in data:
        for qa in conv.get("qa", []):
            question = qa["question"]
            # Handle both regular and adversarial answers
            answer = qa.get("answer") or qa.get("adversarial_answer", "")
            answer = str(answer)
            category = qa.get("category", 0)

            if not answer:
                continue  # Skip if no answer available

            # Query memory
            start = time.perf_counter()
            retrieved = store.search(question, limit=5)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            # Combine retrieved texts into prediction
            prediction = " ".join(retrieved) if retrieved else ""

            # Compute F1
            f1 = compute_f1(prediction, answer)

            results_by_category[category].append(f1)
            all_f1_scores.append(f1)

            processed += 1
            if processed % 400 == 0:
                avg_so_far = sum(all_f1_scores) / len(all_f1_scores) * 100
                print(f"   {processed}/{total_qa} ({100*processed/total_qa:.0f}%) - F1 so far: {avg_so_far:.1f}%")

    # Compute final metrics
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) * 100
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n📊 Overall F1: {avg_f1:.2f}%")
    print(f"\n⚡ Latency:")
    print(f"   Mean: {avg_latency:.3f}ms")
    print(f"   P50:  {p50_latency:.3f}ms")
    print(f"   P95:  {p95_latency:.3f}ms")
    print(f"   P99:  {p99_latency:.3f}ms")

    print(f"\n📈 By Category:")
    for cat in sorted(results_by_category.keys()):
        scores = results_by_category[cat]
        cat_f1 = sum(scores) / len(scores) * 100
        name = category_names.get(cat, f"Cat{cat}")
        print(f"   {name}: {cat_f1:.2f}% ({len(scores)} questions)")

    print("\n" + "=" * 60)
    print("COMPARISON WITH OTHER SYSTEMS")
    print("=" * 60)
    print(f"   SimpleMem (SOTA):     43.24%")
    print(f"   Mem0:                 34.15%")
    print(f"   LocalCat FTS:         {avg_f1:.2f}%")

    if avg_f1 > 43.24:
        print(f"\n🎉 LocalCat BEATS SimpleMem by {avg_f1 - 43.24:.2f}%!")
    elif avg_f1 > 34.15:
        gap = 43.24 - avg_f1
        print(f"\n⚠️  LocalCat beats Mem0 but is {gap:.2f}% behind SimpleMem")
        print(f"   Gap likely due to: missing semantic search, no extraction")
    else:
        print(f"\n❌ LocalCat is {34.15 - avg_f1:.2f}% behind Mem0")
        print(f"   FTS alone is insufficient - need semantic retrieval")

    return {
        "benchmark": "LoCoMo-10",
        "method": "FTS-only",
        "overall_f1": avg_f1,
        "latency_ms": {
            "mean": avg_latency,
            "p50": p50_latency,
            "p95": p95_latency,
            "p99": p99_latency,
        },
        "by_category": {
            category_names.get(k, f"Cat{k}"): {
                "f1": sum(v) / len(v) * 100,
                "count": len(v)
            }
            for k, v in results_by_category.items()
        },
        "total_questions": total_qa,
        "total_turns": total_turns,
        "comparison": {
            "SimpleMem": 43.24,
            "Mem0": 34.15,
            "LocalCat_FTS": avg_f1,
        }
    }


def run_full_benchmark(data: list) -> dict:
    """
    Run full benchmark using HotMemory with extraction.
    Requires spacy and other heavy dependencies.
    """
    print("\n" + "=" * 60)
    print("LOCOMO BENCHMARK - FULL HOTMEMORY")
    print("=" * 60)

    # Import memory components
    try:
        from core.memory.memory_store import MemoryStore, Paths
        from core.memory.memory_hotpath import HotMemory
        print("✅ Memory components imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nFull benchmark requires:")
        print("  - spacy + en_core_web_sm model")
        print("  - All dependencies from requirements.txt")
        print("\nFalling back to FTS benchmark...")
        return run_fts_benchmark(data)

    # Initialize memory
    print("\n📥 Initializing memory system...")
    try:
        os.environ["HOTMEM_SQLITE"] = ":memory:"
        store = MemoryStore(Paths(sqlite_path=":memory:"))
        hot = HotMemory(store)
        print("✅ HotMemory initialized")
    except Exception as e:
        print(f"❌ Memory init error: {e}")
        return run_fts_benchmark(data)

    # Load conversations
    print("\n📥 Processing conversations through HotMemory...")
    total_turns = 0

    for conv_idx, conv_data in enumerate(data):
        turns = extract_turns_from_conversation(conv_data)

        for turn in turns:
            try:
                hot.process_turn(
                    f"{turn['speaker']}: {turn['text']}",
                    session_id=f"conv{conv_idx}_{turn['session']}",
                    turn_id=total_turns
                )
                total_turns += 1
            except Exception as e:
                if total_turns == 0:
                    print(f"❌ Error processing turn: {e}")
                    return run_fts_benchmark(data)

        if (conv_idx + 1) % 2 == 0:
            print(f"   Processed {conv_idx + 1}/10 conversations...")

    print(f"✅ Processed {total_turns} turns")

    # Run QA evaluation
    print("\n📊 Evaluating QA pairs...")

    results_by_category = defaultdict(list)
    all_f1_scores = []
    latencies = []

    total_qa = sum(len(conv.get("qa", [])) for conv in data)
    processed = 0

    category_names = {
        1: "Factual",
        2: "Temporal",
        3: "Inference",
        4: "Multi-hop",
        5: "Adversarial"
    }

    for conv in data:
        for qa in conv.get("qa", []):
            question = qa["question"]
            # Handle both regular and adversarial answers
            answer = qa.get("answer") or qa.get("adversarial_answer", "")
            answer = str(answer)
            category = qa.get("category", 0)

            if not answer:
                continue  # Skip if no answer available

            # Query memory
            start = time.perf_counter()
            try:
                bullets = hot.retrieve_bullets(question, read_only=True)
            except Exception:
                bullets = []
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            # Combine retrieved bullets into prediction
            prediction = " ".join(bullets) if bullets else ""

            # Compute F1
            f1 = compute_f1(prediction, answer)

            results_by_category[category].append(f1)
            all_f1_scores.append(f1)

            processed += 1
            if processed % 400 == 0:
                avg_so_far = sum(all_f1_scores) / len(all_f1_scores) * 100
                print(f"   {processed}/{total_qa} ({100*processed/total_qa:.0f}%) - F1 so far: {avg_so_far:.1f}%")

    # Compute final metrics
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) * 100
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n📊 Overall F1: {avg_f1:.2f}%")
    print(f"\n⚡ Latency:")
    print(f"   Mean: {avg_latency:.3f}ms")
    print(f"   P50:  {p50_latency:.3f}ms")
    print(f"   P95:  {p95_latency:.3f}ms")
    print(f"   P99:  {p99_latency:.3f}ms")

    print(f"\n📈 By Category:")
    for cat in sorted(results_by_category.keys()):
        scores = results_by_category[cat]
        cat_f1 = sum(scores) / len(scores) * 100
        name = category_names.get(cat, f"Cat{cat}")
        print(f"   {name}: {cat_f1:.2f}% ({len(scores)} questions)")

    print("\n" + "=" * 60)
    print("COMPARISON WITH OTHER SYSTEMS")
    print("=" * 60)
    print(f"   SimpleMem (SOTA):     43.24%")
    print(f"   Mem0:                 34.15%")
    print(f"   LocalCat HotMemory:   {avg_f1:.2f}%")

    if avg_f1 > 43.24:
        print(f"\n🎉 LocalCat BEATS SimpleMem by {avg_f1 - 43.24:.2f}%!")
    elif avg_f1 > 34.15:
        gap = 43.24 - avg_f1
        print(f"\n⚠️  LocalCat beats Mem0 but is {gap:.2f}% behind SimpleMem")
    else:
        print(f"\n❌ LocalCat is {34.15 - avg_f1:.2f}% behind Mem0")

    return {
        "benchmark": "LoCoMo-10",
        "method": "HotMemory-full",
        "overall_f1": avg_f1,
        "latency_ms": {
            "mean": avg_latency,
            "p50": p50_latency,
            "p95": p95_latency,
            "p99": p99_latency,
        },
        "by_category": {
            category_names.get(k, f"Cat{k}"): {
                "f1": sum(v) / len(v) * 100,
                "count": len(v)
            }
            for k, v in results_by_category.items()
        },
        "total_questions": total_qa,
        "total_turns": total_turns,
        "comparison": {
            "SimpleMem": 43.24,
            "Mem0": 34.15,
            "LocalCat": avg_f1,
        }
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark on LocalCat memory system"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full HotMemory benchmark (requires spacy)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to locomo10.json dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Load dataset
    print("📂 Loading LoCoMo-10 dataset...")
    try:
        data = load_locomo_dataset(args.data)
        print(f"✅ Loaded {len(data)} conversations")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Count QA pairs
    total_qa = sum(len(conv.get("qa", [])) for conv in data)
    print(f"   Total QA pairs: {total_qa}")

    # Run benchmark
    if args.full:
        results = run_full_benchmark(data)
    else:
        results = run_fts_benchmark(data)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = SERVER_DIR / "tmp" / "locomo_results.json"

    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
