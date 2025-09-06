#!/usr/bin/env python3
"""
Rebuild the LEANN index from the current SQLite HotMem store.

Usage examples:
  python3 utils/rebuild_leann.py \
    --sqlite data/memory.db \
    --index  data/memory_vectors.leann \
    --backend hnsw \
    --include-summaries

Env fallbacks:
  HOTMEM_SQLITE, LEANN_INDEX_PATH, LEANN_BACKEND
"""
import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any


def load_docs(sqlite_path: str, include_summaries: bool = True) -> List[Dict[str, Any]]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    docs: List[Dict[str, Any]] = []
    for s, r, d, w, ts, status in cur.execute(
        "SELECT src, rel, dst, weight, updated_at, status FROM edge WHERE status >= 0"
    ):
        text = f"{s} {r} {d}"
        docs.append({
            'text': text,
            'metadata': {'src': s, 'rel': r, 'dst': d, 'weight': float(w), 'ts': int(ts)}
        })
    if include_summaries:
        for (text,) in cur.execute(
            "SELECT text FROM mention WHERE eid LIKE 'summary:%' OR eid LIKE 'session:%' ORDER BY ts DESC LIMIT 200"
        ):
            if text:
                docs.append({'text': str(text), 'metadata': {'type': 'session_summary'}})
    con.close()
    return docs


def ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Rebuild LEANN index from SQLite HotMem store")
    ap.add_argument("--sqlite", default=os.getenv("HOTMEM_SQLITE", "data/memory.db"))
    ap.add_argument("--index", default=os.getenv("LEANN_INDEX_PATH", "data/memory_vectors.leann"))
    ap.add_argument("--backend", default=os.getenv("LEANN_BACKEND", "hnsw"), choices=["hnsw", "diskann"])
    ap.add_argument("--include-summaries", action="store_true", help="Include periodic/session summaries")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of docs (debug)")
    args = ap.parse_args()

    try:
        from leann.api import LeannBuilder  # type: ignore
    except ImportError:
        print("ERROR: `leann` package not installed. Run: pip install leann")
        raise SystemExit(2)

    sqlite_path = str(Path(args.sqlite).expanduser())
    index_path = str(Path(args.index).expanduser())

    if not os.path.exists(sqlite_path):
        print(f"ERROR: SQLite not found at {sqlite_path}")
        raise SystemExit(2)

    docs = load_docs(sqlite_path, include_summaries=args.include_summaries)
    if args.limit > 0:
        docs = docs[: args.limit]
    print(f"Loaded {len(docs)} docs from {sqlite_path}")

    ensure_parent(index_path)
    builder = LeannBuilder(backend_name=args.backend)
    for d in docs:
        text = d.get('text') or ''
        meta = d.get('metadata') or {}
        if text.strip():
            builder.add_text(str(text), meta)
    builder.build_index(index_path)
    print(f"LEANN index rebuilt at {index_path}")


if __name__ == "__main__":
    main()

