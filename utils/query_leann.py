#!/usr/bin/env python3
"""
Query a LEANN index for quick sanity checks.

Usage:
  python3 utils/query_leann.py --index data/memory_vectors.leann --k 5 --complexity 16 "dog name"
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Query a LEANN index")
    ap.add_argument("query", help="search query text")
    ap.add_argument("--index", default="data/memory_vectors.leann")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--complexity", type=int, default=16)
    args = ap.parse_args()

    try:
        from leann.api import LeannSearcher  # type: ignore
    except ImportError:
        print("ERROR: `leann` package not installed. Run: pip install leann")
        raise SystemExit(2)

    idx = str(Path(args.index).expanduser())
    s = LeannSearcher(idx)
    results = s.search(args.query, top_k=args.k, complexity=args.complexity)
    for i, r in enumerate(results):
        text = getattr(r, 'text', None) or str(r)
        meta = getattr(r, 'metadata', None)
        print(f"{i+1}. {text[:120].replace('\n',' ')}  {meta if meta else ''}")


if __name__ == "__main__":
    main()

