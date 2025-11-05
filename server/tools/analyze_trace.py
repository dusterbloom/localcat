#!/usr/bin/env python3
"""
Analyze retrieval trace NDJSON and produce a compact scorecard.

Reads MEMORY_TRACE_FILE output and aggregates:
- Profile/plan usage (when available)
- Source distribution in candidates and selected bullets
- Average candidate counts and selection rates per source
Usage:
  python server/tools/analyze_trace.py /path/to/trace.ndjson
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_trace.py TRACE.ndjson")
        sys.exit(1)
    path = sys.argv[1]
    plans = Counter()
    cand_src = Counter()
    sel_src = Counter()
    cand_len = []
    sel_len = []
    per_src_counts = defaultdict(list)
    per_src_selected = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            plan = j.get('plan') or {}
            if plan:
                # Normalize plan signature as a tuple for counting
                sig = tuple(sorted((k, int(v)) for k, v in plan.items()))
                plans[sig] += 1
            cands = j.get('candidates') or []
            selected = j.get('selected') or []
            cand_len.append(len(cands))
            sel_len.append(len(selected))
            # Count sources
            src_counts = Counter(x.get('source') for x in cands)
            for s, c in src_counts.items():
                cand_src[s] += c
                per_src_counts[s].append(c)
            sel_counts = Counter(x.get('source') for x in selected)
            for s, c in sel_counts.items():
                sel_src[s] += c
                per_src_selected[s].append(c)

    def avg(xs):
        return sum(xs) / max(1, len(xs))

    print("=== Retrieval Trace Summary ===")
    print(f"Traces: {sum(plans.values()) or (len(cand_len))}")
    if plans:
        print("Plan usage (quotas):")
        for sig, cnt in plans.most_common():
            print(f"  {dict(sig)} : {cnt}")
    print(f"Avg candidates: {avg(cand_len):.2f} | Avg selected: {avg(sel_len):.2f}")
    print("Candidate source totals:")
    for s, c in cand_src.most_common():
        print(f"  {s}: {c}")
    print("Selected source totals:")
    for s, c in sel_src.most_common():
        print(f"  {s}: {c}")


if __name__ == '__main__':
    main()

