#!/usr/bin/env python3
"""
Admin CLI for KG edges: list, promote, demote, forget.

Examples:
  source .venv/bin/activate && cd server
  python scripts/admin/edges.py list --rel live_in --status-min 0 --limit 20
  python scripts/admin/edges.py promote --s "parents" --r "watch_from" --d "wooden benches" --conf 0.8
  python scripts/admin/edges.py demote --s "parents" --r "watch_from" --d "wooden benches" --conf 0.5
  python scripts/admin/edges.py forget --s "startup" --r "moved_from" --d "San Francisco"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from components.memory.memory_store import MemoryStore, Paths  # type: ignore


def cmd_list(store: MemoryStore, args: argparse.Namespace) -> int:
    cur = store.sql.cursor()
    q = [
        "SELECT e.src, e.rel, e.dst, e.weight, e.status, e.updated_at, m.prov, m.props",
        "FROM edge e LEFT JOIN edge_meta m ON e.id = m.id",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    if args.s:
        q.append("AND e.src LIKE ?")
        params.append(args.s)
    if args.r:
        q.append("AND e.rel LIKE ?")
        params.append(args.r)
    if args.d:
        q.append("AND e.dst LIKE ?")
        params.append(args.d)
    if args.status_min is not None:
        q.append("AND e.status >= ?")
        params.append(int(args.status_min))
    if args.prov:
        q.append("AND m.prov LIKE ?")
        params.append(args.prov)
    q.append("ORDER BY e.updated_at DESC")
    if args.limit:
        q.append("LIMIT ?")
        params.append(int(args.limit))

    rows = cur.execute("\n".join(q), tuple(params)).fetchall()
    out = []
    for (s, r, d, w, status, upd, prov, props) in rows:
        try:
            pj = json.loads(props) if props else {}
        except Exception:
            pj = {}
        out.append({
            's': s, 'r': r, 'd': d,
            'weight': float(w), 'status': int(status), 'updated_at': int(upd or 0),
            'prov': prov or '', 'props': pj,
        })
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for o in out:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(o['updated_at'])) if o['updated_at'] else '-'
            print(f"({o['s']}, {o['r']}, {o['d']}) w={o['weight']:.2f} status={o['status']} ts={ts} prov={o['prov']}")
    return 0


def cmd_promote(store: MemoryStore, args: argparse.Namespace) -> int:
    now = int(time.time() * 1000)
    store.observe_edge(args.s, args.r, args.d, float(args.conf), now)
    store.flush()
    print(f"Promoted ({args.s}, {args.r}, {args.d}) conf={args.conf}")
    return 0


def cmd_demote(store: MemoryStore, args: argparse.Namespace) -> int:
    now = int(time.time() * 1000)
    store.negate_edge(args.s, args.r, args.d, float(args.conf), now)
    store.flush()
    print(f"Demoted ({args.s}, {args.r}, {args.d}) conf={args.conf}")
    return 0


def cmd_forget(store: MemoryStore, args: argparse.Namespace) -> int:
    store.hard_forget(args.s, args.r, args.d)
    store.flush()
    target = (args.s, args.r, args.d)
    print(f"Forgot edge {target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap.add_argument('--json', action='store_true')

    s_list = sub.add_parser('list')
    s_list.add_argument('--s')
    s_list.add_argument('--r')
    s_list.add_argument('--d')
    s_list.add_argument('--status-min', type=int, default=None)
    s_list.add_argument('--prov')
    s_list.add_argument('--limit', type=int, default=50)

    s_prom = sub.add_parser('promote')
    s_prom.add_argument('--s', required=True)
    s_prom.add_argument('--r', required=True)
    s_prom.add_argument('--d', required=True)
    s_prom.add_argument('--conf', type=float, default=0.8)

    s_demo = sub.add_parser('demote')
    s_demo.add_argument('--s', required=True)
    s_demo.add_argument('--r', required=True)
    s_demo.add_argument('--d', required=True)
    s_demo.add_argument('--conf', type=float, default=0.5)

    s_forget = sub.add_parser('forget')
    s_forget.add_argument('--s', required=True)
    s_forget.add_argument('--r')
    s_forget.add_argument('--d')

    args = ap.parse_args()

    paths = Paths()
    store = MemoryStore(paths=paths)

    if args.cmd == 'list':
        return cmd_list(store, args)
    if args.cmd == 'promote':
        return cmd_promote(store, args)
    if args.cmd == 'demote':
        return cmd_demote(store, args)
    if args.cmd == 'forget':
        return cmd_forget(store, args)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

