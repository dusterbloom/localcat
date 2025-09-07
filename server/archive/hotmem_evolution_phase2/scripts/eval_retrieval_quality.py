#!/usr/bin/env python3
"""
Retrieval Quality Evaluation
===========================

Computes P@K, MRR, MAP, nDCG, diversity over a small gold set.

Runs A/B with LEANN on/off (if `leann` package installed and index exists) and
prints aggregated metrics. Uses a seeded in-memory dataset so results are
stable and reproducible.
"""
import os
import math
import time
import tempfile
from typing import List, Dict, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory


def seed_kg(store: MemoryStore):
    now = int(time.time() * 1000)
    def add(s,r,d,w=0.9):
        store.observe_edge(s,r,d,w,now)
    # User facts
    add('you','name','Sarah')
    add('you','works_at','Microsoft')
    add('you','lives_in','Seattle')
    add('you','favorite_color','purple')
    add('you','favorite_number','7')
    add('you','has','Tesla Model 3')
    # Brother facts
    add('brother','also_known_as','Tom')
    add('brother','lives_in','Portland')
    add('brother','teach_at','Reed College')
    # Pet facts
    add('dog','name','Luna')
    add('you','owns','dog')
    add('Luna','age','3 years old')


Query = Tuple[str, List[str]]  # (query, relevant substrings)

GOLD: List[Query] = [
    ("who works at microsoft", ["works at microsoft", "work at microsoft", "you work at microsoft"]),
    ("where does tom live", ["lives in portland", "tom lives in portland", "brother lives in portland"]),
    ("what is my dog's name", ["dog's name is luna", "luna"]),
    ("how old is luna", ["3 years", "3 years old"]),
    ("what car do i drive", ["tesla model 3", "you have tesla", "you have tesla model 3"]),
    ("what is my favorite color", ["favorite color is purple", "your favorite color is purple", "purple"]),
    ("what is my lucky number", ["favorite number is 7", "your favorite number is 7", "number is 7", "7"]),
    ("where does my brother teach", ["teach at reed", "reed college"]),
    ("tell me about my dog luna", ["dog's name is luna", "luna", "you have dog"]),
]


def score_run(hot: HotMemory, K: int = 5) -> Dict[str, float]:
    hits = []
    rr_list = []
    ap_list = []
    ndcg_list = []
    diversity_counts = []
    lat_ms = []

    for (q, rel_subs) in GOLD:
        t0 = time.perf_counter()
        # Heuristic entity selection: choose entity_index keys that fuzzy-match query tokens
        import re
        toks = re.findall(r"[a-z0-9]+", q.lower())
        keys = list(hot.entity_index.keys())
        ents = []
        for k in keys:
            kl = str(k).lower()
            if any(t in kl for t in toks):
                ents.append(k)
                if len(ents) >= 6:
                    break
        bullets = hot._retrieve_context(q, ents, turn_id=999, intent=None)
        lat_ms.append((time.perf_counter() - t0) * 1000)

        # Lower-case bullets for matching
        L = [b.lower() for b in bullets[:K]]
        # Gains vector for nDCG
        gains = [0]*len(L)

        # Compute first relevant rank (MRR), precision@K, AP
        first_rel = None
        rel_hits = 0
        precisions = []
        for i, b in enumerate(L, start=1):
            match = any(sub in b for sub in rel_subs)
            if match:
                gains[i-1] = 1
                rel_hits += 1
                if first_rel is None:
                    first_rel = i
                precisions.append(rel_hits / i)

        p_at_k = rel_hits / max(1, K)
        hits.append(p_at_k)
        rr_list.append(1.0/first_rel if first_rel else 0.0)
        ap_list.append(sum(precisions)/max(1, rel_hits) if rel_hits>0 else 0.0)

        # nDCG@K (binary gains)
        def dcg(vec):
            return sum(v / math.log2(i+2) for i, v in enumerate(vec))
        ideal = sorted(gains, reverse=True)
        ndcg = dcg(gains) / (dcg(ideal) or 1.0)
        ndcg_list.append(ndcg)

        # Diversity: unique (subject, relation) pairs in bullets parsed back
        uniq = set()
        for b in bullets[:K]:
            # crude parse: extract pattern "X rel Y"
            s = b.strip('• ').lower()
            # split first verb/prep marker heuristically
            for token in [' lives in ', ' works at ', ' was born in ', ' is ', ' has ', ' favorite number is ', ' favorite color is ']:
                if token in s:
                    left = s.split(token)[0].strip()
                    rel = token.strip()
                    uniq.add((left, rel))
                    break
        diversity_counts.append(len(uniq))

    metrics = {
        'P@5': sum(hits)/len(hits),
        'MRR': sum(rr_list)/len(rr_list),
        'MAP': sum(ap_list)/len(ap_list),
        'nDCG@5': sum(ndcg_list)/len(ndcg_list),
        'Diversity': sum(diversity_counts)/len(diversity_counts),
        'Latency_ms': sum(lat_ms)/len(lat_ms),
    }
    # Optional per-query breakdown
    if os.getenv('RETRIEVAL_EVAL_VERBOSE', 'false').lower() in ('1','true','yes'):
        print(f"Evaluated {len(GOLD)} queries; avg latency {metrics['Latency_ms']:.2f}ms")
    return metrics


def run_ab(leann: bool) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tdir:
        store = MemoryStore(Paths(
            sqlite_path=os.path.join(tdir, 'memory.db'),
            lmdb_dir=os.path.join(tdir, 'graph.lmdb')
        ))
        seed_kg(store)
        store.flush()
        hot = HotMemory(store)
        hot.rebuild_from_store()
        # A/B flag
        os.environ['HOTMEM_USE_LEANN'] = 'true' if leann else 'false'
        return score_run(hot)


def main():
    print("Retrieval Quality Evaluation (P@5, MRR, MAP, nDCG, Diversity)")
    print("="*72)
    res_lex = run_ab(leann=False)
    res_sem = run_ab(leann=True)

    def fmt(d):
        return ", ".join(f"{k}={v:.3f}" if isinstance(v,float) else f"{k}={v}" for k,v in d.items())

    print("Lexical (LEANN off): ", fmt(res_lex))
    print("Semantic (LEANN on):  ", fmt(res_sem))

    # Simple winner note
    better = 'Semantic' if res_sem['P@5'] >= res_lex['P@5'] else 'Lexical'
    print(f"\nWinner by P@5: {better}")


if __name__ == '__main__':
    main()
