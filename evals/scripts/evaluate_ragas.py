#!/usr/bin/env python3
"""
RAGAS evaluation for LocalCat memory retrieval.

Builds an in-memory HotMemory per case, ingests setup utterances, retrieves
context bullets for the query, and evaluates retrieval quality using RAGAS.

Usage:
  server/.venv/bin/python evals/scripts/evaluate_ragas.py \
    --cases server/tmp/cases_eval.jsonl \
    --out evals/outputs/ragas/baseline.json \
    --env MEMORY_RERANK_JINA_ENABLED=false

To compare variants, run the script twice with different --env overrides
and write to separate outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from pathlib import Path


def set_env(overrides: List[str]) -> Dict[str, str | None]:
    prev: Dict[str, str | None] = {}
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        prev[k] = os.getenv(k)
        os.environ[k] = v
    return prev


def restore_env(prev: Dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def load_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            # expected keys: id, setup (list[str]), query, gold (list[str])
            cases.append(j)
    return cases


def build_hotmemory():
    import sys
    ROOT = Path(__file__).resolve().parents[2] / "server"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.memory_hotpath import HotMemory

    # Use in-memory SQLite and disable LMDB explicitly to avoid permission/path issues
    # Honor workspace-local overrides for fidelity with LMDB
    db_path = os.getenv("MEMORY_DB_PATH", "").strip()
    lmdb_path = os.getenv("MEMORY_LMDB_PATH", "").strip()
    use_lmdb = os.getenv("MEMORY_USE_LMDB", "true").lower() not in ("0", "false", "no", "off")

    if db_path or (use_lmdb and lmdb_path):
        paths = Paths(
            sqlite_path=db_path if db_path else None,
            lmdb_dir=lmdb_path if use_lmdb else ""
        )
    else:
        # Default to in-memory for isolated evals
        paths = Paths(sqlite_path=":memory:", lmdb_dir="")

    store = MemoryStore(paths)
    hot = HotMemory(store)
    hot.prewarm("en")
    return hot


def normalize_contexts(bullets: List[str]) -> List[str]:
    out: List[str] = []
    for b in bullets:
        t = b
        # strip bullet marker and source tag if present
        if t.startswith("• "):
            t = t[2:]
        for tag in ("[graph]", "[convo]", "[summary]", "[semantic]"):
            t = t.replace(tag, "").strip()
        out.append(t)
    return out


def _load_dotenv_env(env_path: str) -> None:
    """Lightweight .env loader (no external deps). Only sets KEY=VAL pairs.

    - Ignores comments and blank lines
    - Does not expand variables or support export/quotes fully; keeps it simple
    - Does not override already-set environment variables
    """
    try:
        p = Path(env_path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "#" in s:
                # Remove inline comments
                s = s.split("#", 1)[0].strip()
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            # Do not overwrite existing env
            if k and (os.getenv(k) is None):
                os.environ[k] = v
    except Exception:
        # Best-effort loader; ignore errors
        pass


def main():
    ap = argparse.ArgumentParser(description="RAGAS evaluation for LocalCat memory retrieval")
    ap.add_argument("--cases", default="server/tmp/cases_eval.jsonl", help="Cases JSONL with setup/query/gold")
    ap.add_argument("--out", default="", help="Output JSON path for RAGAS scores")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of cases")
    ap.add_argument("--no-concurrency", action="store_true", help="Evaluate sequentially (no parallel workers)")
    ap.add_argument("--export-human", default="", help="Path to export JSONL for human eval (question, contexts, reference)")
    ap.add_argument("--env", nargs="*", default=[], help="Environment overrides KEY=VAL")
    ap.add_argument("--llm-base", default="", help="OpenAI-compatible base URL (e.g., LM Studio http://localhost:1234/v1)")
    ap.add_argument("--llm-model", default="", help="Model name for the OpenAI-compatible LLM")
    ap.add_argument("--llm-api-key", default="not-needed", help="API key for OpenAI-compatible LLM (LM Studio ignores it)")
    ap.add_argument("--use-llm", action="store_true", help="Explicitly enable LLM for RAGAS (off by default to avoid flooding)")
    # Load server/.env first so CLI defaults can pick them up
    _load_dotenv_env(str(Path(__file__).resolve().parents[2] / "server" / ".env"))

    args = ap.parse_args()

    # Force safer tokenizer mode for forks
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Only backfill LLM settings from env if explicitly requested
    if args.use_llm:
        if not args.llm_base:
            args.llm_base = os.getenv("LLM_BASE_URL") or os.getenv("VOICE_AGENT_LLM_BASE_URL") or ""
        if not args.llm_model:
            args.llm_model = os.getenv("VOICE_AGENT_LLM_MODEL") or os.getenv("LLM_MODEL") or ""
        if not args.llm_api_key:
            args.llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("VOICE_AGENT_LLM_API_KEY") or "not-needed"

    prev = set_env(args.env)
    try:
        cases = load_cases(args.cases)
        if int(args.limit or 0) > 0:
            cases = cases[: int(args.limit)]
        if not cases:
            raise SystemExit("No cases loaded")

        # lazy import ragas deps after potential env overrides
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
        )
        # Optional metrics needing LLM judge
        answer_rel = None
        faithful = None
        if args.use_llm:
            try:
                from ragas.metrics import answer_relevancy as answer_rel
            except Exception:
                try:
                    from ragas.metrics import answer_relevance as answer_rel  # alt name in some versions
                except Exception:
                    answer_rel = None
            try:
                from ragas.metrics import faithfulness as faithful
            except Exception:
                faithful = None

        # Build list of evaluation samples
        samples: List[Dict[str, Any]] = []

        for case in cases:
            hot = build_hotmemory()
            sid = f"sess-{case.get('id', 'x')}"
            for i, text in enumerate(case.get("setup", [])):
                hot.process_turn(text, sid, i)
            hot.store.flush_if_needed(max_ops=1)

            bullets = hot.retrieve_bullets(case.get("query", ""), read_only=True)
            contexts = normalize_contexts(bullets)

            # For retrieval-only metrics, we can set 'answer' as ground truth text
            # to keep the schema; RAGAS will ignore 'answer' for context metrics.
            gt_list = case.get("gold") or []
            ground_truth = ", ".join(gt_list) if gt_list else ""

            row = {
                "question": case.get("query", ""),
                "contexts": contexts,
                "answer": ground_truth,
            }
            # RAGAS >=0.1.15 expects 'reference' for ground truth text
            if ground_truth:
                row["reference"] = ground_truth
            samples.append(row)

        dataset = Dataset.from_list(samples)

        try:
            # Avoid OpenAI dependencies by default to prevent flooding LM Studio
            evaluator_llm = None
            if args.use_llm and args.llm_base:
                # Route OpenAI client (if used internally) to local server
                os.environ["OPENAI_BASE_URL"] = args.llm_base
                os.environ["OPENAI_API_KEY"] = args.llm_api_key or "not-needed"
                from langchain_community.chat_models import ChatOpenAI
                evaluator_llm = ChatOpenAI(
                    base_url=args.llm_base,
                    api_key=args.llm_api_key or "not-needed",
                    model=args.llm_model or "local-model",
                    temperature=0.0,
                )

            # Ensure OpenAI envs reflect target base; preserve existing API key unless explicitly provided
            if args.llm_base:
                os.environ["OPENAI_BASE_URL"] = args.llm_base
                if args.llm_api_key and args.llm_api_key != "not-needed":
                    os.environ["OPENAI_API_KEY"] = args.llm_api_key

            # Local embeddings via sentence-transformers to avoid OpenAI
            from langchain_community.embeddings import HuggingFaceEmbeddings
            hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            print(f"Using LLM base: {args.llm_base or 'none'} model: {args.llm_model or 'n/a'} use_llm={args.use_llm}")
            metrics_list = [context_precision, context_recall]
            if evaluator_llm is not None:
                if answer_rel is not None:
                    metrics_list.append(answer_rel)
                if faithful is not None:
                    metrics_list.append(faithful)
            if args.no_concurrency:
                # Sequential evaluation: one sample at a time to avoid concurrency issues
                totals: Dict[str, float] = {}
                count = 0
                for row in samples:
                    ds_single = Dataset.from_list([row])
                    try:
                        res = evaluate(ds_single, metrics=metrics_list, llm=evaluator_llm, embeddings=hf_embeddings)
                        # Convert result to dict
                        try:
                            d = {k: float(v) for k, v in dict(res).items()}
                        except Exception:
                            d = res.to_dict() if hasattr(res, 'to_dict') else {}
                        for k, v in d.items():
                            try:
                                totals[k] = totals.get(k, 0.0) + float(v)
                            except Exception:
                                pass
                        count += 1
                    except Exception as e:
                        print(f"Sequential eval failed on a sample: {e}")
                        continue
                # Average
                class _Obj:
                    def __init__(self, d): self._d=d
                    def __iter__(self): return iter(self._d)
                    def items(self): return self._d.items()
                if count == 0:
                    results = _Obj({k: None for k in ["context_precision","context_recall","answer_relevancy","faithfulness"]})
                else:
                    results = _Obj({k: (totals.get(k, 0.0)/count) for k in totals.keys()})
            else:
                results = evaluate(dataset, metrics=metrics_list, llm=evaluator_llm, embeddings=hf_embeddings)
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            if args.out:
                Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump({"error": str(e)}, f, ensure_ascii=False, indent=2)
                print(f"Saved error to {args.out}")
            return

        # Convert to plain dict
        try:
            out = {k: float(v) for k, v in dict(results).items()}
        except Exception:
            try:
                out = results.to_dict()
            except Exception:
                out = {"context_precision": None, "context_recall": None}

        print("RAGAS results:")
        for k, v in out.items():
            try:
                print(f"  {k}: {float(v):.3f}")
            except Exception:
                print(f"  {k}: {v}")

        if args.export_human:
            try:
                Path(args.export_human).parent.mkdir(parents=True, exist_ok=True)
                with open(args.export_human, "w", encoding="utf-8") as f:
                    for row in samples:
                        rec = {
                            "question": row.get("question"),
                            "reference": row.get("reference") or row.get("answer", ""),
                            "contexts": row.get("contexts", [])[:5],
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"Exported human-eval pack to {args.export_human}")
            except Exception as e:
                print(f"Failed to export human-eval pack: {e}")

        if args.out:
            Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {args.out}")

    finally:
        restore_env(prev)


if __name__ == "__main__":
    main()
