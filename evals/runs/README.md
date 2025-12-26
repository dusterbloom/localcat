# Evaluation Run Archive

This directory stores timestamped evaluation runs for version control and comparison.

## Directory Structure
```
evals/runs/
  2025-11-06_143022_baseline_fe8415e/
    config.json          # Environment flags used
    results.json         # Micro-eval output (precision, latency)
    timing.ndjson        # Per-query instrumentation
    trace.ndjson         # Retrieval trace (candidates, scoring)
    commit.txt           # Git SHA
    metadata.json        # Run info (date, user, duration)
  2025-11-06_145533_verifier_jina_fe8415e/
    ...
  leaderboard.md         # Comparison table (auto-generated)
```

## Naming Convention
`{date}_{time}_{variant}_{git_sha}/`

## Usage
```bash
# Run eval with versioning
python server/tools/eval_runner.py --cases evals/ragas/test_queries.jsonl --variant baseline --save

# Quick run without archiving
python server/tools/eval_runner.py --cases cases.jsonl --variant baseline

# View leaderboard
cat evals/runs/leaderboard.md
```

## Comparison Workflow

1. **Run baseline**: `python server/tools/eval_runner.py --cases cases.jsonl --variant baseline --save`
2. **Make changes**: Edit code, tune config, etc.
3. **Run new variant**: `python server/tools/eval_runner.py --cases cases.jsonl --variant my_change --save`
4. **Compare results**: Check `evals/runs/leaderboard.md` for delta analysis

## Leaderboard Metrics

- **Precision@K**: Fraction of queries where retrieved context contains gold keywords
- **Has Gold**: Rate of queries with at least one gold keyword match
- **P95 Latency**: 95th percentile retrieval latency (SLO: <100ms)
- **Mean Latency**: Average retrieval time
- **Over Budget**: Whether P95 exceeds 100ms SLO

## Cleanup

Keep the most recent 50 runs, archive older ones:
```bash
cd evals/runs
ls -t | tail -n +51 | xargs -I {} mv {} archive/
```
