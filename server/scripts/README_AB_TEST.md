# A/B Testing for Graph Extraction

## Quick Start

Run the A/B test comparison:
```bash
cd server
./scripts/run_ab_test.sh
```

## Test Individual Methods

### 1. YAML Baseline (50ms)
```bash
python scripts/eval_extraction_ab.py \
    --dataset tests/data/complexity_test_set.json \
    --methods yaml
```

### 2. YAML with GraphJudge (60ms)
```bash
YAML_GRAPH_JUDGE=on \
python scripts/eval_extraction_ab.py \
    --dataset tests/data/complexity_test_set.json \
    --methods yaml_judge
```

### 3. Hybrid SpaCy+LLM (50-500ms adaptive)
```bash
HOTMEM_LLM_ASSISTED=true \
HOTMEM_LLM_ASSISTED_MODEL=qwen2.5-coder:3b \
python scripts/eval_extraction_ab.py \
    --dataset tests/data/complexity_test_set.json \
    --methods hybrid_spacy
```

### 4. YAML + SLM Refinement (200ms)
```bash
SLM_REFINEMENT_ENABLED=true \
SLM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit \
python scripts/eval_extraction_ab.py \
    --dataset tests/data/complexity_test_set.json \
    --methods yaml_slm
```

### 5. DSPy Enhanced (300-800ms)
```bash
ENABLE_DSPY=true \
DSPY_MODEL=openai/llama-3.2-3b-instruct \
DSPY_BASE_URL=http://localhost:1234/v1 \
python scripts/eval_extraction_ab.py \
    --dataset tests/data/complexity_test_set.json \
    --methods dspy
```

## Compare All Methods
```bash
python scripts/eval_extraction_ab.py \
    --dataset tests/data/complexity_test_set.json \
    --methods all \
    --output results/ab_test_results.json
```

## Test Datasets

- `tests/data/complexity_test_set.json` - Mixed complexity examples
- `tests/data/yaml_eval_l1_en_easy.json` - Simple sentences
- `tests/data/yaml_eval_l1_en_medium.json` - Medium complexity
- `tests/data/yaml_eval_l1_en_long.json` - Complex sentences

## Environment Variables

Configure in `.env`:

```bash
# Enable/disable methods
ENABLE_YAML_JUDGE=true
ENABLE_HYBRID=true
ENABLE_SLM=true
ENABLE_DSPY=false

# Hybrid settings
HOTMEM_COMPLEXITY_THRESHOLD=0.6
HOTMEM_LLM_ASSISTED_MODEL=qwen2.5-coder:3b

# SLM settings
SLM_REFINEMENT_ENABLED=false
SLM_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit

# GraphJudge
YAML_GRAPH_JUDGE=on
YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json
```

## Expected Results

| Method | Simple F1 | Complex F1 | P95 Latency |
|--------|-----------|------------|-------------|
| YAML | 0.85 | 0.40 | 50ms |
| YAML+Judge | 0.80 | 0.45 | 60ms |
| Hybrid | 0.88 | 0.70 | 300ms |
| YAML+SLM | 0.82 | 0.65 | 200ms |
| DSPy | 0.92 | 0.85 | 800ms |

## Integration with LocalCat

For voice interaction, use complexity-based routing:
- Simple sentences (<0.4 complexity) → YAML
- Medium sentences (0.4-0.7) → YAML+SLM
- Complex sentences (>0.7) → Hybrid or DSPy

Target: <200ms p95 latency with >70% F1 accuracy