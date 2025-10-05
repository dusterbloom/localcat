#!/bin/bash
# A/B Testing Script for ASI1 Extraction Methods
# Usage: ./scripts/run_ab_test.sh <dataset_path> [complexity_threshold]

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [complexity_threshold]"
    echo "Example: $0 tests/data/yaml_eval_l1_en_medium.json 0.6"
    exit 1
fi

DATASET=$1
COMPLEXITY_THRESHOLD=${2:-0.6}

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset file not found: $DATASET"
    exit 1
fi

# Extract dataset name for output file
DATASET_NAME=$(basename "$DATASET" .json)
OUTPUT_DIR="results/ab_tests"
mkdir -p "$OUTPUT_DIR"

echo "=== ASI1 A/B Testing Runner ==="
echo "Dataset: $DATASET"
echo "Complexity Threshold: $COMPLEXITY_THRESHOLD"
echo ""

# Check LM Studio connection
echo "Checking LM Studio connection..."
LM_STUDIO_URL="${LLM_JUDGE_BASE_URL:-http://127.0.0.1:1234/v1}"
if curl -s -o /dev/null -w "%{http_code}" "${LM_STUDIO_URL}/models" | grep -q "200"; then
    echo "✓ LM Studio is running"
else
    echo "⚠ LM Studio not responding at $LM_STUDIO_URL"
    echo "  Hybrid extraction will fall back to YAML"
fi
echo ""

# Set environment for optimal performance
export SPACY_MODEL_EN=${SPACY_MODEL_EN:-en_core_web_trf}
export YAML_DENSITY_CAPS=${YAML_DENSITY_CAPS:-off}
export YAML_NOMINALS=${YAML_NOMINALS:-on}
export YAML_COREF=${YAML_COREF:-off}

# GraphJudge settings
export YAML_GRAPH_JUDGE=${YAML_GRAPH_JUDGE:-on}
export YAML_GRAPH_JUDGE_MODEL=${YAML_GRAPH_JUDGE_MODEL:-models/graph_judge.json}
export YAML_GRAPH_JUDGE_GRAY_BAND=${YAML_GRAPH_JUDGE_GRAY_BAND:-0.10}
export YAML_GRAPH_JUDGE_GRAYZONE_LOG=${YAML_GRAPH_JUDGE_GRAYZONE_LOG:-data/judge_grayzone.jsonl}

# Enable methods
export ENABLE_YAML_JUDGE=${ENABLE_YAML_JUDGE:-true}
export ENABLE_SLM=${ENABLE_SLM:-true}
export ENABLE_HYBRID=${ENABLE_HYBRID:-true}
export ENABLE_DSPY=${ENABLE_DSPY:-false}

# SLM settings
export SLM_REFINEMENT_ENABLED=${SLM_REFINEMENT_ENABLED:-true}
export SLM_MODEL_PATH=${SLM_MODEL_PATH:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}
export SLM_MAX_REFINEMENT_MS=${SLM_MAX_REFINEMENT_MS:-150}

# Hybrid settings
export HOTMEM_LLM_ASSISTED=${HOTMEM_LLM_ASSISTED:-true}
export HOTMEM_LLM_ASSISTED_MODEL=${HOTMEM_LLM_ASSISTED_MODEL:-llama-3.2-3b-instruct}
export HOTMEM_LLM_ASSISTED_BASE_URL=${HOTMEM_LLM_ASSISTED_BASE_URL:-$LM_STUDIO_URL}
export HOTMEM_LLM_ASSISTED_TIMEOUT_MS=${HOTMEM_LLM_ASSISTED_TIMEOUT_MS:-500}
export HOTMEM_COMPLEXITY_THRESHOLD=${HOTMEM_COMPLEXITY_THRESHOLD:-$COMPLEXITY_THRESHOLD}

# LM Studio settings (for judge and hybrid)
export LLM_JUDGE_BASE_URL=${LLM_JUDGE_BASE_URL:-http://127.0.0.1:1234/v1}
export LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL:-llama-3.2-3b-instruct}

# Print configuration
echo "Configuration:"
echo "  Model: $SPACY_MODEL_EN"
echo "  Methods: YAML=$ENABLE_YAML_JUDGE SLM=$ENABLE_SLM Hybrid=$ENABLE_HYBRID"
echo "  Judge: $YAML_GRAPH_JUDGE"
echo "  Complexity Threshold: $HOTMEM_COMPLEXITY_THRESHOLD"
echo ""

# Run the A/B test
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_ct${COMPLEXITY_THRESHOLD}_$(date +%Y%m%d_%H%M%S).json"
echo "Running A/B test..."

# Activate virtual environment if needed
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the test
python scripts/eval_extraction_ab.py \
    --dataset "$DATASET" \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
    --output "$OUTPUT_FILE"

# Check if gray-zone logging produced data
if [ -f "$YAML_GRAPH_JUDGE_GRAYZONE_LOG" ]; then
    GRAY_COUNT=$(wc -l < "$YAML_GRAPH_JUDGE_GRAYZONE_LOG" 2>/dev/null || echo 0)
    if [ "$GRAY_COUNT" -gt 0 ]; then
        echo ""
        echo "Gray-zone cases logged: $GRAY_COUNT"
        echo "  To label and retrain:"
        echo "  1. python -m scripts.llm_judge_labeler --log $YAML_GRAPH_JUDGE_GRAYZONE_LOG --out data/judge_labels.jsonl"
        echo "  2. python -m scripts.train_graph_judge_from_labels --labels data/judge_labels.jsonl --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --out models/graph_judge.json --keep_rate 0.35"
    fi
fi

echo ""
echo "✓ Results saved to: $OUTPUT_FILE"
echo ""

# If running complexity sweep
if [ "$3" = "--sweep" ]; then
    echo "Running complexity threshold sweep..."
    for threshold in 0.4 0.5 0.6 0.7; do
        echo "Testing threshold: $threshold"
        HOTMEM_COMPLEXITY_THRESHOLD=$threshold "$0" "$DATASET" "$threshold"
    done

    echo ""
    echo "Sweep complete! Results in: $OUTPUT_DIR"
fi