#!/usr/bin/env bash
set -euo pipefail

# Runner for LM Studio bakeoffs (easy + medium) with consolidated summary
# Usage: from repo root or server/:
#   cd server && bash scripts/run_lmstudio_bakeoff.sh

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Export only the SLM/ENABLE vars from .env if present
if [[ -f .env ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^\# ]] && continue
    if [[ "$line" =~ ^(SLM_|ENABLE_SLM|ENABLE_YAML_JUDGE|ENABLE_HYBRID|ENABLE_DSPY)= ]]; then
      export "$line"
    fi
  done < .env
fi

# Sensible defaults if not set
: "${SLM_REFINEMENT_ENABLED:=true}"
: "${SLM_FORCE:=true}"
: "${SLM_PROVIDER:=openai}"
: "${SLM_BASE_URL:=http://127.0.0.1:1234/v1}"
: "${SLM_API_KEY:=not-needed}"
: "${SLM_PRIMARY_MODEL:=lfm2-350m-extract}"
: "${SLM_SECONDARY_MODEL:=qwen2.5-coder-0.5b-instruct}"
: "${SLM_MODE:=fallback}"
: "${SLM_MAX_REFINEMENT_MS:=200}"
: "${SLM_TEMP:=0.1}"
: "${SLM_MAX_TOKENS:=120}"
: "${SLM_STRICT_JSON:=true}"
: "${SLM_RESPONSE_JSON_SCHEMA:=true}"

# Ensure local imports work
export PYTHONPATH=.

echo "Checking LM Studio at ${SLM_BASE_URL}..."
if ! curl -sS -m 2 "${SLM_BASE_URL}/models" > /dev/null; then
  echo "ERROR: LM Studio not responding at ${SLM_BASE_URL}. Start the server and load models." >&2
  exit 1
fi
echo "✓ LM Studio reachable"

EASY=tests/data/yaml_eval_l1_en_easy.json
MED=tests/data/yaml_eval_l1_en_medium.json
YAML=archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml

mkdir -p results

echo "\nRunning EASY set (primary=${SLM_PRIMARY_MODEL})..."
ENABLE_YAML_JUDGE=false ENABLE_HYBRID=false ENABLE_DSPY=false ENABLE_SLM=true \
SLM_REFINEMENT_ENABLED=${SLM_REFINEMENT_ENABLED} SLM_FORCE=${SLM_FORCE} SLM_PROVIDER=${SLM_PROVIDER} \
SLM_BASE_URL=${SLM_BASE_URL} SLM_API_KEY=${SLM_API_KEY} \
SLM_MODE=single SLM_PRIMARY_MODEL=${SLM_PRIMARY_MODEL} \
SLM_MAX_REFINEMENT_MS=${SLM_MAX_REFINEMENT_MS} SLM_TEMP=${SLM_TEMP} SLM_MAX_TOKENS=${SLM_MAX_TOKENS} \
SLM_STRICT_JSON=${SLM_STRICT_JSON} SLM_RESPONSE_JSON_SCHEMA=${SLM_RESPONSE_JSON_SCHEMA} \
python scripts/eval_extraction_ab.py --dataset "$EASY" --yaml "$YAML" --methods yaml yaml_slm \
  --output results/slm_bakeoff_easy_primary.json

echo "\nRunning EASY set (secondary=${SLM_SECONDARY_MODEL})..."
ENABLE_YAML_JUDGE=false ENABLE_HYBRID=false ENABLE_DSPY=false ENABLE_SLM=true \
SLM_REFINEMENT_ENABLED=${SLM_REFINEMENT_ENABLED} SLM_FORCE=${SLM_FORCE} SLM_PROVIDER=${SLM_PROVIDER} \
SLM_BASE_URL=${SLM_BASE_URL} SLM_API_KEY=${SLM_API_KEY} \
SLM_MODE=single SLM_PRIMARY_MODEL=${SLM_SECONDARY_MODEL} \
SLM_MAX_REFINEMENT_MS=${SLM_MAX_REFINEMENT_MS} SLM_TEMP=${SLM_TEMP} SLM_MAX_TOKENS=${SLM_MAX_TOKENS} \
SLM_STRICT_JSON=${SLM_STRICT_JSON} SLM_RESPONSE_JSON_SCHEMA=${SLM_RESPONSE_JSON_SCHEMA} \
python scripts/eval_extraction_ab.py --dataset "$EASY" --yaml "$YAML" --methods yaml yaml_slm \
  --output results/slm_bakeoff_easy_secondary.json

echo "\nRunning MEDIUM set (primary=${SLM_PRIMARY_MODEL})..."
ENABLE_YAML_JUDGE=false ENABLE_HYBRID=false ENABLE_DSPY=false ENABLE_SLM=true \
SLM_REFINEMENT_ENABLED=${SLM_REFINEMENT_ENABLED} SLM_FORCE=${SLM_FORCE} SLM_PROVIDER=${SLM_PROVIDER} \
SLM_BASE_URL=${SLM_BASE_URL} SLM_API_KEY=${SLM_API_KEY} \
SLM_MODE=single SLM_PRIMARY_MODEL=${SLM_PRIMARY_MODEL} \
SLM_MAX_REFINEMENT_MS=${SLM_MAX_REFINEMENT_MS} SLM_TEMP=${SLM_TEMP} SLM_MAX_TOKENS=${SLM_MAX_TOKENS} \
SLM_STRICT_JSON=${SLM_STRICT_JSON} SLM_RESPONSE_JSON_SCHEMA=${SLM_RESPONSE_JSON_SCHEMA} \
python scripts/eval_extraction_ab.py --dataset "$MED" --yaml "$YAML" --methods yaml yaml_slm \
  --output results/slm_bakeoff_medium_primary.json

echo "\nRunning MEDIUM set (secondary=${SLM_SECONDARY_MODEL})..."
ENABLE_YAML_JUDGE=false ENABLE_HYBRID=false ENABLE_DSPY=false ENABLE_SLM=true \
SLM_REFINEMENT_ENABLED=${SLM_REFINEMENT_ENABLED} SLM_FORCE=${SLM_FORCE} SLM_PROVIDER=${SLM_PROVIDER} \
SLM_BASE_URL=${SLM_BASE_URL} SLM_API_KEY=${SLM_API_KEY} \
SLM_MODE=single SLM_PRIMARY_MODEL=${SLM_SECONDARY_MODEL} \
SLM_MAX_REFINEMENT_MS=${SLM_MAX_REFINEMENT_MS} SLM_TEMP=${SLM_TEMP} SLM_MAX_TOKENS=${SLM_MAX_TOKENS} \
SLM_STRICT_JSON=${SLM_STRICT_JSON} SLM_RESPONSE_JSON_SCHEMA=${SLM_RESPONSE_JSON_SCHEMA} \
python scripts/eval_extraction_ab.py --dataset "$MED" --yaml "$YAML" --methods yaml yaml_slm \
  --output results/slm_bakeoff_medium_secondary.json

echo "\nConsolidating..."
python scripts/bakeoff_consolidated.py --output results/bakeoff_consolidated.json

echo "\nPolicy bakeoff (staged policy) – EASY & MEDIUM"
python scripts/eval_staged_policy.py --dataset "$EASY" --yaml "$YAML" --output results/staged_policy_easy.json || true
python scripts/eval_staged_policy.py --dataset "$MED" --yaml "$YAML" --output results/staged_policy_medium.json || true

echo "\nDone. See results in server/results/:"
ls -1 results/slm_bakeoff_*_*.json results/bakeoff_consolidated.json 2>/dev/null || true
ls -1 results/staged_policy_*.json 2>/dev/null || true
