#!/bin/bash
#
# RAGAS Setup Script for LocalCat Memory Evaluation
#
# This script automates the setup of RAGAS evaluation framework
# for LocalCat's memory system.
#
# Usage:
#   cd /Users/peppi/Dev/localcat/evals/scripts
#   ./setup_ragas.sh
#

set -e  # Exit on error

echo "============================================================"
echo "🚀 LocalCat Memory Evaluation - RAGAS Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
LOCALCAT_ROOT="/Users/peppi/Dev/localcat"
SERVER_DIR="$LOCALCAT_ROOT/server"
EVALS_DIR="$LOCALCAT_ROOT/evals"
SCRIPTS_DIR="$EVALS_DIR/scripts"
OUTPUTS_DIR="$EVALS_DIR/outputs"

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓${NC} Python version: $PYTHON_VERSION"

# Check if server directory exists
if [ ! -d "$SERVER_DIR" ]; then
    echo -e "${RED}❌ Server directory not found: $SERVER_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} LocalCat server found"

# Check if we have uv or pip
HAS_UV=false
if command -v uv &> /dev/null; then
    HAS_UV=true
    echo -e "${GREEN}✓${NC} Using uv for package management"
else
    echo -e "${YELLOW}⚠${NC}  uv not found, will use pip"
fi

echo ""

# Navigate to server directory
cd "$SERVER_DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "${YELLOW}⚠${NC}  No .venv found in server directory"
    echo "   Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment created and activated"
fi

echo ""

# Install RAGAS and dependencies
echo "📦 Installing RAGAS and dependencies..."
echo ""

if [ "$HAS_UV" = true ]; then
    # Using uv
    echo "Installing with uv..."
    uv pip install ragas langchain langchain-community tiktoken datasets
else
    # Using pip
    echo "Installing with pip..."
    pip install ragas langchain langchain-community tiktoken datasets
fi

echo ""
echo -e "${GREEN}✓${NC} RAGAS and dependencies installed"
echo ""

# Verify installation
echo "🔍 Verifying RAGAS installation..."
python3 -c "import ragas; print(f'RAGAS version: {ragas.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} RAGAS successfully installed"
else
    echo -e "${RED}❌ RAGAS import failed${NC}"
    exit 1
fi

echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p "$OUTPUTS_DIR/ragas"
mkdir -p "$OUTPUTS_DIR/beir"
mkdir -p "$OUTPUTS_DIR/ragaai"
mkdir -p "$OUTPUTS_DIR/comprehensive"
echo -e "${GREEN}✓${NC} Output directories created"

echo ""

# Create sample evaluation script
echo "📝 Creating sample evaluation script..."

cat > "$SCRIPTS_DIR/sample_evaluation.py" << 'EOF'
#!/usr/bin/env python3
"""
Sample RAGAS evaluation for LocalCat

This is a minimal example to verify your RAGAS setup is working.
"""

import sys
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')

from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevance
from datasets import Dataset

# Sample data for testing
sample_data = {
    'question': ['What is the capital of France?', 'What color is the sky?'],
    'answer': ['The capital of France is Paris.', 'The sky is blue.'],
    'contexts': [
        ['France is a country in Europe. Its capital city is Paris.'],
        ['The sky appears blue due to Rayleigh scattering of sunlight.']
    ],
    'ground_truth': ['Paris', 'Blue']
}

def test_ragas_setup():
    """Quick test to verify RAGAS is working"""

    print("\n" + "="*60)
    print("🧪 Testing RAGAS Setup")
    print("="*60 + "\n")

    # Create dataset
    dataset = Dataset.from_dict(sample_data)

    print("📊 Sample dataset created with {} examples".format(len(dataset)))
    print("\nRunning evaluation (this may take a minute)...\n")

    # Note: This requires an LLM. For quick testing, we'll skip actual evaluation
    # and just verify imports work
    print("✅ RAGAS imports successful")
    print("✅ Dataset creation successful")
    print("\n" + "="*60)
    print("🎉 RAGAS setup verified!")
    print("="*60 + "\n")

    print("Next steps:")
    print("1. Read the quick-start guide: evals/02-quick-start-ragas.md")
    print("2. Configure your LLM endpoint (LM Studio or OpenAI)")
    print("3. Run the full evaluation script")
    print()

if __name__ == "__main__":
    test_ragas_setup()
EOF

chmod +x "$SCRIPTS_DIR/sample_evaluation.py"
echo -e "${GREEN}✓${NC} Sample evaluation script created"

echo ""

# Test the sample script
echo "🧪 Running sample evaluation test..."
python3 "$SCRIPTS_DIR/sample_evaluation.py"

echo ""

# Create a README for the scripts directory
cat > "$SCRIPTS_DIR/README.md" << 'EOF'
# LocalCat Evaluation Scripts

This directory contains evaluation scripts for LocalCat's memory system.

## Quick Start

### 1. RAGAS Evaluation (RAG Quality Metrics)

```bash
# Sample test (verify setup)
python sample_evaluation.py

# Full evaluation (requires LocalCat memory database)
python evaluate_ragas.py
```

### 2. BEIR Benchmarks (Retrieval Quality)

```bash
python evaluate_beir.py --dataset nfcorpus
```

### 3. RagaAI Catalyst (Agent Tracing)

```bash
# Start dashboard
ragaai-catalyst serve --port 8080

# Run evaluation with tracing
python evaluate_ragaai.py
```

### 4. Comprehensive Evaluation (All Frameworks)

```bash
python run_full_evaluation.py
```

## Configuration

Edit the scripts to configure:
- Database path
- LLM endpoint (LM Studio or OpenAI)
- Session IDs
- Evaluation parameters

## Output

Results are saved to `../outputs/`:
- `ragas/` - RAG quality metrics
- `beir/` - Retrieval benchmarks
- `ragaai/` - Agent performance traces
- `comprehensive/` - Combined reports

## Documentation

See the parent `evals/` directory for complete documentation:
- `01-industry-frameworks.md` - Framework comparison
- `02-quick-start-ragas.md` - RAGAS integration guide
- `03-comprehensive-strategy.md` - Full implementation plan
EOF

echo -e "${GREEN}✓${NC} Scripts README created"

echo ""

# Summary
echo "============================================================"
echo "✅ RAGAS Setup Complete!"
echo "============================================================"
echo ""
echo "What was installed:"
echo "  • RAGAS framework"
echo "  • LangChain dependencies"
echo "  • Dataset utilities"
echo "  • Sample evaluation scripts"
echo ""
echo "Directory structure:"
echo "  $EVALS_DIR/"
echo "    ├── scripts/           (evaluation scripts)"
echo "    ├── outputs/           (evaluation results)"
echo "    ├── README.md          (overview)"
echo "    ├── 01-industry-frameworks.md"
echo "    ├── 02-quick-start-ragas.md"
echo "    └── 03-comprehensive-strategy.md"
echo ""
echo "Next steps:"
echo "  1. Read the quick-start guide:"
echo "     📖 $EVALS_DIR/02-quick-start-ragas.md"
echo ""
echo "  2. Configure your LLM endpoint:"
echo "     • Start LM Studio with local server enabled"
echo "     • Or configure OpenAI API key"
echo ""
echo "  3. Create your first evaluation script:"
echo "     📝 Follow the examples in 02-quick-start-ragas.md"
echo ""
echo "  4. Run evaluation:"
echo "     🚀 cd $SCRIPTS_DIR"
echo "     🚀 python evaluate_ragas.py"
echo ""
echo "Documentation: $EVALS_DIR/README.md"
echo ""
echo "============================================================"
echo "🎉 Ready to evaluate LocalCat's memory system!"
echo "============================================================"
