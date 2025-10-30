# Quick Start: RAGAS Evaluation for LocalCat

**Time to integrate:** 1 day
**Difficulty:** Low
**Best for:** Immediate memory quality metrics

---

## What is RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) is the industry-standard framework for evaluating RAG systems. It provides reference-free evaluation metrics specifically designed for memory/retrieval systems.

**Key metrics for LocalCat:**
- **Context Precision**: Are retrieved memories relevant to the query?
- **Context Recall**: Did we retrieve all relevant memories?
- **Faithfulness**: Does the LLM response stick to retrieved context? (hallucination detection)
- **Answer Relevance**: Does the answer actually address the question?

---

## Installation

### Option 1: Using uv (Recommended)
```bash
cd /Users/peppi/Dev/localcat/server
uv pip install ragas langchain langchain-community
```

### Option 2: Using pip
```bash
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate
pip install ragas langchain langchain-community
```

---

## LocalCat Integration Strategy

RAGAS needs to evaluate your memory retrieval pipeline. Here's how we connect it:

```
User Query → LocalCat Memory Retrieval → Retrieved Context + LLM Response → RAGAS Metrics
```

### Architecture Integration Points

1. **Memory Retrieval**: `core.memory.retrieval.MemoryRetrieval.retrieve_relevant()`
2. **Context Formatting**: Convert retrieved memories to text for RAGAS
3. **LLM Response**: Use DirectMLXLLM or LM Studio response
4. **Evaluation**: Feed into RAGAS for scoring

---

## Implementation

### Step 1: Create Evaluation Script

Create `/Users/peppi/Dev/localcat/evals/scripts/evaluate_ragas.py`:

```python
#!/usr/bin/env python3
"""
RAGAS evaluation for LocalCat's memory system
"""

import sys
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevance
)
from datasets import Dataset
from langchain_community.chat_models import ChatOpenAI

# Import LocalCat components
from core.memory.memory_store import MemoryStore
from core.memory.retrieval import MemoryRetrieval
from core.llm.direct_mlx_llm import DirectMLXLLM


def format_retrieved_context(retrieved_memories):
    """Convert LocalCat's retrieved memories to text context"""
    context_parts = []

    for memory in retrieved_memories:
        # Handle different memory types from LocalCat
        if 'src' in memory and 'rel' in memory and 'dst' in memory:
            # Graph memory: "Alice graduated_from MIT"
            context_parts.append(f"{memory['src']} {memory['rel']} {memory['dst']}")
        elif 'text' in memory:
            # Conversational memory
            context_parts.append(memory['text'])
        elif 'content' in memory:
            # Semantic memory
            context_parts.append(memory['content'])

    return "\n".join(context_parts)


def evaluate_localcat_memory(
    test_queries,
    db_path="/Users/peppi/Library/Application Support/LocalCat/data/memory.db",
    session_id="ragas_eval",
    slot="general",
    model_endpoint="http://localhost:1234/v1"  # LM Studio default
):
    """
    Evaluate LocalCat's memory retrieval using RAGAS

    Args:
        test_queries: List of dicts with 'question' and 'ground_truth' (optional)
        db_path: Path to LocalCat's memory database
        session_id: Session ID for retrieval
        slot: Slot for slot-aware retrieval
        model_endpoint: LM Studio or OpenAI-compatible endpoint
    """

    # Initialize LocalCat memory system
    memory_store = MemoryStore(db_path=db_path, session_id=session_id)
    retrieval = MemoryRetrieval(memory_store)
    llm = DirectMLXLLM(model="mlx-community/gemma-3n-4b")

    # Initialize RAGAS evaluator LLM (using local LM Studio)
    evaluator_llm = ChatOpenAI(
        model="gemma3n-4b",  # Or whatever model you have in LM Studio
        openai_api_base=model_endpoint,
        openai_api_key="dummy"  # LM Studio doesn't need real key
    )

    results = []

    for query_data in test_queries:
        question = query_data['question']

        # Step 1: Retrieve from LocalCat memory
        retrieved = retrieval.retrieve_relevant(
            query=question,
            slot=slot,
            k=10,  # Top 10 memories
            session_id=session_id
        )

        # Step 2: Format retrieved context
        context = format_retrieved_context(retrieved)

        # Step 3: Generate answer using LocalCat's LLM
        prompt = f"""Context: {context}

Question: {question}

Answer based only on the context provided:"""

        answer = llm.generate(prompt, max_tokens=200)

        # Step 4: Prepare for RAGAS evaluation
        results.append({
            'question': question,
            'contexts': [context],  # RAGAS expects list of contexts
            'answer': answer,
            'ground_truth': query_data.get('ground_truth', '')  # Optional
        })

        print(f"✓ Processed: {question[:60]}...")

    # Convert to RAGAS dataset format
    dataset = Dataset.from_list(results)

    # Evaluate with RAGAS metrics
    print("\n🔍 Running RAGAS evaluation...")

    evaluation_result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevance
        ],
        llm=evaluator_llm
    )

    return evaluation_result


# Example test queries
TEST_QUERIES = [
    {
        'question': "What is my favorite color?",
        'ground_truth': "Your favorite color is yellow."  # Optional
    },
    {
        'question': "What programming language do I prefer?",
        'ground_truth': "You prefer Python."
    },
    {
        'question': "Where did I graduate from?",
        'ground_truth': "You graduated from MIT."
    },
    {
        'question': "What project am I working on?",
        'ground_truth': "You're working on a voice AI project."
    }
]


if __name__ == "__main__":
    # Run evaluation
    results = evaluate_localcat_memory(TEST_QUERIES)

    print("\n" + "=" * 60)
    print("📊 RAGAS Evaluation Results")
    print("=" * 60)

    for metric, score in results.items():
        print(f"{metric:25s}: {score:.3f}")

    print("\n💡 Interpretation:")
    print("  - Context Precision > 0.8: Excellent retrieval relevance")
    print("  - Context Recall > 0.8: Good coverage of facts")
    print("  - Faithfulness > 0.9: Minimal hallucination")
    print("  - Answer Relevance > 0.8: Responses on-topic")
```

---

## Step 2: Create Test Data from Production

Instead of synthetic queries, use real conversations from your production logs:

```python
#!/usr/bin/env python3
"""
Extract real test queries from LocalCat production logs
"""

import re
from pathlib import Path

def extract_queries_from_logs(
    log_path="/Users/peppi/Library/Logs/LocalCat/server.log"
):
    """Extract real user queries and bot responses from logs"""

    queries = []

    with open(log_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Look for user questions
        if "User:" in line or "user:" in line:
            question = line.split(":", 2)[-1].strip()

            # Look for bot response in next few lines
            answer = None
            for j in range(i+1, min(i+10, len(lines))):
                if "Bot:" in lines[j] or "Assistant:" in lines[j]:
                    answer = lines[j].split(":", 2)[-1].strip()
                    break

            if answer:
                queries.append({
                    'question': question,
                    'ground_truth': answer
                })

    return queries

# Use real production data
real_queries = extract_queries_from_logs()
print(f"Extracted {len(real_queries)} real queries from production logs")
```

---

## Step 3: Run Evaluation

```bash
cd /Users/peppi/Dev/localcat/evals/scripts
python evaluate_ragas.py
```

Expected output:
```
✓ Processed: What is my favorite color?...
✓ Processed: What programming language do I prefer?...
✓ Processed: Where did I graduate from?...
✓ Processed: What project am I working on?...

🔍 Running RAGAS evaluation...

============================================================
📊 RAGAS Evaluation Results
============================================================
context_precision        : 0.850
context_recall           : 0.920
faithfulness             : 0.880
answer_relevance         : 0.910

💡 Interpretation:
  - Context Precision > 0.8: Excellent retrieval relevance
  - Context Recall > 0.8: Good coverage of facts
  - Faithfulness > 0.9: Minimal hallucination
  - Answer Relevance > 0.8: Responses on-topic
```

---

## Metrics Interpretation for LocalCat

### Context Precision (0-1, higher better)
**What it measures**: Are the retrieved memories relevant to the query?

**For LocalCat**: Tests your slot-aware retrieval and composite scoring (wsrc, wconf, wrec, wuse, wsim, wpro, wdiv).

- **0.9-1.0**: Excellent - slot routing working perfectly
- **0.7-0.9**: Good - some irrelevant memories retrieved
- **<0.7**: Needs work - check composite scoring weights

### Context Recall (0-1, higher better)
**What it measures**: Did we retrieve all relevant memories?

**For LocalCat**: Tests multi-source retrieval (graph + convo + summary + semantic).

- **0.9-1.0**: Excellent - all sources contributing
- **0.7-0.9**: Good - most facts retrieved
- **<0.7**: Missing memories - check FTS indexing

### Faithfulness (0-1, higher better)
**What it measures**: Does the LLM stick to retrieved context?

**For LocalCat**: Tests if DirectMLXLLM hallucinates or invents facts.

- **0.95-1.0**: Excellent - no hallucination
- **0.85-0.95**: Good - minor extrapolation
- **<0.85**: Concerning - LLM adding unsupported facts

### Answer Relevance (0-1, higher better)
**What it measures**: Does the answer address the question?

**For LocalCat**: Tests overall system quality (retrieval + generation).

- **0.9-1.0**: Excellent - answers on-point
- **0.7-0.9**: Good - answers mostly relevant
- **<0.7**: Needs work - check prompt engineering

---

## Advanced: Evaluating Specific LocalCat Features

### Test Slot-Aware Retrieval

```python
# Test that slot routing prevents cross-contamination
SLOT_TEST_QUERIES = [
    {
        'question': "What is my favorite color?",
        'slot': 'favorite_color',
        'expected_no_match_from': 'favorite_number'  # Should not retrieve from this slot
    },
    {
        'question': "What is my favorite number?",
        'slot': 'favorite_number',
        'expected_no_match_from': 'favorite_color'
    }
]
```

### Test Composite Scoring

```python
# Evaluate different scoring weight configurations
SCORING_CONFIGS = [
    {'wsrc': 0.3, 'wconf': 0.3, 'wrec': 0.2, 'wuse': 0.1, 'wsim': 0.1, 'wpro': 0.0, 'wdiv': 0.0},
    {'wsrc': 0.4, 'wconf': 0.2, 'wrec': 0.2, 'wuse': 0.1, 'wsim': 0.1, 'wpro': 0.0, 'wdiv': 0.0},
    # ... test different weights
]

for config in SCORING_CONFIGS:
    # Update composite scorer weights
    # Run RAGAS evaluation
    # Compare results
```

### Test Cross-Session Retrieval

```python
# Test that memories persist across sessions
SESSION_TEST = [
    # Session 1: Memorize
    {'session_id': 'session_1', 'action': 'memorize', 'text': 'My name is Alice'},
    # Session 2: Recall
    {'session_id': 'session_2', 'action': 'query', 'question': 'What is my name?'}
]
```

---

## Continuous Evaluation

### Option 1: Pre-commit Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Run memory evaluation before commits

cd /Users/peppi/Dev/localcat/evals/scripts
python evaluate_ragas.py --quick

# Fail if scores drop below threshold
if [ $? -ne 0 ]; then
    echo "❌ Memory evaluation failed - scores below threshold"
    exit 1
fi
```

### Option 2: Periodic Evaluation

Run nightly evaluations:
```bash
# Add to crontab
0 2 * * * cd /Users/peppi/Dev/localcat/evals/scripts && python evaluate_ragas.py --save-results
```

---

## Tracking Improvements

Create a simple dashboard to track metrics over time:

```python
import json
from datetime import datetime
import matplotlib.pyplot as plt

def save_evaluation_results(results, output_dir="./outputs/ragas"):
    """Save evaluation results with timestamp"""
    timestamp = datetime.now().isoformat()

    result_data = {
        'timestamp': timestamp,
        'metrics': dict(results)
    }

    # Append to history
    history_file = f"{output_dir}/evaluation_history.jsonl"
    with open(history_file, 'a') as f:
        f.write(json.dumps(result_data) + '\n')

    print(f"✅ Results saved to {history_file}")

def plot_metrics_over_time(history_file="./outputs/ragas/evaluation_history.jsonl"):
    """Plot how metrics change over time"""

    data = []
    with open(history_file) as f:
        for line in f:
            data.append(json.loads(line))

    # Extract timestamps and metrics
    timestamps = [d['timestamp'] for d in data]
    precision = [d['metrics']['context_precision'] for d in data]
    recall = [d['metrics']['context_recall'] for d in data]
    faithfulness = [d['metrics']['faithfulness'] for d in data]
    relevance = [d['metrics']['answer_relevance'] for d in data]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, precision, label='Context Precision', marker='o')
    plt.plot(timestamps, recall, label='Context Recall', marker='s')
    plt.plot(timestamps, faithfulness, label='Faithfulness', marker='^')
    plt.plot(timestamps, relevance, label='Answer Relevance', marker='d')

    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('LocalCat Memory Quality Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./outputs/ragas/metrics_over_time.png')
    print("📈 Metrics plot saved to ./outputs/ragas/metrics_over_time.png")
```

---

## Troubleshooting

### Issue: "No retrieved context"
**Solution**: Check that your memory database has indexed conversations:
```bash
sqlite3 "/Users/peppi/Library/Application Support/LocalCat/data/memory.db" "SELECT COUNT(*) FROM memories;"
```

### Issue: "Low context_recall scores"
**Solution**: Increase `k` parameter in retrieval:
```python
retrieved = retrieval.retrieve_relevant(query=question, slot=slot, k=20)  # Increased from 10
```

### Issue: "RAGAS evaluation is slow"
**Solution**: Use smaller LLM for RAGAS evaluation or batch queries:
```python
evaluator_llm = ChatOpenAI(model="gemma3n-4b")  # Smaller, faster model
```

### Issue: "LM Studio connection refused"
**Solution**: Check that LM Studio is running with server enabled:
```bash
curl http://localhost:1234/v1/models
```

---

## Next Steps

1. ✅ You now have RAGAS integrated with LocalCat
2. 📊 Run initial evaluation to establish baseline
3. 🔄 Set up continuous evaluation (pre-commit or nightly)
4. 📈 Track improvements over time
5. 🎯 Move to [03-comprehensive-strategy.md](./03-comprehensive-strategy.md) to add BEIR and RagaAI Catalyst

---

## Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [LocalCat Memory Architecture](../docs/02-architecture/memory-system-map.md)
- [Metrics Interpretation Guide](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
