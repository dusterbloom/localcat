#!/usr/bin/env python3
"""
Compare local LLM judge vs SOTA (Claude Sonnet) for eval reliability.

Measures inter-annotator agreement and identifies cases where local judge is unreliable.

Usage:
    python evals/scripts/llm_judge_comparison.py \\
        --cases evals/ragas/test_queries.jsonl \\
        --local-model gemma3n-4b \\
        --local-base http://localhost:1234/v1 \\
        --sota-model claude-sonnet-4.5 \\
        --out comparison.json
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

# Add server root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "server"))

# Check for required dependencies
try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Warning: anthropic package not available. Install with: pip install anthropic")
    anthropic = None


def judge_quality(query: str, context: List[str], gold: List[str], llm_client, model: str) -> Dict:
    """
    Use LLM to judge if context is relevant and helpful for query.

    Returns:
        {
            "relevant": true/false,
            "confidence": 0.0-1.0,
            "explanation": "...",
            "matches_gold": true/false
        }
    """
    context_str = "\n".join(f"- {c}" for c in context)
    gold_str = ", ".join(gold) if gold else "none"

    prompt = f"""Given the user query and retrieved context, assess if the context is relevant and helpful.

Query: {query}

Retrieved Context:
{context_str}

Gold Standard Keywords: {gold_str}

Provide your assessment:
1. Is the context relevant to the query? (yes/no)
2. Does it match the gold standard? (yes/no)
3. Confidence (0-100%)
4. Brief explanation

Format your response as JSON:
{{"relevant": true/false, "matches_gold": true/false, "confidence": 0.0-1.0, "explanation": "..."}}"""

    try:
        if hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions'):
            # OpenAI-compatible client (including local models)
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating information retrieval quality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
        elif anthropic and hasattr(llm_client, 'messages'):
            # Anthropic client
            response = llm_client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content[0].text.strip()
        else:
            raise ValueError(f"Unsupported client type: {type(llm_client)}")

        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
            else:
                # Fallback parsing
                result = json.loads(content)

            # Validate required fields
            required_fields = ["relevant", "matches_gold", "confidence", "explanation"]
            for field in required_fields:
                if field not in result:
                    result[field] = None

            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {content}")
            return {
                "relevant": None,
                "matches_gold": None,
                "confidence": 0.0,
                "explanation": f"JSON parse error: {e}",
                "raw_response": content
            }

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {
            "relevant": None,
            "matches_gold": None,
            "confidence": 0.0,
            "explanation": f"API error: {e}"
        }


def calculate_agreement(local_judgments: List[Dict], sota_judgments: List[Dict]) -> Dict:
    """Calculate inter-annotator agreement metrics."""
    if len(local_judgments) != len(sota_judgments):
        raise ValueError("Judgment lists must have same length")

    agreements = []
    disagreements = []
    relevance_agreements = []
    gold_agreements = []

    for i, (local, sota) in enumerate(zip(local_judgments, sota_judgments)):
        # Check if judgments are valid
        if local.get("relevant") is None or sota.get("relevant") is None:
            disagreements.append({
                "index": i,
                "local": local,
                "sota": sota,
                "reason": "invalid_judgment"
            })
            continue

        # Overall agreement (both relevant and gold match)
        overall_agree = (local["relevant"] == sota["relevant"] and
                        local["matches_gold"] == sota["matches_gold"])

        if overall_agree:
            agreements.append({
                "index": i,
                "local": local,
                "sota": sota
            })
        else:
            disagreements.append({
                "index": i,
                "local": local,
                "sota": sota,
                "reason": "disagreement"
            })

        # Component-wise agreements
        if local["relevant"] == sota["relevant"]:
            relevance_agreements.append(i)
        if local["matches_gold"] == sota["matches_gold"]:
            gold_agreements.append(i)

    total_cases = len(local_judgments)
    agreement_rate = len(agreements) / total_cases if total_cases > 0 else 0.0
    relevance_agreement_rate = len(relevance_agreements) / total_cases if total_cases > 0 else 0.0
    gold_agreement_rate = len(gold_agreements) / total_cases if total_cases > 0 else 0.0

    # Calculate confidence statistics
    local_confidences = [j.get("confidence", 0.0) for j in local_judgments if j.get("confidence") is not None]
    sota_confidences = [j.get("confidence", 0.0) for j in sota_judgments if j.get("confidence") is not None]

    return {
        "agreement_rate": agreement_rate,
        "relevance_agreement_rate": relevance_agreement_rate,
        "gold_agreement_rate": gold_agreement_rate,
        "total_cases": total_cases,
        "agreements": len(agreements),
        "disagreements": len(disagreements),
        "local_avg_confidence": sum(local_confidences) / len(local_confidences) if local_confidences else 0.0,
        "sota_avg_confidence": sum(sota_confidences) / len(sota_confidences) if sota_confidences else 0.0,
        "disagreement_cases": disagreements[:10],  # Sample of disagreements
        "agreement_cases": agreements[:5]  # Sample of agreements
    }


def load_cases(path: str) -> List[Dict]:
    """Load test cases from JSONL file."""
    cases = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    case = json.loads(line)
                    cases.append(case)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed case on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Test cases file not found: {path}")
        return []

    print(f"Loaded {len(cases)} test cases")
    return cases


def run_retrieval(case: Dict) -> List[str]:
    """Run retrieval for a case using HotMemory."""
    try:
        from server.core.memory.memory_hotpath import HotMemory
    except ImportError as e:
        print(f"Failed to import HotMemory: {e}")
        return []

    # Create fresh HotMemory instance
    try:
        memory = HotMemory()
    except Exception as e:
        print(f"Failed to create HotMemory instance: {e}")
        return []

    # Set up the case context
    for i, utterance in enumerate(case.get("setup", [])):
        try:
            memory.process_turn(utterance, f"test-session-{case.get('id', i)}", i)
        except Exception as e:
            print(f"Failed to process setup utterance {i}: {e}")
            continue

    # Run retrieval
    try:
        retrieved_bullets = memory.retrieve_bullets(case["query"], read_only=True)
        return retrieved_bullets
    except Exception as e:
        print(f"Failed to retrieve for query '{case['query']}': {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="LLM judge comparison")
    parser.add_argument("--cases", required=True, help="Path to test cases JSONL")
    parser.add_argument("--local-model", default="gemma3n-4b", help="Local model name")
    parser.add_argument("--local-base", default="http://localhost:1234/v1", help="Local model API base URL")
    parser.add_argument("--sota-model", default="claude-sonnet-4.5", help="SOTA model name")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, help="Limit number of cases to evaluate")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")

    args = parser.parse_args()

    # Load cases
    cases = load_cases(args.cases)
    if not cases:
        print("No cases loaded, exiting")
        return

    # Apply limit if specified
    if args.limit:
        cases = cases[:args.limit]
        print(f"Limited to {len(cases)} cases")

    # Check environment variables
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key and "claude" in args.sota_model.lower():
        print("Error: ANTHROPIC_API_KEY environment variable required for Claude models")
        return

    # Initialize clients
    print("Initializing LLM clients...")

    # Local model client (OpenAI-compatible)
    local_client = None
    try:
        local_client = OpenAI(
            base_url=args.local_base,
            api_key="local-key",  # Not used for local models
            timeout=args.timeout
        )
        print(f"✅ Local client initialized: {args.local_base}")
    except Exception as e:
        print(f"❌ Failed to initialize local client: {e}")
        return

    # SOTA model client
    sota_client = None
    try:
        if "claude" in args.sota_model.lower() or "anthropic" in args.sota_model.lower():
            if not anthropic:
                print("Error: anthropic package not available for Claude models")
                return
            sota_client = anthropic.Anthropic(
                api_key=anthropic_api_key,
                timeout=args.timeout
            )
            print(f"✅ Anthropic client initialized")
        else:
            # Assume OpenAI-compatible SOTA model
            sota_client = OpenAI(timeout=args.timeout)
            print(f"✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize SOTA client: {e}")
        return

    # Run evaluation
    local_judgments = []
    sota_judgments = []
    evaluation_results = []

    print(f"\n🧪 Evaluating {len(cases)} cases...")
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Processing case: {case.get('id', 'unknown')}")

        # Run retrieval
        retrieved = run_retrieval(case)
        if not retrieved:
            print(f"  ⚠️  No retrieval results, skipping case")
            continue

        # Get local judgment
        print(f"  🤖 Local judge evaluating...")
        local_j = judge_quality(case["query"], retrieved, case.get("gold", []), local_client, args.local_model)
        local_judgments.append(local_j)

        # Get SOTA judgment
        print(f"  🧠 SOTA judge evaluating...")
        sota_j = judge_quality(case["query"], retrieved, case.get("gold", []), sota_client, args.sota_model)
        sota_judgments.append(sota_j)

        # Store evaluation result
        evaluation_results.append({
            "case_id": case.get("id", f"case-{i}"),
            "query": case["query"],
            "gold": case.get("gold", []),
            "retrieved": retrieved,
            "local_judgment": local_j,
            "sota_judgment": sota_j
        })

        print(f"  📊 Local: {local_j.get('relevant', 'N/A')} | SOTA: {sota_j.get('relevant', 'N/A')}")

    # Calculate agreement
    print(f"\n📈 Calculating agreement metrics...")
    agreement = calculate_agreement(local_judgments, sota_judgments)

    # Prepare final results
    results = {
        "local_model": args.local_model,
        "local_base": args.local_base,
        "sota_model": args.sota_model,
        "cases_evaluated": len(evaluation_results),
        "agreement": agreement,
        "evaluation_results": evaluation_results,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎯 Results Summary:")
    print(f"   Cases evaluated: {results['cases_evaluated']}")
    print(f"   Overall agreement: {agreement['agreement_rate']:.1%}")
    print(f"   Relevance agreement: {agreement['relevance_agreement_rate']:.1%}")
    print(f"   Gold match agreement: {agreement['gold_agreement_rate']:.1%}")
    print(f"   Local avg confidence: {agreement['local_avg_confidence']:.2f}")
    print(f"   SOTA avg confidence: {agreement['sota_avg_confidence']:.2f}")

    if agreement['disagreements'] > 0:
        print(f"   Disagreements: {agreement['disagreements']} ({agreement['disagreements']/agreement['total_cases']:.1%})")

    print(f"\n📄 Results saved to: {output_path}")

    # Provide recommendation
    if agreement['agreement_rate'] >= 0.7:
        print(f"\n✅ Local judge is RELIABLE (agreement >= 70%)")
    elif agreement['agreement_rate'] >= 0.5:
        print(f"\n⚠️  Local judge is MODERATELY reliable (50-70% agreement)")
    else:
        print(f"\n❌ Local judge is UNRELIABLE (agreement < 50%)")
        print(f"   Consider using SOTA judge for final evaluation")


if __name__ == "__main__":
    main()