"""
Evaluation framework for confidence scoring strategies

Provides tools to:
1. Build evaluation datasets from stored conversations
2. Measure confidence calibration quality
3. Compare different confidence strategies
"""

import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from .memory_store import MemoryStore
from .confidence_strategy import ConfidenceStrategy, Edge, Context


@dataclass
class EvalExample:
    """Single evaluation example"""
    text: str
    fact: Tuple[str, str, str]  # (src, rel, dst)
    relation_type: str
    source_count: int
    reinforcements: int
    negations: int
    is_correct: bool
    current_confidence: float
    edge_id: str
    age_days: float


def build_eval_dataset(store: MemoryStore, limit: int = 500) -> List[EvalExample]:
    """
    Create evaluation dataset from stored conversations

    Uses heuristics to label ground truth:
    - Facts with pos>0 and neg=0 are likely correct
    - Facts still active after 30 days with weight>0.7 are likely correct
    - Facts with neg>0 are likely incorrect
    - Facts with very low weight are likely incorrect

    Args:
        store: MemoryStore with conversation history
        limit: Maximum examples to return

    Returns:
        List of labeled examples for evaluation
    """
    examples = []

    cur = store.sql.cursor()

    # Get all edges with their metadata
    edges = cur.execute("""
        SELECT e.id, e.src, e.rel, e.dst, e.pos, e.neg, e.weight, e.updated_at, e.status
        FROM edge e
        WHERE e.status = 1
        ORDER BY e.updated_at DESC
        LIMIT ?
    """, (limit,)).fetchall()

    for edge_id, src, rel, dst, pos, neg, weight, updated_at, status in edges:
        # Get source text from provenance
        provenance = store.get_edge_provenance(edge_id)
        if not provenance:
            continue  # Skip edges without provenance

        text = provenance[0][0]  # Most recent source text
        source_count = len(provenance)

        # Ground truth labeling heuristics
        is_correct = _infer_correctness(pos, neg, weight, updated_at)

        # Calculate age
        age_days = (time.time() - updated_at / 1000) / 86400

        examples.append(EvalExample(
            text=text,
            fact=(src, rel, dst),
            relation_type=rel,
            source_count=source_count,
            reinforcements=pos,
            negations=neg,
            is_correct=is_correct,
            current_confidence=weight,
            edge_id=edge_id,
            age_days=age_days
        ))

    return examples


def _infer_correctness(pos: int, neg: int, weight: float, updated_at: int) -> bool:
    """
    Infer whether a fact is correct using heuristics

    Rules:
    1. Facts reinforced without negation are likely correct
    2. Facts that survived >30 days with high confidence are likely correct
    3. Facts with negations are likely incorrect
    4. Facts with very low confidence are likely incorrect
    """
    # Strong negative signals
    if neg > 0:
        return False

    if weight < 0.3:
        return False

    # Strong positive signals
    if pos > 0 and neg == 0:
        return True

    # Facts that survived time are likely correct
    age_days = (time.time() - updated_at / 1000) / 86400
    if age_days > 30 and weight > 0.7:
        return True

    # Default: assume correct if no strong signals
    # (Conservative: prefer false negatives over false positives)
    return weight >= 0.5


def evaluate_confidence_calibration(
    strategy: ConfidenceStrategy,
    test_set: List[EvalExample],
    store: Optional[MemoryStore] = None
) -> Dict[str, float]:
    """
    Measure confidence calibration quality

    Calibration metrics:
    - MSE: Mean squared error between predicted confidence and correctness
    - MAE: Mean absolute error
    - Correlation: Pearson correlation between predictions and actuals
    - Accuracy@threshold: Classification accuracy at confidence threshold
    - ECE: Expected Calibration Error (binned calibration)

    Args:
        strategy: Confidence strategy to evaluate
        test_set: List of labeled examples
        store: Optional MemoryStore for context

    Returns:
        Dictionary of calibration metrics
    """
    if not test_set:
        return {
            'mse': 0.0,
            'mae': 0.0,
            'correlation': 0.0,
            'mean_confidence': 0.0,
            'accuracy_at_70': 0.0,
            'ece': 0.0,
            'count': 0
        }

    predictions = []
    actuals = []

    for example in test_set:
        # Create Edge object for strategy
        edge = Edge(
            src=example.fact[0],
            rel=example.fact[1],
            dst=example.fact[2],
            pos=example.reinforcements,
            neg=example.negations,
            updated_at=int((time.time() - example.age_days * 86400) * 1000),
            id=example.edge_id
        )

        context = Context(
            store=store,
            text=example.text
        )

        # Score confidence
        conf = strategy.score(edge, context)
        predictions.append(conf)
        actuals.append(1.0 if example.is_correct else 0.0)

    # Calculate metrics
    metrics = {}

    # Mean Squared Error
    mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
    metrics['mse'] = mse

    # Mean Absolute Error
    mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)
    metrics['mae'] = mae

    # Pearson correlation
    mean_pred = sum(predictions) / len(predictions)
    mean_actual = sum(actuals) / len(actuals)

    numerator = sum((p - mean_pred) * (a - mean_actual)
                   for p, a in zip(predictions, actuals))

    denom_pred = sum((p - mean_pred) ** 2 for p in predictions) ** 0.5
    denom_actual = sum((a - mean_actual) ** 2 for a in actuals) ** 0.5

    if denom_pred > 0 and denom_actual > 0:
        correlation = numerator / (denom_pred * denom_actual)
    else:
        correlation = 0.0

    metrics['correlation'] = correlation

    # Mean confidence
    metrics['mean_confidence'] = mean_pred

    # Accuracy at 0.7 threshold
    correct = sum(1 for p, a in zip(predictions, actuals)
                 if (p >= 0.7 and a == 1.0) or (p < 0.7 and a == 0.0))
    metrics['accuracy_at_70'] = correct / len(predictions)

    # Expected Calibration Error (10 bins)
    ece = _calculate_ece(predictions, actuals, n_bins=10)
    metrics['ece'] = ece

    # Metadata
    metrics['count'] = len(predictions)

    return metrics


def _calculate_ece(predictions: List[float], actuals: List[float], n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error

    ECE measures the difference between predicted confidence and actual accuracy
    across confidence bins.

    Lower ECE = better calibration
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    # Assign predictions to bins
    for pred, actual in zip(predictions, actuals):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1
        bin_correct[bin_idx] += actual
        bin_conf_sum[bin_idx] += pred

    # Calculate ECE
    ece = 0.0
    total = len(predictions)

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_acc = bin_correct[i] / bin_counts[i]
            bin_conf = bin_conf_sum[i] / bin_counts[i]
            weight = bin_counts[i] / total
            ece += weight * abs(bin_conf - bin_acc)

    return ece


def compare_strategies(
    strategies: Dict[str, ConfidenceStrategy],
    test_set: List[EvalExample],
    store: Optional[MemoryStore] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple confidence strategies

    Args:
        strategies: Dict of {name: strategy} to compare
        test_set: Evaluation examples
        store: Optional MemoryStore for context

    Returns:
        Dict of {name: metrics} for each strategy
    """
    results = {}

    for name, strategy in strategies.items():
        metrics = evaluate_confidence_calibration(strategy, test_set, store)
        results[name] = metrics

    return results


def print_evaluation_report(results: Dict[str, Dict[str, float]], title: str = "Evaluation Results"):
    """
    Pretty-print evaluation results

    Args:
        results: Dict of {strategy_name: metrics}
        title: Report title
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    if not results:
        print("No results to display")
        return

    # Find best strategy for each metric
    best_mse = min(results.items(), key=lambda x: x[1]['mse'])
    best_mae = min(results.items(), key=lambda x: x[1]['mae'])
    best_corr = max(results.items(), key=lambda x: x[1]['correlation'])
    best_acc = max(results.items(), key=lambda x: x[1]['accuracy_at_70'])
    best_ece = min(results.items(), key=lambda x: x[1]['ece'])

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Count:           {metrics['count']}")
        print(f"  MSE:             {metrics['mse']:.4f}" +
              (" ✓ BEST" if name == best_mse[0] else ""))
        print(f"  MAE:             {metrics['mae']:.4f}" +
              (" ✓ BEST" if name == best_mae[0] else ""))
        print(f"  Correlation:     {metrics['correlation']:.4f}" +
              (" ✓ BEST" if name == best_corr[0] else ""))
        print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
        print(f"  Accuracy@0.7:    {metrics['accuracy_at_70']:.4f}" +
              (" ✓ BEST" if name == best_acc[0] else ""))
        print(f"  ECE:             {metrics['ece']:.4f}" +
              (" ✓ BEST" if name == best_ece[0] else ""))

    print("\n" + "=" * 80)


def confidence_distribution(predictions: List[float], n_bins: int = 5) -> Dict[str, int]:
    """
    Analyze confidence distribution across bins

    Args:
        predictions: List of confidence scores
        n_bins: Number of bins (default 5: very-low, low, medium, high, very-high)

    Returns:
        Dict of {bin_label: count}
    """
    bins = {
        "[0.0-0.3)": 0,   # Very low
        "[0.3-0.5)": 0,   # Low
        "[0.5-0.7)": 0,   # Medium
        "[0.7-0.9)": 0,   # High
        "[0.9-1.0]": 0,   # Very high
    }

    for pred in predictions:
        if pred < 0.3:
            bins["[0.0-0.3)"] += 1
        elif pred < 0.5:
            bins["[0.3-0.5)"] += 1
        elif pred < 0.7:
            bins["[0.5-0.7)"] += 1
        elif pred < 0.9:
            bins["[0.7-0.9)"] += 1
        else:
            bins["[0.9-1.0]"] += 1

    return bins