import dspy
from typing import List, Dict
import json

# Sample HotMem graph data (triples from dual_graph_architecture)
SAMPLE_TRIPLES = [
    {"subject": "Sarah", "predicate": "works_at", "object": "Google", "confidence": 0.9},
    {"subject": "Sarah", "predicate": "is", "object": "software_engineer", "confidence": 0.8},
    {"subject": "John", "predicate": "is_manager_of", "object": "Sarah", "confidence": 0.85},
    {"subject": "Sarah", "predicate": "married_to", "object": "Michael_Chen", "confidence": 0.95},
    {"subject": "Michael_Chen", "predicate": "is", "object": "cardiologist", "confidence": 0.9},
]

# Mock LM for feasibility test (no API key needed; simulate local LM)
class MockLM(dspy.LM):
    def __init__(self):
        super().__init__(model='mock_local_lm')
    
    def basic_request(self, prompt, **kwargs):
        # Mock response for TripleRetrieval: keyword-based retrieval from SAMPLE_TRIPLES
        query_lower = prompt.lower()
        relevant = []
        for triple in SAMPLE_TRIPLES:
            triple_str = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            if any(word in query_lower for word in ['sarah', 'family', 'married']):
                relevant.append(triple_str)
        mock_output = json.dumps({'relevant_triples': relevant[:2]})  # Limit to top 2
        return dspy.Prediction(output=mock_output)

dspy.settings.configure(lm=MockLM())

class TripleRetrieval(dspy.Signature):
    """Retrieve relevant entity triples from knowledge graph for a given query."""
    query: str = dspy.InputField()
    context: str = dspy.InputField(desc="Relevant triples from HotMem graph")
    relevant_triples: List[str] = dspy.OutputField(desc="List of relevant subject-predicate-object triples")

class HotMemRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.ChainOfThought(TripleRetrieval)

    def forward(self, query: str):
        # Mock retrieval: pass sample triples as context (in real: integrate with dual_graph.get_relationships)
        context = json.dumps(SAMPLE_TRIPLES)
        pred = self.retrieve(query=query, context=context)
        return pred

# Evaluation metric (simple: check if key triples are retrieved)
def evaluate_retrieval(pred, gold):
    retrieved = set(pred.relevant_triples)
    gold_set = set(gold)
    precision = len(retrieved & gold_set) / len(retrieved) if retrieved else 0
    recall = len(retrieved & gold_set) / len(gold_set) if gold_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# Test example
if __name__ == "__main__":
    rag = HotMemRAG()
    
    # Test query for family relationship (from roadmap example)
    query = "What is Sarah's family relationship?"
    gold_triples = ["Sarah married_to Michael_Chen"]  # Expected gold
    
    pred = rag(query=query)
    print("Query:", query)
    print("Retrieved triples:", pred.relevant_triples)
    
    metrics = evaluate_retrieval(pred, gold_triples)
    print("Evaluation metrics:", metrics)
    
    # Check if >80% F1 (towards 90% target)
    if metrics["f1"] > 0.8:
        print("Feasibility test passed: High retrieval accuracy")
    else:
        print("Feasibility test: Room for optimization")
</content>