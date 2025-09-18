#!/usr/bin/env python3
"""
Test DSPy with Local LLM (LM Studio or Ollama)
Based on Stanford NLP DSPy documentation
"""

import dspy
import json
import os
from typing import List, Optional

# Sample knowledge graph data (simulating HotMem's dual graph)
KNOWLEDGE_GRAPH = [
    {"subject": "Sarah", "predicate": "works_at", "object": "Google", "confidence": 0.9},
    {"subject": "Sarah", "predicate": "is", "object": "software_engineer", "confidence": 0.8},
    {"subject": "Sarah", "predicate": "married_to", "object": "Michael_Chen", "confidence": 0.95},
    {"subject": "Michael_Chen", "predicate": "is", "object": "cardiologist", "confidence": 0.9},
    {"subject": "Michael_Chen", "predicate": "works_at", "object": "Seattle_General_Hospital", "confidence": 0.85},
    {"subject": "Sarah", "predicate": "has_child", "object": "Emma", "confidence": 0.95},
    {"subject": "Emma", "predicate": "age", "object": "5", "confidence": 0.8},
    {"subject": "Sarah", "predicate": "lives_in", "object": "Seattle", "confidence": 0.7},
]

def configure_local_llm():
    """Configure DSPy to use local LLM server (LM Studio or Ollama)"""

    # Check environment for LLM configuration
    api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:1234/v1")
    model = os.getenv("LLM_MODEL", "openai/local-model")  # OpenAI-compatible format

    print(f"Configuring DSPy with local LLM:")
    print(f"  API Base: {api_base}")
    print(f"  Model: {model}")

    # Configure DSPy with local LLM
    # Using openai/ prefix for OpenAI-compatible endpoints (as per DSPy docs)
    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key="dummy",  # Local servers don't need real API keys
        max_tokens=256,
        temperature=0.1,
        top_p=0.9
    )

    dspy.settings.configure(lm=lm)
    return lm

# Define DSPy Signatures (declarative input/output specifications)
class TripleRetrievalSignature(dspy.Signature):
    """Retrieve relevant triples from a knowledge graph based on a query"""

    query: str = dspy.InputField(desc="The user's question or query")
    graph_context: str = dspy.InputField(desc="Available triples from the knowledge graph in JSON format")
    relevant_triples: List[str] = dspy.OutputField(desc="List of relevant triples as 'subject predicate object' strings")

class QuestionAnsweringSignature(dspy.Signature):
    """Answer a question using knowledge graph context"""

    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Relevant facts from knowledge graph")
    answer: str = dspy.OutputField(desc="Concise answer based on the context")

class MultiHopReasoningSignature(dspy.Signature):
    """Perform multi-hop reasoning over graph relationships"""

    query: str = dspy.InputField(desc="Complex query requiring multiple reasoning steps")
    graph_facts: str = dspy.InputField(desc="Available facts from knowledge graph")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning process")
    answer: str = dspy.OutputField(desc="Final answer after reasoning")

# DSPy Modules (composable AI components)
class KnowledgeGraphRetriever(dspy.Module):
    """Retrieve relevant triples from knowledge graph"""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning
        self.retrieve = dspy.ChainOfThought(TripleRetrievalSignature)

    def forward(self, query: str, graph: Optional[List[dict]] = None):
        """Forward pass to retrieve relevant triples"""
        if graph is None:
            graph = KNOWLEDGE_GRAPH

        # Convert graph to JSON string for context
        graph_json = json.dumps(graph, indent=2)

        # Execute retrieval
        result = self.retrieve(query=query, graph_context=graph_json)
        return result

class GraphQA(dspy.Module):
    """Question answering over knowledge graph"""

    def __init__(self):
        super().__init__()
        self.retriever = KnowledgeGraphRetriever()
        self.answer = dspy.ChainOfThought(QuestionAnsweringSignature)

    def forward(self, question: str):
        """Answer question using graph context"""
        # First retrieve relevant triples
        retrieval_result = self.retriever(query=question)

        # Format context from retrieved triples
        context = "\n".join(retrieval_result.relevant_triples)

        # Generate answer
        answer_result = self.answer(question=question, context=context)

        return {
            "question": question,
            "retrieved_triples": retrieval_result.relevant_triples,
            "answer": answer_result.answer
        }

class MultiHopReasoner(dspy.Module):
    """Multi-hop reasoning over knowledge graph"""

    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(MultiHopReasoningSignature)

    def forward(self, query: str):
        """Perform multi-hop reasoning"""
        # Convert graph to readable format
        facts = []
        for triple in KNOWLEDGE_GRAPH:
            facts.append(f"{triple['subject']} {triple['predicate']} {triple['object']}")

        facts_text = "\n".join(facts)

        # Execute reasoning
        result = self.reason(query=query, graph_facts=facts_text)
        return result

def test_retrieval():
    """Test basic retrieval functionality"""
    print("\n=== Testing Triple Retrieval ===\n")

    retriever = KnowledgeGraphRetriever()

    queries = [
        "What is Sarah's family relationship?",
        "Where does Michael Chen work?",
        "How old is Emma?",
        "What is Sarah's profession?"
    ]

    for query in queries:
        print(f"Query: {query}")
        try:
            result = retriever(query=query)
            print(f"Retrieved triples:")
            for triple in result.relevant_triples[:3]:  # Show top 3
                print(f"  - {triple}")
            print()
        except Exception as e:
            print(f"  Error: {e}\n")

def test_question_answering():
    """Test question answering with retrieval"""
    print("\n=== Testing Question Answering ===\n")

    qa = GraphQA()

    questions = [
        "Who is Sarah married to and what is his profession?",
        "Where does Sarah's husband work?",
        "What is the age of Sarah's child?",
        "What city does Sarah live in?"
    ]

    for question in questions:
        print(f"Question: {question}")
        try:
            result = qa(question=question)
            print(f"Retrieved: {result['retrieved_triples'][:2]}")  # Show first 2
            print(f"Answer: {result['answer']}\n")
        except Exception as e:
            print(f"  Error: {e}\n")

def test_multihop_reasoning():
    """Test multi-hop reasoning capabilities"""
    print("\n=== Testing Multi-Hop Reasoning ===\n")

    reasoner = MultiHopReasoner()

    queries = [
        "What hospital does Emma's father work at?",
        "What is the profession of Sarah's spouse?",
        "In which city does Michael Chen's wife live?"
    ]

    for query in queries:
        print(f"Query: {query}")
        try:
            result = reasoner(query=query)
            print(f"Reasoning: {result.reasoning[:200]}...")  # Show first 200 chars
            print(f"Answer: {result.answer}\n")
        except Exception as e:
            print(f"  Error: {e}\n")

def test_optimization_example():
    """Demonstrate DSPy optimization capabilities"""
    print("\n=== Testing DSPy Optimization ===\n")

    # Create training examples for optimization
    trainset = [
        dspy.Example(
            query="What is Sarah's occupation?",
            relevant_triples=["Sarah is software_engineer", "Sarah works_at Google"]
        ).with_inputs("query"),
        dspy.Example(
            query="Who is married to Sarah?",
            relevant_triples=["Sarah married_to Michael_Chen"]
        ).with_inputs("query"),
        dspy.Example(
            query="What is Michael's job?",
            relevant_triples=["Michael_Chen is cardiologist"]
        ).with_inputs("query"),
    ]

    print(f"Training set: {len(trainset)} examples")

    # Use BootstrapFewShot optimizer (good for small datasets)
    from dspy.teleprompt import BootstrapFewShot

    def metric(example, pred, trace=None):
        """Evaluation metric for optimization"""
        # Simple overlap metric
        predicted = set(pred.relevant_triples) if pred.relevant_triples else set()
        expected = set(example.relevant_triples)

        if not predicted:
            return 0.0

        overlap = len(predicted & expected) / len(predicted)
        return overlap

    # Create and compile optimized retriever
    retriever = KnowledgeGraphRetriever()

    try:
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
        optimized_retriever = optimizer.compile(retriever, trainset=trainset)
        print("✓ Optimization completed successfully")

        # Test optimized retriever
        test_query = "What is Sarah's family status?"
        result = optimized_retriever(query=test_query)
        print(f"\nOptimized retrieval for: '{test_query}'")
        print(f"Results: {result.relevant_triples[:2]}")
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("(This is expected if no LLM server is running)")

def main():
    """Main test function"""
    print("=" * 60)
    print("DSPy Local LLM Integration Test")
    print("=" * 60)

    # Configure local LLM
    try:
        configure_local_llm()
        print("✓ DSPy configured successfully\n")
    except Exception as e:
        print(f"Failed to configure DSPy: {e}")
        print("\nPlease ensure you have a local LLM server running:")
        print("  - LM Studio: Start server in Developer tab")
        print("  - Ollama: ollama serve")
        print("  - Set LLM_API_BASE environment variable if not using default")
        return

    # Run tests
    try:
        test_retrieval()
        test_question_answering()
        test_multihop_reasoning()
        # test_optimization_example()  # Uncomment to test optimization

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during tests: {e}")
        print("\nTroubleshooting:")
        print("1. Check if LLM server is running")
        print("2. Verify API endpoint is accessible")
        print("3. Ensure model is loaded in server")

if __name__ == "__main__":
    main()