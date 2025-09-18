#!/usr/bin/env python
"""
Test DSPy with SGLang backend for local model serving
"""

import os
import sys
import json
import dspy
import sglang as sgl
from typing import List, Dict, Optional
import subprocess
import time
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.components.ai.dspy_modules import (
    GraphBuilder,
    KnowledgeGraph,
    Entity,
    Relationship,
    DSPyFramework
)

def check_sglang_server(port: int = 30000) -> bool:
    """Check if SGLang server is running"""
    try:
        response = requests.get(f"http://localhost:{port}/health")
        return response.status_code == 200
    except:
        return False

def start_sglang_server(model_path: str = "Qwen/Qwen2.5-0.5B-Instruct", port: int = 30000):
    """Start SGLang server with a local model"""
    print(f"Starting SGLang server with model: {model_path}")

    # Command to start SGLang server
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--device", "mps",  # Use Metal Performance Shaders for Apple Silicon
        "--dtype", "float16",
        "--context-length", "4096",
        "--mem-fraction-static", "0.85"
    ]

    # Start server in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    print("Waiting for SGLang server to start...")
    max_wait = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if check_sglang_server(port):
            print("SGLang server is ready!")
            return process
        time.sleep(2)

    print("Timeout waiting for SGLang server")
    process.terminate()
    return None

def configure_dspy_with_sglang(port: int = 30000):
    """Configure DSPy to use SGLang backend"""

    # Configure DSPy with SGLang endpoint
    lm = dspy.LM(
        model="sglang/local",  # Model identifier for SGLang
        api_base=f"http://localhost:{port}/v1",  # SGLang OpenAI-compatible endpoint
        api_key="dummy",  # SGLang doesn't need API key for local models
        max_tokens=256,
        temperature=0.1,
        top_p=0.9
    )

    dspy.settings.configure(lm=lm)
    print(f"DSPy configured with SGLang at port {port}")
    return lm

def test_basic_extraction():
    """Test basic entity and relationship extraction"""
    print("\n=== Testing Basic Extraction ===")

    # Sample text
    text = """
    Sarah is married to Dr. Michael Chen, who works as a cardiologist at Seattle General Hospital.
    They have a daughter named Emma who is 5 years old. Sarah works at TechCorp as a senior engineer.
    The family lives in a beautiful house in Seattle.
    """

    # Create framework
    framework = DSPyFramework()

    # Extract graph
    print(f"Input text: {text[:100]}...")
    graph = framework.extract_graph(text)

    print(f"\nExtracted {len(graph.entities)} entities:")
    for entity in graph.entities[:5]:
        print(f"  - {entity.text} ({entity.label})")

    print(f"\nExtracted {len(graph.relationships)} relationships:")
    for rel in graph.relationships[:5]:
        print(f"  - {rel.subject} --[{rel.predicate}]--> {rel.object} (conf: {rel.confidence:.2f})")

    return graph

def test_retrieval_augmentation():
    """Test retrieval-augmented generation with DSPy"""
    print("\n=== Testing Retrieval Augmentation ===")

    # Create graph builder
    graph_builder = GraphBuilder()

    # Test queries
    queries = [
        "What is Sarah's family relationship?",
        "Where does Michael work?",
        "How old is Emma?",
        "What is Sarah's profession?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        retrieved_triples = graph_builder.retrieve_triples(query)
        print(f"Retrieved triples:")
        for triple in retrieved_triples[:3]:
            print(f"  - {triple}")

def test_dspy_signatures():
    """Test custom DSPy signatures for specific tasks"""
    print("\n=== Testing DSPy Signatures ===")

    # Define a custom signature for question answering over graph
    class GraphQASignature(dspy.Signature):
        """Answer questions using knowledge graph context"""
        question = dspy.InputField(desc="The question to answer")
        graph_context = dspy.InputField(desc="Relevant triples from knowledge graph")
        answer = dspy.OutputField(desc="Concise answer based on graph context")

    # Create a predictor
    qa_predictor = dspy.Predict(GraphQASignature)

    # Test with sample data
    question = "Who is Sarah married to and what is his profession?"
    graph_context = """
    - Sarah married_to Michael_Chen
    - Michael_Chen profession cardiologist
    - Michael_Chen works_at Seattle_General_Hospital
    """

    # Get prediction
    result = qa_predictor(question=question, graph_context=graph_context)
    print(f"Question: {question}")
    print(f"Graph Context: {graph_context}")
    print(f"Answer: {result.answer}")

def test_chain_of_thought():
    """Test Chain of Thought reasoning with DSPy"""
    print("\n=== Testing Chain of Thought ===")

    # Define signature for multi-hop reasoning
    class MultiHopReasoningSignature(dspy.Signature):
        """Perform multi-hop reasoning over graph relationships"""
        query = dspy.InputField(desc="Complex query requiring multiple steps")
        initial_facts = dspy.InputField(desc="Initial facts from knowledge graph")
        reasoning_steps = dspy.OutputField(desc="Step-by-step reasoning process")
        final_answer = dspy.OutputField(desc="Final answer after reasoning")

    # Use ChainOfThought
    reasoner = dspy.ChainOfThought(MultiHopReasoningSignature)

    query = "What hospital does Emma's father work at?"
    initial_facts = """
    - Emma parent Michael_Chen
    - Michael_Chen works_at Seattle_General_Hospital
    - Michael_Chen profession cardiologist
    """

    result = reasoner(query=query, initial_facts=initial_facts)
    print(f"Query: {query}")
    print(f"Initial Facts: {initial_facts}")
    print(f"Reasoning: {result.reasoning_steps}")
    print(f"Answer: {result.final_answer}")

def main():
    """Main test function"""
    print("DSPy + SGLang Integration Test")
    print("=" * 50)

    # Check if we should use existing server or start new one
    use_existing_server = os.getenv("USE_EXISTING_SGLANG", "false").lower() == "true"
    port = int(os.getenv("SGLANG_PORT", "30000"))

    server_process = None

    try:
        if use_existing_server:
            if not check_sglang_server(port):
                print(f"No SGLang server found at port {port}")
                print("Please start SGLang server or set USE_EXISTING_SGLANG=false")
                return
            print(f"Using existing SGLang server at port {port}")
        else:
            # Try to use a smaller model for testing
            model_options = [
                "Qwen/Qwen2.5-0.5B-Instruct",  # 0.5B model
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B model
                "microsoft/phi-2",  # 2.7B model
            ]

            for model in model_options:
                print(f"Attempting to use model: {model}")
                server_process = start_sglang_server(model, port)
                if server_process:
                    break

            if not server_process:
                print("Failed to start SGLang server")
                print("\nTo manually start SGLang server:")
                print(f"python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port {port}")
                return

        # Configure DSPy with SGLang
        configure_dspy_with_sglang(port)

        # Run tests
        test_basic_extraction()
        test_retrieval_augmentation()
        test_dspy_signatures()
        test_chain_of_thought()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if server_process:
            print("\nStopping SGLang server...")
            server_process.terminate()
            server_process.wait(timeout=5)
            print("Server stopped")

if __name__ == "__main__":
    main()