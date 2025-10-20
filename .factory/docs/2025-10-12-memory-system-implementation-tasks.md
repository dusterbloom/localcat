Execute memory tasks in order:
1. Implement memory baseline metrics (performance tests and metrics helper) - already has metrics_helper.py, need to create performance tests
2. Implement memory hardening (composite scoring, token budget, dedupe) - enhance retrieval.py
3. Implement semantic memory sidecar (FAISS index, embeddings) - create semantic_sidecar.py
4. Implement coreference upgrade (processor integration with timeout) - enhance memory_hotpath.py

All tasks follow TDD approach with comprehensive tests.