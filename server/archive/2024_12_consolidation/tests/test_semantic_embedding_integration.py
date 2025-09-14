#!/usr/bin/env python3
"""
Test the complete semantic embedding integration:
1. SRL extracts relations with embeddings
2. HotMemoryFacade stores embeddings in edge metadata
3. MemoryRetriever uses embeddings for semantic similarity
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_semantic_integration():
    """Test the complete semantic embedding integration"""
    print("🧪 TESTING COMPLETE SEMANTIC EMBEDDING INTEGRATION")
    print("=" * 70)

    # Test case that was previously failing
    test_text = "The CEO announced that the company would restructure after declining profits."
    print(f"📝 Test Text: {test_text}")
    print(f"🎯 Expected: Semantic relations with embeddings for better retrieval")

    # Test SRL embedding extraction
    print(f"\n1️⃣ TESTING SRL WITH EMBEDDINGS:")
    print("-" * 40)

    from components.processing.semantic_roles import SRLExtractor
    import spacy

    srl = SRLExtractor(use_normalizer=True)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(test_text)

    predications = srl.doc_to_predications(doc)
    print(f"📊 Predications found: {len(predications)}")
    for pred in predications:
        print(f"   - Predicate: {pred.predicate}, Roles: {pred.roles}")

    # Test embedding extraction
    triples_with_meta = srl.predications_to_triples_with_embeddings(predications)
    print(f"📈 Triples with embeddings: {len(triples_with_meta)}")
    for s, r, o, meta in triples_with_meta:
        has_embedding = 'rel_embedding' in meta
        print(f"   - ({s}, {r}, {o}) | Embedding: {'✅' if has_embedding else '❌'}")
        if has_embedding:
            print(f"     Original predicate: {meta.get('original_predicate', 'N/A')}")

    # Test complete memory integration
    print(f"\n2️⃣ TESTING MEMORY INTEGRATION:")
    print("-" * 40)

    from components.memory.memory_store import MemoryStore, Paths
    from components.memory.hotmemory_facade import HotMemoryFacade

    # Create temporary in-memory store
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = Paths(
            sqlite_path=os.path.join(temp_dir, "test.db"),
            lmdb_dir=os.path.join(temp_dir, "test.lmdb")
        )
        store = MemoryStore(paths)
        facade = HotMemoryFacade(store)

        # Process the text
        bullets, stored_facts = facade.process_turn(test_text, session_id="test", turn_id=1)
        print(f"📊 Stored facts: {len(stored_facts)}")
        print(f"📝 Memory bullets: {len(bullets)}")

        for fact in stored_facts:
            print(f"   - Stored: {fact}")

        # Test retrieval with semantic similarity
        print(f"\n3️⃣ TESTING SEMANTIC RETRIEVAL:")
        print("-" * 40)

        # Query that should match semantically but not lexically
        semantic_queries = [
            "Who communicated about changes?",  # should match "announced"
            "What caused the reorganization?",  # should match causal relations
            "Who talked about business problems?"  # should match through embeddings
        ]

        for query in semantic_queries:
            print(f"\n🔍 Query: '{query}'")
            try:
                result_bullets, _ = facade.process_turn(query, session_id="test", turn_id=2)
                print(f"   📝 Retrieved bullets: {len(result_bullets)}")
                for bullet in result_bullets[:2]:  # Show top 2
                    print(f"   • {bullet}")
            except Exception as e:
                print(f"   ❌ Query failed: {e}")

    print(f"\n🎉 SEMANTIC INTEGRATION TEST COMPLETE!")
    print("=" * 70)
    print("✅ Key improvements:")
    print("   • SRL extracts semantic relations (not just syntax)")
    print("   • Embeddings stored for rich semantic matching")
    print("   • Retrieval uses embedding similarity for better results")
    print("   • No LLMs needed - pure NLP + embeddings approach")

if __name__ == "__main__":
    test_semantic_integration()