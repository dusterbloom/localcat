#!/usr/bin/env python3
"""Comprehensive test of Level 3 extraction on all Level1to3_text.md examples"""

from level3_universal_kg import UniversalKGExtractor
import time

def test_all_examples():
    extractor = UniversalKGExtractor()

    # All test sentences from Level1to3_text.md
    test_cases = [
        # News-Style Sentences
        ("Easy News (≈10 words)", "The cat chased the ball across the sunny yard."),
        ("Simple News (≈20 words)", "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees."),
        ("Complex News (≈30 words)", "Yesterday, firefighters quickly responded to a small kitchen fire caused by an unattended stove, saving the family's home and ensuring no one was injured in the timely rescue operation."),
        ("Technical Quantum (≈50 words)", "In the quantum computing algorithm, qubits entangled through superposition states enable parallel processing, where error correction codes, such as surface codes implemented via lattice surgery, mitigate decoherence effects by repeatedly measuring stabilizers to preserve computational fidelity across multiple logical gates."),
        ("Technical Neural Net (≈70 words)", "Engineers designing the neural network architecture incorporated convolutional layers to extract hierarchical features from input images, followed by recurrent units that process sequential data through long short-term memory cells, which selectively retain or discard information via gating mechanisms, ultimately optimizing the model's performance on tasks like object recognition by minimizing cross-entropy loss during backpropagation training cycles."),
        ("Scientific Evolution (≈90 words)", "Evolutionary biologists posit that the adaptive radiation of Darwin's finches on the Galápagos Islands exemplifies punctuated equilibrium, wherein rapid speciation events, driven by ecological niches and selective pressures from varying food sources, interrupt long periods of stasis, as evidenced by morphological divergences in beak structures that correlate with genetic drift and founder effects, thereby challenging gradualist models and underscoring the interplay between contingency and constraint in phylogenetic trajectories."),

        # Conversational Examples
        ("Easy Chat (≈3 words)", "Hey, how's your day?"),
        ("Simple Chat (≈5 words)", "I just ate a tasty sandwich."),
        ("Basic Exchange (≈8 words)", "What movie did you watch last night? It sounds fun."),
        ("Descriptive Chat (≈12 words)", "My dog loves chasing squirrels in the park every morning."),
        ("Technical Chat (≈20 words)", "In coding, loops like for-statements repeat tasks efficiently, but watch for infinite ones that crash your program."),
        ("Scientific Chat (≈30 words)", "Evolution via natural selection favors traits like camouflage in prey animals, as predators' eyes evolve to spot patterns, creating an ongoing arms race in ecosystems."),
    ]

    print("🎯 LEVEL 3 UNIVERSAL KG - COMPREHENSIVE TEST")
    print("=" * 80)

    total_start = time.time()

    for i, (category, text) in enumerate(test_cases, 1):
        print(f"\n📝 TEST {i}: {category}")
        print("─" * 60)
        print(f"Input: \"{text}\"")
        print()

        # Extract knowledge graph
        start_time = time.time()
        kg = extractor.extract_universal_kg(text)
        extraction_time = (time.time() - start_time) * 1000

        # Performance metrics
        print(f"⚡ Performance: {extraction_time:.1f}ms | 📊 Results: {len(kg.entities)} entities, {len(kg.relations)} relations")

        # Show coreference clusters if any
        if hasattr(kg, 'coreference_clusters') and kg.coreference_clusters:
            print(f"🔗 Coreference: {len(kg.coreference_clusters)} clusters")

        # Show discourse structure if any
        if hasattr(kg, 'discourse_relations') and kg.discourse_relations:
            print(f"📚 Discourse: {len(kg.discourse_relations)} RST relations")

        print()

        # Show all relations with quality filtering
        if kg.relations:
            print("🔥 ALL EXTRACTED RELATIONS:")
            for j, relation in enumerate(kg.relations, 1):
                if ('has_attribute' not in relation.predicate and
                    'modifies' not in relation.predicate and
                    'type' not in relation.predicate and
                    'participates_in' not in relation.predicate):
                    print(f"  ✅ {j:2d}. {relation.subject} | {relation.predicate} | {relation.object}")
                else:
                    print(f"  📝 {j:2d}. {relation.subject} | {relation.predicate} | {relation.object} (filtered)")

            # Show beautiful core relations
            beautiful_relations = [r for r in kg.relations
                                 if ('has_attribute' not in r.predicate and
                                     'modifies' not in r.predicate and
                                     'participates_in' not in r.predicate and
                                     'type' not in r.predicate)]

            if beautiful_relations:
                print(f"\n🌟 BEAUTIFUL CORE RELATIONS ({len(beautiful_relations)} relations):")
                for j, relation in enumerate(beautiful_relations, 1):
                    print(f"  🎯 {j}. {relation.subject} | {relation.predicate} | {relation.object}")
        else:
            print("❌ No relations extracted")

        print()

    total_time = (time.time() - total_start) * 1000
    print("=" * 80)
    print(f"🏆 TOTAL TEST TIME: {total_time:.1f}ms for {len(test_cases)} sentences")
    print(f"📈 AVERAGE: {total_time/len(test_cases):.1f}ms per sentence")

if __name__ == "__main__":
    test_all_examples()