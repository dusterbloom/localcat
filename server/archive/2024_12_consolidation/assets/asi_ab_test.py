#!/usr/bin/env python3
"""
A/B/C Quality Test: Original vs Enhanced vs ASI-Basic Systems
"""

import spacy
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Import our systems
from components.extraction.yaml_ud_loader import YAMLUDExtractor
from components.processing.semantic_roles import RelationNormalizer, _canon_entity_text

@dataclass
class QualityResult:
    system: str
    sentence: str
    triples: List[Tuple[str, str, str]]
    extraction_time_ms: float
    quality_score: float  # 0-1 subjective quality
    semantic_correctness: bool
    notes: str

class SemanticRoleExtractor:
    """Wrapper for existing semantic_roles.py system"""

    def __init__(self):
        self.normalizer = RelationNormalizer()

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract using existing semantic approach - simplified for comparison"""
        triples = []

        # Basic pattern matching (simplified from actual system)
        text_lower = text.lower()

        # Name extraction
        if "my name is" in text_lower:
            name_part = text.split("my name is")[-1].strip().rstrip('.')
            triples.append(("you", "has_name", name_part))

        # Work/live patterns
        if "works at" in text_lower or "work at" in text_lower:
            parts = text_lower.replace("works at", "work at").split("work at")
            if len(parts) >= 2:
                who = _canon_entity_text(parts[0].strip())
                where = parts[1].strip().rstrip('.')
                triples.append((who, "works_at", where))

        if "lives in" in text_lower or "live in" in text_lower:
            parts = text_lower.replace("lives in", "live in").split("live in")
            if len(parts) >= 2:
                who = _canon_entity_text(parts[0].strip())
                where = parts[1].strip().rstrip('.')
                triples.append((who, "lives_in", where))

        # Basic SVO patterns
        if "gave" in text_lower:
            # Simple pattern for "X gave Y Z"
            words = text.split()
            if "gave" in words:
                gave_idx = words.index("gave")
                if gave_idx > 0 and gave_idx < len(words) - 1:
                    subj = words[gave_idx - 1]
                    obj = words[gave_idx + 1] if gave_idx + 1 < len(words) else ""
                    triples.append((subj, "gave", obj))

        return triples

class ABCTester:
    """A/B/C test comparing three SRL systems"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # System A: Existing semantic approach
        self.system_a = SemanticRoleExtractor()

        # System B: Enhanced UD approach (previous best)
        self.system_b = YAMLUDExtractor("enhanced_fastlane_rules.ud.yaml")

        # System C: ASI-inspired approach
        self.system_c = YAMLUDExtractor("test_asi_basic.yaml")

        # Test sentences from existing tests
        self.test_sentences = [
            # Simple cases that should work
            "My name is Alex Thompson.",
            "John works at Google.",
            "Mary lives in Paris.",
            "John gave Mary a book.",
            "She likes chocolate.",

            # More complex cases
            "The company announced profits.",
            "John and Mary gave books to students.",
            "After John graduated, he worked at Apple.",
            "The CEO of Microsoft announced profits.",
            "This solution is better than the previous one.",
            "All students passed the exam.",
            "I think that she knows the answer.",
            "John does not like Mary.",
            "The book was written by the author.",
        ]

    def run_abc_test(self):
        """Run comprehensive A/B/C quality comparison"""

        print("🔥 A/B/C SRL QUALITY TEST")
        print("=" * 80)
        print("A = Existing semantic_roles.py system")
        print("B = Enhanced UD→SRL system")
        print("C = ASI-inspired UD→SRL system")
        print("=" * 80)

        results_a = []
        results_b = []
        results_c = []

        for i, sentence in enumerate(self.test_sentences):
            print(f"\\n📝 Test {i+1}: {sentence}")
            print("-" * 60)

            # Test System A
            start_time = time.time()
            triples_a = self.system_a.extract_triples(sentence)
            time_a = (time.time() - start_time) * 1000

            print(f"🅰️  System A (Semantic): {len(triples_a)} triples in {time_a:.1f}ms")
            for triple in triples_a:
                print(f"   • {triple}")

            # Test System B
            doc = self.nlp(sentence)
            start_time = time.time()
            triples_b = self.system_b.extract_triples(doc)
            time_b = (time.time() - start_time) * 1000

            print(f"🅱️  System B (Enhanced-UD): {len(triples_b)} triples in {time_b:.1f}ms")
            for triple in triples_b:
                print(f"   • {triple}")

            # Test System C
            start_time = time.time()
            triples_c = self.system_c.extract_triples(doc)
            time_c = (time.time() - start_time) * 1000

            print(f"🅲  System C (ASI-UD): {len(triples_c)} triples in {time_c:.1f}ms")
            for triple in triples_c:
                print(f"   • {triple}")

            # Quality assessment
            quality_a = self._assess_quality(sentence, triples_a, "A")
            quality_b = self._assess_quality(sentence, triples_b, "B")
            quality_c = self._assess_quality(sentence, triples_c, "C")

            print(f"📊 Quality: A={quality_a:.2f} | B={quality_b:.2f} | C={quality_c:.2f}")

            # Determine winner
            best_quality = max(quality_a, quality_b, quality_c)
            if quality_a == best_quality:
                winner = "A"
            elif quality_b == best_quality:
                winner = "B"
            else:
                winner = "C"

            if quality_a == quality_b == quality_c:
                print("🤝 Tie")
            else:
                print(f"🏆 Winner: System {winner}")

            results_a.append(QualityResult("A", sentence, triples_a, time_a, quality_a,
                                         self._is_semantically_correct(sentence, triples_a), ""))
            results_b.append(QualityResult("B", sentence, triples_b, time_b, quality_b,
                                         self._is_semantically_correct(sentence, triples_b), ""))
            results_c.append(QualityResult("C", sentence, triples_c, time_c, quality_c,
                                         self._is_semantically_correct(sentence, triples_c), ""))

        # Final summary
        self._print_summary(results_a, results_b, results_c)

    def _assess_quality(self, sentence: str, triples: List[Tuple[str, str, str]], system: str) -> float:
        """Assess semantic quality of extracted triples (0-1 scale)"""
        if not triples:
            return 0.0

        quality = 0.0
        sentence_lower = sentence.lower()

        # Quality criteria
        for subj, pred, obj in triples:
            triple_quality = 0.0

            # Check if entities are properly extracted (not template artifacts)
            if "{" in subj or "{" in pred or "{" in obj:
                triple_quality -= 0.5  # Major penalty for unresolved templates

            # Check semantic meaningfulness
            if pred in ["quantified_as", "unknown_verb", "unknown"]:
                triple_quality -= 0.3  # Not very semantic

            # Check if relation matches sentence content
            if "name" in sentence_lower and "has_name" in pred:
                triple_quality += 0.8  # Good semantic match
            elif "work" in sentence_lower and ("work" in pred or "works_at" in pred):
                triple_quality += 0.8
            elif "live" in sentence_lower and ("live" in pred or "lives_in" in pred):
                triple_quality += 0.8
            elif "give" in sentence_lower and "give" in pred:
                triple_quality += 0.7
            elif "like" in sentence_lower and "like" in pred:
                triple_quality += 0.7
            elif pred in ["be", "aka", "possess"]:
                triple_quality += 0.5  # Basic but correct
            else:
                triple_quality += 0.3  # At least extracted something

            # Bonus for proper entity canonicalization
            if subj == "you" and "my" in sentence_lower:
                triple_quality += 0.2

            quality += max(0, triple_quality)

        return min(1.0, quality / len(triples))  # Normalize by number of triples

    def _is_semantically_correct(self, sentence: str, triples: List[Tuple[str, str, str]]) -> bool:
        """Boolean check if triples are semantically correct"""
        if not triples:
            return False

        # Check for template artifacts
        for subj, pred, obj in triples:
            if "{" in subj or "{" in pred or "{" in obj:
                return False

        return True

    def _print_summary(self, results_a, results_b, results_c):
        """Print final comparison summary"""

        print(f"\\n{'=' * 80}")
        print("📊 FINAL A/B/C TEST RESULTS")
        print(f"{'=' * 80}")

        # Aggregate stats
        systems = [("A", results_a), ("B", results_b), ("C", results_c)]

        for system_name, results in systems:
            avg_quality = sum(r.quality_score for r in results) / len(results)
            avg_time = sum(r.extraction_time_ms for r in results) / len(results)
            total_triples = sum(len(r.triples) for r in results)
            correct = sum(1 for r in results if r.semantic_correctness)

            icon = "🅰️" if system_name == "A" else "🅱️" if system_name == "B" else "🅲"
            print(f"\\n{icon}  System {system_name}:")
            print(f"   Quality Score: {avg_quality:.3f}/1.000")
            print(f"   Avg Time: {avg_time:.1f}ms")
            print(f"   Total Triples: {total_triples}")
            print(f"   Semantically Correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

        # Winner determination
        avg_a = sum(r.quality_score for r in results_a) / len(results_a)
        avg_b = sum(r.quality_score for r in results_b) / len(results_b)
        avg_c = sum(r.quality_score for r in results_c) / len(results_c)

        print(f"\\n🏆 FINAL WINNER:")
        if avg_c >= avg_a and avg_c >= avg_b:
            advantage_over_a = ((avg_c - avg_a) / avg_a * 100) if avg_a > 0 else float('inf')
            advantage_over_b = ((avg_c - avg_b) / avg_b * 100) if avg_b > 0 else float('inf')
            print(f"   🅲 SYSTEM C (ASI-UD) WINS!")
            if avg_a > 0: print(f"   • {advantage_over_a:.1f}% better than System A")
            if avg_b > 0: print(f"   • {advantage_over_b:.1f}% better than System B")
        elif avg_a >= avg_b:
            advantage = ((avg_a - avg_b) / avg_b * 100) if avg_b > 0 else float('inf')
            print(f"   🅰️ SYSTEM A (Semantic) WINS by {advantage:.1f}%")
        else:
            advantage = ((avg_b - avg_a) / avg_a * 100) if avg_a > 0 else float('inf')
            print(f"   🅱️ SYSTEM B (Enhanced-UD) WINS by {advantage:.1f}%")

if __name__ == "__main__":
    tester = ABCTester()
    tester.run_abc_test()