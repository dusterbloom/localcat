#!/usr/bin/env python3
"""
LLM-based Multilingual Semantic Role Labeling System
====================================================

Research-backed approach using retrieval-augmented framework with step-by-step conversational task reformulation.
Based on findings that LLMs can achieve state-of-the-art SRL with proper prompting and external knowledge.

Supports: English, Italian, Spanish, German, French (and any language the LLM supports)
"""

import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import openai

@dataclass
class SRLResult:
    """Structured SRL result with predicate-argument structure"""
    sentence: str
    language: str
    predicates: List[Dict[str, Any]]
    semantic_triples: List[Tuple[str, str, str]]  # (arg, relation, pred)
    confidence: float

class LLMSemanticRoleLabeler:
    """
    LLM-based Semantic Role Labeling system using retrieval-augmented prompting.
    Implements the four-stage pipeline: predicate disambiguation, role retrieval,
    argument labeling, and post-processing.
    """

    def __init__(self, api_base: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        """Initialize with local LM Studio or any OpenAI-compatible API"""
        self.client = openai.AsyncOpenAI(
            base_url=api_base,
            api_key=api_key
        )

        # PropBank role examples for retrieval augmentation
        self.role_examples = {
            "ARG0": "Agent, typically the subject performing the action",
            "ARG1": "Patient, typically the direct object being affected",
            "ARG2": "Beneficiary, instrument, or indirect object",
            "ARG3": "Starting point, beneficiary, or attribute",
            "ARG4": "Ending point or destination",
            "ARGM-TMP": "Temporal modifier (when)",
            "ARGM-LOC": "Location modifier (where)",
            "ARGM-CAU": "Cause or reason (why)",
            "ARGM-MNR": "Manner (how)",
            "ARGM-PRP": "Purpose (for what)",
            "ARGM-DIR": "Direction",
            "ARGM-ADV": "Adverbial modifier"
        }

        # Language-specific examples for better multilingual performance
        self.examples = {
            "en": [
                ("John gave Mary a book yesterday.", "gave", ["John:ARG0", "Mary:ARG2", "book:ARG1", "yesterday:ARGM-TMP"]),
                ("The company announced profits.", "announced", ["company:ARG0", "profits:ARG1"]),
            ],
            "es": [
                ("Juan le dio un libro a María ayer.", "dio", ["Juan:ARG0", "María:ARG2", "libro:ARG1", "ayer:ARGM-TMP"]),
                ("La empresa anunció ganancias.", "anunció", ["empresa:ARG0", "ganancias:ARG1"]),
            ],
            "it": [
                ("Giovanni ha dato un libro a Maria ieri.", "dato", ["Giovanni:ARG0", "Maria:ARG2", "libro:ARG1", "ieri:ARGM-TMP"]),
                ("L'azienda ha annunciato profitti.", "annunciato", ["azienda:ARG0", "profitti:ARG1"]),
            ],
            "de": [
                ("Johann gab Maria gestern ein Buch.", "gab", ["Johann:ARG0", "Maria:ARG2", "Buch:ARG1", "gestern:ARGM-TMP"]),
                ("Das Unternehmen kündigte Gewinne an.", "kündigte", ["Unternehmen:ARG0", "Gewinne:ARG1"]),
            ],
            "fr": [
                ("Jean a donné un livre à Marie hier.", "donné", ["Jean:ARG0", "Marie:ARG2", "livre:ARG1", "hier:ARGM-TMP"]),
                ("L'entreprise a annoncé des bénéfices.", "annoncé", ["entreprise:ARG0", "bénéfices:ARG1"]),
            ]
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["the", "and", "is", "was", "have"]):
            return "en"
        elif any(word in text_lower for word in ["el", "la", "es", "fue", "tiene", "de", "en"]):
            return "es"
        elif any(word in text_lower for word in ["il", "la", "è", "era", "ha", "di", "in"]):
            return "it"
        elif any(word in text_lower for word in ["der", "die", "das", "ist", "war", "hat", "von"]):
            return "de"
        elif any(word in text_lower for word in ["le", "la", "est", "était", "a", "de", "dans"]):
            return "fr"
        else:
            return "en"  # default to English

    def _build_retrieval_context(self, language: str, predicate: str) -> str:
        """Build retrieval-augmented context with role definitions and examples"""
        role_definitions = "\n".join([f"- {role}: {desc}" for role, desc in self.role_examples.items()])

        examples = self.examples.get(language, self.examples["en"])
        example_text = "\n".join([
            f"Sentence: {sent}\nPredicate: {pred}\nRoles: {', '.join(roles)}"
            for sent, pred, roles in examples[:2]  # Use 2 examples
        ])

        return f"""
SEMANTIC ROLE DEFINITIONS:
{role_definitions}

EXAMPLES IN {language.upper()}:
{example_text}

PREDICATE TO ANALYZE: {predicate}
"""

    async def _stage1_predicate_disambiguation(self, sentence: str, language: str) -> List[str]:
        """Stage 1: Identify and disambiguate predicates in the sentence"""

        prompt = f"""
You are a semantic role labeling expert. Your task is to identify ALL predicates (verbs that can have semantic arguments) in the sentence.

Sentence: "{sentence}"
Language: {language}

Instructions:
1. Identify all verbs that represent actions, states, or events
2. Focus on main verbs, not auxiliary verbs
3. Return ONLY the predicate words, one per line
4. If no clear predicates, return "NONE"

Predicates:
"""

        response = await self.client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )

        predicates = []
        for line in response.choices[0].message.content.strip().split('\n'):
            line = line.strip()
            if line and line != "NONE" and not line.startswith("Predicates"):
                predicates.append(line.lower())

        return predicates

    async def _stage2_role_retrieval(self, sentence: str, predicate: str, language: str) -> Dict[str, str]:
        """Stage 2: Retrieve appropriate semantic roles for this predicate"""

        context = self._build_retrieval_context(language, predicate)

        prompt = f"""
{context}

You are analyzing this sentence for semantic roles:
Sentence: "{sentence}"
Target Predicate: "{predicate}"

Task: For each word/phrase in the sentence, determine its semantic role relative to the predicate "{predicate}".

Instructions:
1. Identify the agent (ARG0) - who/what performs the action
2. Identify the patient (ARG1) - what is directly affected
3. Identify other arguments (ARG2, ARG3, ARG4) - beneficiaries, instruments, etc.
4. Identify modifiers (ARGM-TMP, ARGM-LOC, ARGM-CAU, etc.)
5. Return ONLY the word/phrase and its role, format: "word/phrase:ROLE"
6. Skip words that don't have semantic roles
7. One role per line

Semantic roles:
"""

        response = await self.client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )

        roles = {}
        for line in response.choices[0].message.content.strip().split('\n'):
            if ':' in line:
                try:
                    word_phrase, role = line.split(':', 1)
                    roles[word_phrase.strip()] = role.strip()
                except ValueError:
                    continue

        return roles

    async def _stage3_argument_labeling(self, sentence: str, predicate: str, roles: Dict[str, str], language: str) -> List[Tuple[str, str, str]]:
        """Stage 3: Create semantic triples from labeled arguments"""

        triples = []
        predicate_sense = predicate

        # Create triples for each argument
        for arg, role in roles.items():
            if role.startswith('ARG'):  # Core arguments
                # Transform into semantic relation
                if role == 'ARG0':
                    relation = f"performs_{predicate}"
                    triples.append((arg, relation, predicate_sense))
                elif role == 'ARG1':
                    relation = f"is_{predicate}ed_by" if predicate.endswith('e') else f"is_{predicate}d_by"
                    triples.append((predicate_sense, f"affects", arg))
                elif role == 'ARG2':
                    triples.append((predicate_sense, f"involves", arg))
                else:
                    triples.append((arg, f"participates_in_{predicate}", predicate_sense))

            elif role.startswith('ARGM'):  # Modifiers
                if role == 'ARGM-TMP':
                    triples.append((predicate_sense, "occurs_at", arg))
                elif role == 'ARGM-LOC':
                    triples.append((predicate_sense, "occurs_at", arg))
                elif role == 'ARGM-CAU':
                    triples.append((predicate_sense, "caused_by", arg))
                elif role == 'ARGM-MNR':
                    triples.append((predicate_sense, "performed_as", arg))
                else:
                    triples.append((predicate_sense, "modified_by", arg))

        return triples

    async def _stage4_post_processing(self, triples: List[Tuple[str, str, str]], sentence: str) -> List[Tuple[str, str, str]]:
        """Stage 4: Post-process and validate semantic triples"""

        # Remove duplicates while preserving order
        seen = set()
        filtered_triples = []
        for triple in triples:
            if triple not in seen:
                seen.add(triple)
                filtered_triples.append(triple)

        # Sort by relevance (core arguments first)
        def sort_key(triple):
            _, relation, _ = triple
            if "performs" in relation:
                return 0
            elif "affects" in relation:
                return 1
            elif "occurs" in relation:
                return 2
            else:
                return 3

        return sorted(filtered_triples, key=sort_key)

    async def extract_semantic_roles(self, sentence: str, language: Optional[str] = None) -> SRLResult:
        """
        Main method: Extract semantic roles using the four-stage LLM pipeline
        """
        if language is None:
            language = self._detect_language(sentence)

        try:
            # Stage 1: Predicate disambiguation
            predicates = await self._stage1_predicate_disambiguation(sentence, language)

            if not predicates:
                return SRLResult(
                    sentence=sentence,
                    language=language,
                    predicates=[],
                    semantic_triples=[],
                    confidence=0.0
                )

            all_triples = []
            predicate_info = []

            # Process each predicate
            for predicate in predicates:
                # Stage 2: Role retrieval
                roles = await self._stage2_role_retrieval(sentence, predicate, language)

                # Stage 3: Argument labeling
                triples = await self._stage3_argument_labeling(sentence, predicate, roles, language)

                predicate_info.append({
                    "predicate": predicate,
                    "roles": roles,
                    "triples": triples
                })

                all_triples.extend(triples)

            # Stage 4: Post-processing
            final_triples = await self._stage4_post_processing(all_triples, sentence)

            return SRLResult(
                sentence=sentence,
                language=language,
                predicates=predicate_info,
                semantic_triples=final_triples,
                confidence=0.8 if len(final_triples) > 0 else 0.2
            )

        except Exception as e:
            print(f"Error in SRL extraction: {e}")
            return SRLResult(
                sentence=sentence,
                language=language,
                predicates=[],
                semantic_triples=[],
                confidence=0.0
            )

# Test function
async def test_multilingual_srl():
    """Test the LLM-based SRL system with multiple languages"""

    srl = LLMSemanticRoleLabeler()

    test_sentences = [
        ("John gave Mary a book yesterday.", "en"),
        ("Juan le dio un libro a María ayer.", "es"),
        ("Giovanni ha dato un libro a Maria ieri.", "it"),
        ("Johann gab Maria gestern ein Buch.", "de"),
        ("Jean a donné un livre à Marie hier.", "fr"),
        ("My name is Alex Thompson.", "en"),
        ("La empresa anunció grandes ganancias.", "es"),
    ]

    print("🚀 Testing LLM-based Multilingual Semantic Role Labeling")
    print("=" * 60)

    for sentence, lang in test_sentences:
        print(f"\n📝 Sentence: {sentence}")
        print(f"🌐 Language: {lang}")

        result = await srl.extract_semantic_roles(sentence, lang)

        print(f"📊 Confidence: {result.confidence:.2f}")
        print("🎯 Predicates:")
        for pred_info in result.predicates:
            print(f"   • {pred_info['predicate']}: {pred_info['roles']}")

        print("🔗 Semantic Triples:")
        for triple in result.semantic_triples:
            print(f"   • {triple}")

        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_multilingual_srl())