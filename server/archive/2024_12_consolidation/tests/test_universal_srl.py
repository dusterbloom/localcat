#!/usr/bin/env python3
"""
🔥 UNIVERSAL SRL TEST
Test HuggingFace multilingual SRL model on our failed cases
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from typing import List, Tuple
import spacy

class UniversalSRLExtractor:
    """Universal semantic role labeling using HuggingFace multilingual models"""

    def __init__(self, model_name="liaad/srl-enpt_xlmr-base"):
        print(f"🔄 Loading universal SRL model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_trf")

        # Create pipeline for easier usage
        self.srl_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract semantic triples using universal SRL"""
        print(f"🔍 Analyzing: '{text}'")

        # Get SRL predictions
        srl_results = self.srl_pipeline(text)
        print(f"📊 SRL Results: {srl_results}")

        # Parse into semantic triples
        triples = self._srl_to_triples(text, srl_results)

        return triples

    def _srl_to_triples(self, text: str, srl_results: List) -> List[Tuple[str, str, str]]:
        """Convert SRL predictions to semantic triples"""
        triples = []

        # Group by verb predicates
        predicates = {}
        args = {}

        for result in srl_results:
            label = result['entity_group']
            word = result['word'].replace('▁', '').replace('##', '')  # Clean subword tokens
            start = result['start']
            end = result['end']

            if label.startswith('V'):
                # This is a predicate
                predicates[start] = {
                    'word': word,
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                }
            elif label.startswith('ARG') or label.startswith('ARGM'):
                # This is an argument
                if start not in args:
                    args[start] = []
                args[start].append({
                    'label': label,
                    'word': word,
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                })

        print(f"🎯 Predicates: {predicates}")
        print(f"📝 Arguments: {args}")

        # Simple triple formation (will need refinement)
        if predicates and args:
            # For now, create basic triples from first predicate and available args
            pred_info = list(predicates.values())[0]
            predicate = pred_info['word'].lower()

            # Find ARG0 (agent) and ARG1 (patient)
            agent = None
            patient = None

            for arg_list in args.values():
                for arg in arg_list:
                    if arg['label'] == 'ARG0':
                        agent = arg['word'].lower()
                    elif arg['label'] == 'ARG1':
                        patient = arg['word'].lower()

            if agent and patient:
                triples.append((agent, predicate, patient))
            elif agent and predicate:
                triples.append((agent, predicate, ""))

        return triples

def test_universal_srl():
    """Test universal SRL on our failed cases"""
    extractor = UniversalSRLExtractor()

    # Test cases that failed with our current SRL
    failed_cases = [
        "My name is Alex Thompson",
        "My dog's name is Potola",
        "Sarah and John are friends",
        "My favorite color is blue",
        "I was born in 1995",
        "My son is named Jake",
    ]

    # Test cases that worked with our SRL
    working_cases = [
        "Alice feeds the cat",
        "I live in Seattle",
        "I work at Microsoft",
    ]

    print("🔥 UNIVERSAL SRL EXTRACTION TEST")
    print("=" * 70)

    print("\n🚨 PREVIOUSLY FAILED CASES:")
    for i, text in enumerate(failed_cases, 1):
        print(f"\n{i}. Testing: '{text}'")
        print("-" * 50)

        try:
            triples = extractor.extract_triples(text)

            if triples:
                print("✅ SEMANTIC TRIPLES:")
                for triple in triples:
                    print(f"   {triple}")
            else:
                print("❌ No triples generated")
        except Exception as e:
            print(f"❌ ERROR: {e}")

    print("\n\n✅ PREVIOUSLY WORKING CASES:")
    for i, text in enumerate(working_cases, 1):
        print(f"\n{i}. Testing: '{text}'")
        print("-" * 50)

        try:
            triples = extractor.extract_triples(text)

            if triples:
                print("✅ SEMANTIC TRIPLES:")
                for triple in triples:
                    print(f"   {triple}")
            else:
                print("❌ No triples generated")
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_universal_srl()