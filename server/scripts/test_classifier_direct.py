#!/usr/bin/env python3
"""
Test HotMem classifier model with proper prompting.

Tests hotmem-relation-classifier-mlx with classifier-specific prompt format.
"""
import json
import os
import time
import urllib.request
from dotenv import load_dotenv

def test_classifier_direct():
    """Test the classifier model directly with classifier-specific prompts."""
    
    # Load .env
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    
    base_url = os.getenv('HOTMEM_LLM_ASSISTED_BASE_URL', 'http://127.0.0.1:1234/v1')
    model = 'hotmem-relation-classifier-mlx'
    
    print("🧪 Testing Classifier Model Direct")
    print("=" * 70)
    
    test_cases = [
        {
            "text": "Tim Cook is the CEO of Apple and lives in California.",
            "entity_pairs": [
                ("tim cook", "apple"),
                ("tim cook", "california")
            ],
            "expected": ["CEO_of", "lives_in"]
        },
        {
            "text": "Sarah works at Microsoft as a senior engineer and develops Windows.",
            "entity_pairs": [
                ("sarah", "microsoft"),
                ("sarah", "windows")
            ],
            "expected": ["works_at", "develops"]
        },
        {
            "text": "Elon Musk founded Tesla and SpaceX in the early 2000s.",
            "entity_pairs": [
                ("elon musk", "tesla"),
                ("elon musk", "spacex")
            ],
            "expected": ["founder_of", "founder_of"]
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. '{case['text']}'")
        
        extracted_relations = []
        
        # Test each entity pair with classifier format
        for subject, obj in case["entity_pairs"]:
            print(f"  Testing pair: ({subject}, {obj})")
            
            # Classifier-specific prompt
            system = """You are a relation classifier. Given two entities and context, output ONLY the relation type.
If no relation exists, output 'none'.
Valid relations: works_at, CEO_of, lives_in, develops, founder_of, married_to, acquired, headquartered_in, competes_with, retired_from, manages, based_in, graduated_from, joined, etc."""
            
            user = f"""Subject: {subject}
Context: {case['text']}
Object: {obj}

What is the relation type?"""
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            try:
                t0 = time.perf_counter()
                data = json.dumps(payload).encode('utf-8')
                headers = {
                    "Content-Type": "application/json", 
                    "Authorization": "Bearer not-needed"
                }
                
                req = urllib.request.Request(f"{base_url}/chat/completions", data=data, headers=headers, method='POST')
                
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode('utf-8')
                    response = json.loads(body)
                    
                    content = response['choices'][0]['message']['content'].strip()
                    inference_time = (time.perf_counter() - t0) * 1000
                    
                    print(f"    Raw response: '{content}' ({inference_time:.0f}ms)")
                    
                    # Clean up response
                    relation = content.strip()
                    if relation.lower() == 'none':
                        relation = None
                    elif relation and not relation.startswith(('none', 'None', 'NONE')):
                        # Normalize relation format
                        relation = relation.lower().replace(' ', '_').replace('-', '_')
                        extracted_relations.append((subject, relation, obj))
                        print(f"    ✅ Extracted: {subject} --{relation}--> {obj}")
                    else:
                        print(f"    ❌ No relation found")
                        
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        # Calculate metrics
        expected_relations = set()
        for j, (s, o) in enumerate(case["entity_pairs"]):
            expected_relations.add((s, case["expected"][j], o))
        
        actual_relations = set(extracted_relations)
        
        true_positives = len(expected_relations.intersection(actual_relations))
        false_positives = len(actual_relations - expected_relations)
        false_negatives = len(expected_relations - actual_relations)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  📊 Metrics: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
        print(f"  Expected: {case['expected']}")
        print(f"  Actual: {[r[1] for r in extracted_relations]}")
        
        results.append({
            'text': case['text'],
            'expected': case['expected'],
            'actual': [r[1] for r in extracted_relations],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'extracted_triples': extracted_relations
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("CLASSIFIER MODEL SUMMARY")
    print("=" * 70)
    
    if results:
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        
        successful_extractions = sum(1 for r in results if r['f1'] > 0)
        success_rate = successful_extractions / len(results)
        
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Success Rate: {success_rate:.1%} ({successful_extractions}/{len(results)})")
        
        # Count all extractions
        total_expected = sum(len(r['expected']) for r in results)
        total_actual = sum(len(r['actual']) for r in results)
        print(f"Total Relations: {total_actual} extracted vs {total_expected} expected")
    
    return results


if __name__ == '__main__':
    test_classifier_direct()