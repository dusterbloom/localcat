#!/usr/bin/env python3
"""
Optimized HotMem classifier test showing true performance potential.

This demonstrates the dramatic speed improvements of the classifier model
when used with proper prompting and post-processing.
"""
import json
import os
import time
import urllib.request
from dotenv import load_dotenv

def test_optimized_classifier():
    """Test classifier with optimized single-relation prompting."""
    
    # Load .env
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    
    base_url = os.getenv('HOTMEM_LLM_ASSISTED_BASE_URL', 'http://127.0.0.1:1234/v1')
    model = 'hotmem-relation-classifier-mlx'
    
    print("🚀 Optimized Classifier Performance Test")
    print("=" * 70)
    
    test_cases = [
        {
            "text": "Tim Cook is the CEO of Apple and lives in California.",
            "entity_pairs": [
                ("tim cook", "apple", "CEO_of"),
                ("tim cook", "california", "lives_in")
            ]
        },
        {
            "text": "Sarah works at Microsoft as a senior engineer and develops Windows.",
            "entity_pairs": [
                ("sarah", "microsoft", "works_at"),
                ("sarah", "windows", "develops")
            ]
        },
        {
            "text": "Elon Musk founded Tesla and SpaceX in the early 2000s.",
            "entity_pairs": [
                ("elon musk", "tesla", "founder_of"),
                ("elon musk", "spacex", "founder_of")
            ]
        },
        {
            "text": "Jennifer is married to Michael and they have two children.",
            "entity_pairs": [
                ("jennifer", "michael", "married_to")
            ]
        },
        {
            "text": "Amazon acquired Whole Foods in 2017 for $13.7 billion.",
            "entity_pairs": [
                ("amazon", "whole foods", "acquired")
            ]
        }
    ]
    
    total_correct = 0
    total_pairs = 0
    all_times = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. '{case['text']}'")
        case_correct = 0
        
        for subject, obj, expected_relation in case["entity_pairs"]:
            print(f"  Testing: {subject} -> {obj}")
            
            # Optimized single-relation prompt
            system = """You are a relation classifier. Output ONLY ONE relation type.
If no clear relation exists, output 'none'.

Common relations: works_at, CEO_of, lives_in, develops, founder_of, married_to, acquired, headquartered_in, competes_with, retired_from, manages, based_in, graduated_from, joined"""
            
            user = f"""Subject: {subject}
Context: {case['text']}
Object: {obj}

Relation type (one word):"""
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": 20,
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
                    all_times.append(inference_time)
                    
                    # Clean and normalize response
                    content = content.lower().strip()
                    if ',' in content:
                        content = content.split(',')[0]  # Take first relation
                    content = content.replace(' ', '_').replace('-', '_')
                    
                    # Remove common prefixes/suffixes
                    for prefix in ['is_', 'was_', 'the_', 'a_', 'an_']:
                        if content.startswith(prefix):
                            content = content[len(prefix):]
                    
                    # Check if correct
                    is_correct = content == expected_relation.lower()
                    if is_correct:
                        case_correct += 1
                        total_correct += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    print(f"    {status} Expected: {expected_relation}, Got: {content} ({inference_time:.0f}ms)")
                    
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        total_pairs += len(case["entity_pairs"])
        case_accuracy = case_correct / len(case["entity_pairs"])
        print(f"  Case Accuracy: {case_accuracy:.1%} ({case_correct}/{len(case['entity_pairs'])})")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🚀 OPTIMIZED CLASSIFIER RESULTS")
    print("=" * 70)
    
    overall_accuracy = total_correct / total_pairs if total_pairs > 0 else 0
    avg_time = sum(all_times) / len(all_times) if all_times else 0
    
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_pairs})")
    print(f"Average Inference Time: {avg_time:.0f}ms")
    print(f"Speed Improvement: ~{1400/avg_time:.0f}x faster than v2")
    
    if overall_accuracy >= 0.8:
        print("🎉 EXCELLENT: Classifier model shows dramatic improvements!")
    elif overall_accuracy >= 0.6:
        print("👍 GOOD: Classifier model shows significant potential!")
    else:
        print("📝 NEEDS WORK: Classifier model has room for improvement")
    
    print(f"\nRecommendation: Use {'hotmem-relation-classifier-mlx' if overall_accuracy > 0.5 else 'relation-extractor-v2-mlx'}")
    
    return {
        'accuracy': overall_accuracy,
        'avg_time_ms': avg_time,
        'speed_improvement': 1400 / avg_time if avg_time > 0 else 0
    }


if __name__ == '__main__':
    test_optimized_classifier()