"""
HotMem V3 Quality Evaluation Script
Detailed analysis of relation-extractor-v2-mlx model performance
"""

import json
import time
import requests
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityEvaluator:
    """Evaluate the quality of relation extraction results"""
    
    def __init__(self, model_name: str = "relation-extractor-v2-mlx", 
                 api_base: str = "http://localhost:1234"):
        self.model_name = model_name
        self.api_base = api_base
        self.api_url = f"{api_base}/v1/chat/completions"
    
    def extract_relations(self, text: str) -> Dict[str, Any]:
        """Extract relations using the model"""
        
        system_prompt = """You are an expert relation extraction system. Extract entities and relations from the given text and output them in JSON format.

Entities should include people, organizations, locations, and other important nouns with their types.
Relations should connect entities with meaningful predicates like 'works_for', 'located_in', 'develops', 'friends_with', etc.

Output format:
{
  "entities": [
    {"text": "Entity Name", "type": "PERSON/ORG/LOC/PRODUCT", "confidence": 0.9}
  ],
  "relations": [
    {"subject": "subject_entity", "predicate": "relation_type", "object": "object_entity", "confidence": 0.8}
  ],
  "confidence": 0.85
}

Only output the JSON, no other text."""

        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract entities and relations from this text:\n\n{text}"}
                ],
                "temperature": 0.1,
                "max_tokens": 512,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract JSON
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    return json.loads(json_str)
            
            return {"entities": [], "relations": [], "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"entities": [], "relations": [], "confidence": 0.0}
    
    def evaluate_quality(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Evaluate quality against ground truth"""
        
        results = {
            'total_tests': len(test_cases),
            'correct_extractions': 0,
            'partial_extractions': 0,
            'failed_extractions': 0,
            'entity_accuracy': 0.0,
            'relation_accuracy': 0.0,
            'confidence_distribution': [],
            'error_analysis': [],
            'detailed_results': []
        }
        
        entity_scores = []
        relation_scores = []
        
        for i, test_case in enumerate(test_cases):
            text = test_case['text']
            expected_entities = test_case.get('expected_entities', [])
            expected_relations = test_case.get('expected_relations', [])
            
            # Get model prediction
            prediction = self.extract_relations(text)
            
            # Evaluate entities
            predicted_entities = [e.get('text', '').lower() for e in prediction.get('entities', [])]
            expected_entities_lower = [e.lower() for e in expected_entities]
            
            # Calculate entity accuracy
            if expected_entities_lower:
                entity_matches = sum(1 for e in expected_entities_lower if e in predicted_entities)
                entity_accuracy = entity_matches / len(expected_entities_lower)
                entity_scores.append(entity_accuracy)
            else:
                entity_accuracy = 1.0 if not predicted_entities else 0.0
                entity_scores.append(entity_accuracy)
            
            # Evaluate relations
            predicted_relations = []
            for rel in prediction.get('relations', []):
                pred_rel = {
                    'subject': rel.get('subject', '').lower(),
                    'predicate': rel.get('predicate', '').lower(),
                    'object': rel.get('object', '').lower()
                }
                predicted_relations.append(pred_rel)
            
            expected_relations_lower = []
            for rel in expected_relations:
                expected_rel = {
                    'subject': rel.get('subject', '').lower(),
                    'predicate': rel.get('predicate', '').lower(),
                    'object': rel.get('object', '').lower()
                }
                expected_relations_lower.append(expected_rel)
            
            # Calculate relation accuracy by comparing string representations
            if expected_relations_lower:
                relation_matches = 0
                for er in expected_relations_lower:
                    er_str = f"{er['subject']}|{er['predicate']}|{er['object']}"
                    for pr in predicted_relations:
                        pr_str = f"{pr['subject']}|{pr['predicate']}|{pr['object']}"
                        if er_str == pr_str:
                            relation_matches += 1
                            break
                relation_accuracy = relation_matches / len(expected_relations_lower)
            else:
                relation_accuracy = 1.0 if not predicted_relations else 0.0
            
            relation_scores.append(relation_accuracy)
            
            # Overall accuracy for this test case
            overall_accuracy = (entity_accuracy + relation_accuracy) / 2
            
            # Categorize result
            if overall_accuracy >= 0.8:
                results['correct_extractions'] += 1
            elif overall_accuracy >= 0.5:
                results['partial_extractions'] += 1
            else:
                results['failed_extractions'] += 1
            
            # Store confidence
            confidence = prediction.get('confidence', 0.0)
            results['confidence_distribution'].append(confidence)
            
            # Detailed result
            detailed_result = {
                'test_id': i + 1,
                'text': text,
                'expected_entities': expected_entities,
                'predicted_entities': [e.get('text', '') for e in prediction.get('entities', [])],
                'expected_relations': expected_relations,
                'predicted_relations': prediction.get('relations', []),
                'entity_accuracy': entity_accuracy,
                'relation_accuracy': relation_accuracy,
                'overall_accuracy': overall_accuracy,
                'confidence': confidence,
                'processing_time': time.time()
            }
            results['detailed_results'].append(detailed_result)
            
            # Error analysis
            if overall_accuracy < 0.8:
                error_info = {
                    'test_id': i + 1,
                    'text': text,
                    'errors': []
                }
                
                if entity_accuracy < 0.8:
                    missing_entities = set(expected_entities_lower) - set(predicted_entities)
                    extra_entities = set(predicted_entities) - set(expected_entities_lower)
                    if missing_entities:
                        error_info['errors'].append(f"Missing entities: {list(missing_entities)}")
                    if extra_entities:
                        error_info['errors'].append(f"Extra entities: {list(extra_entities)}")
                
                if relation_accuracy < 0.8:
                    # Convert to string representations for comparison
                    expected_rel_strs = [f"{er['subject']}|{er['predicate']}|{er['object']}" for er in expected_relations_lower]
                    predicted_rel_strs = [f"{pr['subject']}|{pr['predicate']}|{pr['object']}" for pr in predicted_relations]
                    
                    missing_relations = set(expected_rel_strs) - set(predicted_rel_strs)
                    extra_relations = set(predicted_rel_strs) - set(expected_rel_strs)
                    
                    if missing_relations:
                        error_info['errors'].append(f"Missing relations: {list(missing_relations)}")
                    if extra_relations:
                        error_info['errors'].append(f"Extra relations: {list(extra_relations)}")
                
                results['error_analysis'].append(error_info)
        
        # Calculate overall scores
        if entity_scores:
            results['entity_accuracy'] = sum(entity_scores) / len(entity_scores)
        if relation_scores:
            results['relation_accuracy'] = sum(relation_scores) / len(relation_scores)
        
        return results
    
    def analyze_model_behavior(self) -> Dict[str, Any]:
        """Analyze model behavior patterns"""
        
        behavior_tests = [
            # Entity recognition tests
            {
                'text': "Tim Cook is the CEO of Apple.",
                'category': 'entity_recognition',
                'expected_patterns': ['PERSON', 'ORGANIZATION', 'leadership']
            },
            {
                'text': "Microsoft is headquartered in Redmond, Washington.",
                'category': 'entity_recognition',
                'expected_patterns': ['ORGANIZATION', 'LOCATION']
            },
            {
                'text': "I bought a new iPhone from the Apple Store.",
                'category': 'entity_recognition',
                'expected_patterns': ['PRODUCT', 'ORGANIZATION']
            },
            
            # Relation extraction tests
            {
                'text': "Sarah works at Google as a software engineer.",
                'category': 'relation_extraction',
                'expected_patterns': ['employment', 'organization', 'profession']
            },
            {
                'text': "Tesla was founded by Elon Musk in 2003.",
                'category': 'relation_extraction',
                'expected_patterns': ['founding', 'leadership', 'temporal']
            },
            {
                'text': "Amazon is headquartered in Seattle, Washington.",
                'category': 'relation_extraction',
                'expected_patterns': ['location', 'headquarters']
            },
            
            # Complex sentences
            {
                'text': "My colleague John, who works at Microsoft, told me that his friend Sarah recently joined Google as a product manager.",
                'category': 'complex_sentences',
                'expected_patterns': ['multiple_entities', 'multiple_relations', 'nested_clauses']
            },
            {
                'text': "The new MacBook Pro, developed by Apple in Cupertino, was released last month and is now available in stores worldwide.",
                'category': 'complex_sentences',
                'expected_patterns': ['product_development', 'location', 'temporal', 'distribution']
            },
            
            # Conversational patterns
            {
                'text': "Hi, I'm Alex and I work at Spotify. What about you?",
                'category': 'conversational',
                'expected_patterns': ['self_introduction', 'employment', 'question']
            },
            {
                'text': "I heard that Michael left Apple to join a startup in San Francisco.",
                'category': 'conversational',
                'expected_patterns': ['career_change', 'organization_change', 'location']
            }
        ]
        
        behavior_analysis = {
            'total_tests': len(behavior_tests),
            'category_performance': defaultdict(list),
            'pattern_recognition': defaultdict(int),
            'confidence_by_category': defaultdict(list),
            'detailed_behavior': []
        }
        
        for test in behavior_tests:
            result = self.extract_relations(test['text'])
            category = test['category']
            
            # Analyze performance
            entity_count = len(result.get('entities', []))
            relation_count = len(result.get('relations', []))
            confidence = result.get('confidence', 0.0)
            
            behavior_analysis['category_performance'][category].append({
                'entity_count': entity_count,
                'relation_count': relation_count,
                'confidence': confidence
            })
            
            behavior_analysis['confidence_by_category'][category].append(confidence)
            
            # Pattern recognition
            for entity in result.get('entities', []):
                entity_type = entity.get('type', 'UNKNOWN').upper()
                behavior_analysis['pattern_recognition'][f'entity_{entity_type}'] += 1
            
            for relation in result.get('relations', []):
                predicate = relation.get('predicate', 'UNKNOWN').lower()
                behavior_analysis['pattern_recognition'][f'relation_{predicate}'] += 1
            
            # Detailed behavior
            behavior_analysis['detailed_behavior'].append({
                'text': test['text'],
                'category': category,
                'expected_patterns': test['expected_patterns'],
                'extracted_entities': result.get('entities', []),
                'extracted_relations': result.get('relations', []),
                'confidence': confidence,
                'performance_score': (entity_count + relation_count) / max(len(test['expected_patterns']), 1)
            })
        
        return behavior_analysis

def main():
    """Main evaluation function"""
    
    evaluator = QualityEvaluator()
    
    print("🔍 HOTMEM V3 QUALITY EVALUATION")
    print("="*60)
    
    # Test cases with ground truth
    test_cases = [
        {
            'text': "Tim Cook is the CEO of Apple.",
            'expected_entities': ['Tim Cook', 'Apple'],
            'expected_relations': [
                {'subject': 'Tim Cook', 'predicate': 'CEO_of', 'object': 'Apple'}
            ]
        },
        {
            'text': "Sarah works at Google as a software engineer.",
            'expected_entities': ['Sarah', 'Google'],
            'expected_relations': [
                {'subject': 'Sarah', 'predicate': 'works_for', 'object': 'Google'}
            ]
        },
        {
            'text': "Microsoft is headquartered in Redmond, Washington.",
            'expected_entities': ['Microsoft', 'Redmond', 'Washington'],
            'expected_relations': [
                {'subject': 'Microsoft', 'predicate': 'headquartered_in', 'object': 'Redmond'}
            ]
        },
        {
            'text': "Tesla was founded by Elon Musk.",
            'expected_entities': ['Tesla', 'Elon Musk'],
            'expected_relations': [
                {'subject': 'Tesla', 'predicate': 'founded_by', 'object': 'Elon Musk'}
            ]
        },
        {
            'text': "I bought an iPhone from Apple.",
            'expected_entities': ['iPhone', 'Apple'],
            'expected_relations': [
                {'subject': 'Apple', 'predicate': 'manufactures', 'object': 'iPhone'}
            ]
        },
        {
            'text': "John and Mary are colleagues at Microsoft.",
            'expected_entities': ['John', 'Mary', 'Microsoft'],
            'expected_relations': [
                {'subject': 'John', 'predicate': 'colleagues_with', 'object': 'Mary'},
                {'subject': 'John', 'predicate': 'works_for', 'object': 'Microsoft'},
                {'subject': 'Mary', 'predicate': 'works_for', 'object': 'Microsoft'}
            ]
        },
        {
            'text': "Amazon is based in Seattle, Washington.",
            'expected_entities': ['Amazon', 'Seattle', 'Washington'],
            'expected_relations': [
                {'subject': 'Amazon', 'predicate': 'based_in', 'object': 'Seattle'}
            ]
        },
        {
            'text': "Netflix produces original content.",
            'expected_entities': ['Netflix'],
            'expected_relations': [
                {'subject': 'Netflix', 'predicate': 'produces', 'object': 'original content'}
            ]
        }
    ]
    
    # Run quality evaluation
    print("📊 Running quality evaluation...")
    quality_results = evaluator.evaluate_quality(test_cases)
    
    # Display results
    print(f"\n📈 QUALITY RESULTS:")
    print(f"   Total Tests: {quality_results['total_tests']}")
    print(f"   Correct Extractions: {quality_results['correct_extractions']} ({quality_results['correct_extractions']/quality_results['total_tests']*100:.1f}%)")
    print(f"   Partial Extractions: {quality_results['partial_extractions']} ({quality_results['partial_extractions']/quality_results['total_tests']*100:.1f}%)")
    print(f"   Failed Extractions: {quality_results['failed_extractions']} ({quality_results['failed_extractions']/quality_results['total_tests']*100:.1f}%)")
    print(f"   Entity Accuracy: {quality_results['entity_accuracy']:.2f}")
    print(f"   Relation Accuracy: {quality_results['relation_accuracy']:.2f}")
    print(f"   Average Confidence: {sum(quality_results['confidence_distribution'])/len(quality_results['confidence_distribution']):.2f}")
    
    # Error analysis
    if quality_results['error_analysis']:
        print(f"\n❌ ERROR ANALYSIS:")
        for error in quality_results['error_analysis'][:3]:  # Show first 3 errors
            print(f"   Test {error['test_id']}: {error['text']}")
            for err in error['errors']:
                print(f"     - {err}")
    
    # Behavior analysis
    print(f"\n🧠 BEHAVIOR ANALYSIS:")
    behavior_results = evaluator.analyze_model_behavior()
    
    for category, performances in behavior_results['category_performance'].items():
        avg_entities = sum(p['entity_count'] for p in performances) / len(performances)
        avg_relations = sum(p['relation_count'] for p in performances) / len(performances)
        avg_confidence = sum(p['confidence'] for p in performances) / len(performances)
        
        print(f"   {category.replace('_', ' ').title()}:")
        print(f"     Avg Entities: {avg_entities:.1f}")
        print(f"     Avg Relations: {avg_relations:.1f}")
        print(f"     Avg Confidence: {avg_confidence:.2f}")
    
    # Pattern recognition
    print(f"\n🔍 PATTERN RECOGNITION:")
    sorted_patterns = sorted(behavior_results['pattern_recognition'].items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns[:10]:  # Top 10 patterns
        print(f"   {pattern}: {count}")
    
    # Detailed results for first few tests
    print(f"\n📋 DETAILED RESULTS (First 3):")
    for detail in quality_results['detailed_results'][:3]:
        print(f"   Test {detail['test_id']}: {detail['text']}")
        print(f"     Expected Entities: {detail['expected_entities']}")
        print(f"     Predicted Entities: {detail['predicted_entities']}")
        print(f"     Entity Accuracy: {detail['entity_accuracy']:.2f}")
        print(f"     Relation Accuracy: {detail['relation_accuracy']:.2f}")
        print(f"     Overall Accuracy: {detail['overall_accuracy']:.2f}")
        print(f"     Confidence: {detail['confidence']:.2f}")
        print()
    
    # Save detailed results
    with open('quality_evaluation_results.json', 'w') as f:
        json.dump({
            'quality_results': quality_results,
            'behavior_results': behavior_results
        }, f, indent=2)
    
    print("💾 Detailed results saved to quality_evaluation_results.json")
    print("✅ Quality evaluation completed!")

if __name__ == "__main__":
    main()