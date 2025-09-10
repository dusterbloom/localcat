"""
HotMem V3 Pipeline Optimization Script
Optimize parameters and improve model performance based on quality evaluation
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization experiments"""
    
    # Model parameters
    model_name: str = "relation-extractor-v2-mlx"
    api_base: str = "http://localhost:1234"
    
    # Prompt optimization
    system_prompt_variants: List[str] = None
    temperature_variants: List[float] = None
    max_tokens_variants: List[int] = None
    
    # Post-processing
    confidence_thresholds: List[float] = None
    entity_filters: List[Dict] = None
    
    def __post_init__(self):
        if self.system_prompt_variants is None:
            self.system_prompt_variants = [
                # Original prompt
                """You are an expert relation extraction system. Extract entities and relations from the given text and output them in JSON format.

Entities should include people, organizations, locations, and other important nouns with their types.
Relations should connect entities with meaningful predicates like 'works_for', 'located_in', 'develops', 'friends_with', etc.

Output format:
{
  "entities": [
    {"text": "Entity Name", "type": "PERSON/ORG/LOC/PRODUCT", "confidence": 0.9}
  ],
  "relations": [
    {"subject": "Entity1", "predicate": "relation_type", "object": "Entity2", "confidence": 0.8}
  ],
  "confidence": 0.85
}

Only output the JSON, no other text.""",
                
                # Detailed prompt
                """You are a highly accurate relation extraction system specializing in conversational text. Your task is to identify entities and their relationships with high precision.

ENTITY TYPES:
- PERSON: Names of people (e.g., "John Smith", "Sarah")
- ORGANIZATION: Companies, institutions (e.g., "Google", "Microsoft")
- LOCATION: Places, cities, addresses (e.g., "San Francisco", "Seattle")
- PRODUCT: Products, services (e.g., "iPhone", "Windows")

RELATION TYPES:
- works_for: Person employed by organization
- located_in: Organization based in location
- develops: Organization creates product
- founded_by: Organization established by person
- friends_with: Personal relationships
- headquartered_in: Organization headquarters location

INSTRUCTIONS:
1. Extract ALL entities mentioned in the text
2. Identify meaningful relationships between entities
3. Assign confidence scores (0.0-1.0) based on certainty
4. Output ONLY valid JSON

Output format:
{
  "entities": [{"text": "Entity", "type": "TYPE", "confidence": 0.9}],
  "relations": [{"subject": "Entity1", "predicate": "relation", "object": "Entity2", "confidence": 0.8}],
  "confidence": 0.85
}""",
                
                # Structured prompt
                """Extract entities and relations from the text following these rules:

ENTITIES: Identify all proper nouns, organization names, locations, and products
RELATIONS: Connect entities with logical relationships (works_for, located_in, develops, etc.)

REQUIREMENTS:
- Extract ALL entities, even if they don't have relations
- Use standard entity types: PERSON, ORG, LOC, PRODUCT
- Use specific relation predicates
- Include confidence scores for each extraction

Output JSON format only:
{"entities": [{"text": "entity", "type": "TYPE", "confidence": 0.9}], "relations": [{"subject": "entity1", "predicate": "relation", "object": "entity2", "confidence": 0.8}], "confidence": 0.85}"""
            ]
        
        if self.temperature_variants is None:
            self.temperature_variants = [0.0, 0.1, 0.2, 0.3]
        
        if self.max_tokens_variants is None:
            self.max_tokens_variants = [256, 512, 768]
        
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.3, 0.5, 0.7, 0.8]

class PipelineOptimizer:
    """Optimize HotMem V3 pipeline parameters"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.api_url = f"{config.api_base}/v1/chat/completions"
        self.best_config = None
        self.best_score = 0.0
    
    def extract_with_config(self, text: str, system_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Extract relations with specific configuration"""
        
        try:
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract entities and relations from this text:\n\n{text}"}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
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
                    extraction_result = json.loads(json_str)
                    
                    # Post-process based on confidence threshold
                    return self.post_process_extraction(extraction_result)
            
            return {"entities": [], "relations": [], "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"entities": [], "relations": [], "confidence": 0.0}
    
    def post_process_extraction(self, extraction: Dict[str, Any], confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Post-process extraction results"""
        
        # Filter entities by confidence
        filtered_entities = [
            entity for entity in extraction.get('entities', [])
            if entity.get('confidence', 0.0) >= confidence_threshold
        ]
        
        # Filter relations by confidence
        filtered_relations = [
            relation for relation in extraction.get('relations', [])
            if relation.get('confidence', 0.0) >= confidence_threshold
        ]
        
        # Only keep relations where both entities exist
        entity_texts = {entity.get('text', '').lower() for entity in filtered_entities}
        valid_relations = []
        
        for relation in filtered_relations:
            subject = relation.get('subject', '').lower()
            obj = relation.get('object', '').lower()
            
            if subject in entity_texts and obj in entity_texts:
                valid_relations.append(relation)
        
        return {
            "entities": filtered_entities,
            "relations": valid_relations,
            "confidence": extraction.get('confidence', 0.0)
        }
    
    def evaluate_configuration(self, test_cases: List[Dict], config: Dict) -> Dict[str, Any]:
        """Evaluate a specific configuration"""
        
        system_prompt = config['system_prompt']
        temperature = config['temperature']
        max_tokens = config['max_tokens']
        confidence_threshold = config['confidence_threshold']
        
        total_score = 0.0
        entity_scores = []
        relation_scores = []
        processing_times = []
        
        for test_case in test_cases:
            text = test_case['text']
            expected_entities = test_case.get('expected_entities', [])
            expected_relations = test_case.get('expected_relations', [])
            
            # Extract with configuration
            start_time = time.time()
            result = self.extract_with_config(text, system_prompt, temperature, max_tokens)
            result = self.post_process_extraction(result, confidence_threshold)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Calculate scores
            predicted_entities = [e.get('text', '').lower() for e in result.get('entities', [])]
            expected_entities_lower = [e.lower() for e in expected_entities]
            
            # Entity score
            if expected_entities_lower:
                entity_matches = sum(1 for e in expected_entities_lower if e in predicted_entities)
                entity_score = entity_matches / len(expected_entities_lower)
            else:
                entity_score = 1.0 if not predicted_entities else 0.0
            
            # Relation score
            predicted_relations = []
            for rel in result.get('relations', []):
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
            
            if expected_relations_lower:
                relation_matches = 0
                for er in expected_relations_lower:
                    er_str = f"{er['subject']}|{er['predicate']}|{er['object']}"
                    for pr in predicted_relations:
                        pr_str = f"{pr['subject']}|{pr['predicate']}|{pr['object']}"
                        if er_str == pr_str:
                            relation_matches += 1
                            break
                relation_score = relation_matches / len(expected_relations_lower)
            else:
                relation_score = 1.0 if not predicted_relations else 0.0
            
            # Combined score with processing time penalty
            combined_score = (entity_score + relation_score) / 2
            time_penalty = min(processing_time * 0.1, 0.1)  # 10% penalty per second, max 10%
            final_score = max(0.0, combined_score - time_penalty)
            
            total_score += final_score
            entity_scores.append(entity_score)
            relation_scores.append(relation_score)
        
        avg_score = total_score / len(test_cases)
        avg_entity_score = sum(entity_scores) / len(entity_scores)
        avg_relation_score = sum(relation_scores) / len(relation_scores)
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        return {
            'avg_score': avg_score,
            'avg_entity_score': avg_entity_score,
            'avg_relation_score': avg_relation_score,
            'avg_processing_time': avg_processing_time,
            'config': config
        }
    
    def run_optimization(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Run full optimization process"""
        
        print("🔧 STARTING PIPELINE OPTIMIZATION")
        print("="*60)
        
        # Generate all configurations
        configurations = []
        
        for i, system_prompt in enumerate(self.config.system_prompt_variants):
            for temperature in self.config.temperature_variants:
                for max_tokens in self.config.max_tokens_variants:
                    for confidence_threshold in self.config.confidence_thresholds:
                        config = {
                            'system_prompt_id': i,
                            'system_prompt': system_prompt,
                            'temperature': temperature,
                            'max_tokens': max_tokens,
                            'confidence_threshold': confidence_threshold
                        }
                        configurations.append(config)
        
        print(f"📊 Testing {len(configurations)} configurations...")
        
        # Evaluate each configuration
        results = []
        for i, config in enumerate(configurations):
            print(f"🧪 Testing configuration {i+1}/{len(configurations)}...")
            
            result = self.evaluate_configuration(test_cases, config)
            results.append(result)
            
            # Update best configuration
            if result['avg_score'] > self.best_score:
                self.best_score = result['avg_score']
                self.best_config = config
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(configurations)} tested")
                print(f"   Current best score: {self.best_score:.3f}")
        
        # Sort results by score
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'top_results': results[:10],
            'all_results': results
        }
    
    def validate_best_config(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Validate the best configuration on additional test cases"""
        
        print(f"\n✅ VALIDATING BEST CONFIGURATION")
        print("="*60)
        
        if not self.best_config:
            print("❌ No configuration to validate")
            return {}
        
        # Run validation
        validation_result = self.evaluate_configuration(test_cases, self.best_config)
        
        print(f"📈 Validation Results:")
        print(f"   Average Score: {validation_result['avg_score']:.3f}")
        print(f"   Entity Score: {validation_result['avg_entity_score']:.3f}")
        print(f"   Relation Score: {validation_result['avg_relation_score']:.3f}")
        print(f"   Processing Time: {validation_result['avg_processing_time']:.3f}s")
        
        return validation_result
    
    def generate_optimized_config(self) -> Dict[str, Any]:
        """Generate optimized configuration for HotMem V3"""
        
        if not self.best_config:
            return {}
        
        optimized_config = {
            'model_name': self.config.model_name,
            'api_base': self.config.api_base,
            'system_prompt': self.best_config['system_prompt'],
            'temperature': self.best_config['temperature'],
            'max_tokens': self.best_config['max_tokens'],
            'confidence_threshold': self.best_config['confidence_threshold'],
            'performance_metrics': {
                'optimized_score': self.best_score,
                'validation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return optimized_config

def main():
    """Main optimization function"""
    
    # Test cases for optimization
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
        }
    ]
    
    # Validation cases
    validation_cases = [
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
        },
        {
            'text': "John and Mary work at Microsoft together.",
            'expected_entities': ['John', 'Mary', 'Microsoft'],
            'expected_relations': [
                {'subject': 'John', 'predicate': 'works_for', 'object': 'Microsoft'},
                {'subject': 'Mary', 'predicate': 'works_for', 'object': 'Microsoft'}
            ]
        }
    ]
    
    # Initialize optimizer
    config = OptimizationConfig()
    optimizer = PipelineOptimizer(config)
    
    # Run optimization
    optimization_results = optimizer.run_optimization(test_cases)
    
    # Display results
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"   Best Score: {optimization_results['best_score']:.3f}")
    print(f"   Best Configuration:")
    print(f"     System Prompt ID: {optimizer.best_config['system_prompt_id']}")
    print(f"     Temperature: {optimizer.best_config['temperature']}")
    print(f"     Max Tokens: {optimizer.best_config['max_tokens']}")
    print(f"     Confidence Threshold: {optimizer.best_config['confidence_threshold']}")
    
    print(f"\n🏆 TOP 5 CONFIGURATIONS:")
    for i, result in enumerate(optimization_results['top_results'][:5], 1):
        config = result['config']
        print(f"   {i}. Score: {result['avg_score']:.3f}")
        print(f"      Prompt ID: {config['system_prompt_id']}, Temp: {config['temperature']}, "
              f"Tokens: {config['max_tokens']}, Threshold: {config['confidence_threshold']}")
        print(f"      Entity: {result['avg_entity_score']:.3f}, Relation: {result['avg_relation_score']:.3f}, "
              f"Time: {result['avg_processing_time']:.3f}s")
    
    # Validate best configuration
    validation_result = optimizer.validate_best_config(validation_cases)
    
    # Generate optimized configuration
    optimized_config = optimizer.generate_optimized_config()
    
    # Save results
    with open('optimization_results.json', 'w') as f:
        json.dump({
            'optimization_results': optimization_results,
            'validation_result': validation_result,
            'optimized_config': optimized_config
        }, f, indent=2)
    
    print(f"\n💾 Results saved to optimization_results.json")
    print("✅ Optimization completed!")

if __name__ == "__main__":
    main()