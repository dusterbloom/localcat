"""
HotMem V3 Streaming Augmentation
Creates partial sentence examples for real-time graph extraction during voice conversations

Key concept: Train the model to extract entities and relations from incomplete sentences
as they would appear in real-time voice streaming.
"""

import json
import random
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingPoint:
    """Represents a point in the streaming process"""
    text: str
    entities: List[str]
    relations: List[Dict[str, Any]]
    is_complete: bool
    confidence: float
    position: int  # Word position in original sentence
    streaming_type: str  # 'word_by_word', 'phrase_by_phrase', 'clause_by_clause'

class StreamingAugmentor:
    """Creates streaming augmentation examples for real-time training"""
    
    def __init__(self, 
                 word_probability: float = 0.7,
                 phrase_probability: float = 0.8,
                 clause_probability: float = 0.9,
                 max_streaming_points: int = 10):
        self.word_probability = word_probability
        self.phrase_probability = phrase_probability
        self.clause_probability = clause_probability
        self.max_streaming_points = max_streaming_points
        
        # Sentence boundary patterns
        self.clause_delimiters = r'[.,;:!?]'
        self.phrase_delimiters = r'[,;]'
        
    def extract_words(self, text: str) -> List[str]:
        """Extract words preserving punctuation"""
        # Simple word tokenization that preserves punctuation
        words = re.findall(r'\w+|[^\w\s]', text)
        return words
    
    def extract_phrases(self, text: str) -> List[str]:
        """Extract phrases based on natural breaks"""
        # Split on commas, semicolons, and coordinating conjunctions
        phrases = re.split(r'[,;]|\s+(?:and|or|but|yet|for|nor|so)\s+', text)
        return [p.strip() for p in phrases if p.strip()]
    
    def extract_clauses(self, text: str) -> List[str]:
        """Extract clauses based on sentence boundaries"""
        clauses = re.split(r'[.!?]+\s*', text)
        return [c.strip() for c in clauses if c.strip()]
    
    def is_entity_in_text(self, entity: str, text: str) -> bool:
        """Check if entity appears in text (case-insensitive)"""
        return entity.lower() in text.lower()
    
    def is_relation_in_text(self, relation: Dict[str, Any], text: str) -> bool:
        """Check if relation can be extracted from partial text"""
        subject = relation.get('subject', '')
        obj = relation.get('object', '')
        
        # Both subject and object must be present
        return (self.is_entity_in_text(subject, text) and 
                self.is_entity_in_text(obj, text))
    
    def create_word_by_word_streaming(self, example: Dict[str, Any]) -> List[StreamingPoint]:
        """Create streaming points word by word"""
        words = self.extract_words(example['text'])
        streaming_points = []
        
        # Sample which word positions to create points for
        num_points = min(self.max_streaming_points, len(words))
        selected_positions = sorted(random.sample(
            range(len(words)), 
            min(num_points, int(len(words) * self.word_probability))
        ))
        
        for pos in selected_positions:
            partial_text = ' '.join(words[:pos+1])
            
            # Determine what's extractable at this point
            partial_entities = [
                e for e in example['entities'] 
                if self.is_entity_in_text(e, partial_text)
            ]
            
            partial_relations = [
                r for r in example['relations']
                if self.is_relation_in_text(r, partial_text)
            ]
            
            streaming_points.append(StreamingPoint(
                text=partial_text,
                entities=partial_entities,
                relations=partial_relations,
                is_complete=(pos == len(words) - 1),
                confidence=0.6 + (pos / len(words)) * 0.4,  # Increases as text becomes more complete
                position=pos,
                streaming_type='word_by_word'
            ))
        
        return streaming_points
    
    def create_phrase_by_phrase_streaming(self, example: Dict[str, Any]) -> List[StreamingPoint]:
        """Create streaming points phrase by phrase"""
        phrases = self.extract_phrases(example['text'])
        streaming_points = []
        
        # Sample which phrase positions to create points for
        num_points = min(self.max_streaming_points, len(phrases))
        selected_positions = sorted(random.sample(
            range(len(phrases)), 
            min(num_points, int(len(phrases) * self.phrase_probability))
        ))
        
        for pos in selected_positions:
            partial_text = ' '.join(phrases[:pos+1])
            
            # Determine what's extractable at this point
            partial_entities = [
                e for e in example['entities'] 
                if self.is_entity_in_text(e, partial_text)
            ]
            
            partial_relations = [
                r for r in example['relations']
                if self.is_relation_in_text(r, partial_text)
            ]
            
            streaming_points.append(StreamingPoint(
                text=partial_text,
                entities=partial_entities,
                relations=partial_relations,
                is_complete=(pos == len(phrases) - 1),
                confidence=0.7 + (pos / len(phrases)) * 0.3,
                position=pos,
                streaming_type='phrase_by_phrase'
            ))
        
        return streaming_points
    
    def create_clause_by_clause_streaming(self, example: Dict[str, Any]) -> List[StreamingPoint]:
        """Create streaming points clause by clause"""
        clauses = self.extract_clauses(example['text'])
        streaming_points = []
        
        # Sample which clause positions to create points for
        num_points = min(self.max_streaming_points, len(clauses))
        selected_positions = sorted(random.sample(
            range(len(clauses)), 
            min(num_points, int(len(clauses) * self.clause_probability))
        ))
        
        for pos in selected_positions:
            partial_text = ' '.join(clauses[:pos+1])
            
            # Determine what's extractable at this point
            partial_entities = [
                e for e in example['entities'] 
                if self.is_entity_in_text(e, partial_text)
            ]
            
            partial_relations = [
                r for r in example['relations']
                if self.is_relation_in_text(r, partial_text)
            ]
            
            streaming_points.append(StreamingPoint(
                text=partial_text,
                entities=partial_entities,
                relations=partial_relations,
                is_complete=(pos == len(clauses) - 1),
                confidence=0.8 + (pos / len(clauses)) * 0.2,
                position=pos,
                streaming_type='clause_by_clause'
            ))
        
        return streaming_points
    
    def create_mixed_streaming(self, example: Dict[str, Any]) -> List[StreamingPoint]:
        """Create mixed streaming points combining different granularities"""
        all_points = []
        
        # Randomly choose streaming type for this example
        streaming_types = ['word_by_word', 'phrase_by_phrase', 'clause_by_clause']
        chosen_type = random.choice(streaming_types)
        
        if chosen_type == 'word_by_word':
            all_points.extend(self.create_word_by_word_streaming(example))
        elif chosen_type == 'phrase_by_phrase':
            all_points.extend(self.create_phrase_by_phrase_streaming(example))
        else:
            all_points.extend(self.create_clause_by_clause_streaming(example))
        
        return all_points
    
    def create_progressive_streaming(self, example: Dict[str, Any]) -> List[StreamingPoint]:
        """Create progressive streaming that simulates real conversation"""
        words = self.extract_words(example['text'])
        streaming_points = []
        
        # Create points at natural breakpoints
        natural_breaks = []
        
        # Break at punctuation
        for i, word in enumerate(words):
            if word in ['.', ',', ';', '!', '?']:
                natural_breaks.append(i)
        
        # Break at conjunctions
        conjunctions = ['and', 'or', 'but', 'yet', 'for', 'nor', 'so']
        for i, word in enumerate(words):
            if word.lower() in conjunctions and i > 0:
                natural_breaks.append(i-1)
        
        # Break every N words for longer sentences
        if len(words) > 20:
            for i in range(10, len(words), 10):
                natural_breaks.append(i)
        
        # Remove duplicates and sort
        natural_breaks = sorted(set(natural_breaks))
        
        # Limit number of points
        if len(natural_breaks) > self.max_streaming_points:
            natural_breaks = natural_breaks[:self.max_streaming_points]
        
        for break_pos in natural_breaks:
            partial_text = ' '.join(words[:break_pos+1])
            
            # Determine what's extractable at this point
            partial_entities = [
                e for e in example['entities'] 
                if self.is_entity_in_text(e, partial_text)
            ]
            
            partial_relations = [
                r for r in example['relations']
                if self.is_relation_in_text(r, partial_text)
            ]
            
            streaming_points.append(StreamingPoint(
                text=partial_text,
                entities=partial_entities,
                relations=partial_relations,
                is_complete=(break_pos == len(words) - 1),
                confidence=0.5 + (break_pos / len(words)) * 0.5,
                position=break_pos,
                streaming_type='progressive'
            ))
        
        return streaming_points
    
    def augment_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment a single example with streaming variants"""
        streaming_examples = []
        
        # Choose augmentation strategy
        strategies = [
            'word_by_word',
            'phrase_by_phrase', 
            'clause_by_clause',
            'mixed',
            'progressive'
        ]
        
        strategy = random.choice(strategies)
        
        if strategy == 'word_by_word':
            points = self.create_word_by_word_streaming(example)
        elif strategy == 'phrase_by_phrase':
            points = self.create_phrase_by_phrase_streaming(example)
        elif strategy == 'clause_by_clause':
            points = self.create_clause_by_clause_streaming(example)
        elif strategy == 'mixed':
            points = self.create_mixed_streaming(example)
        else:  # progressive
            points = self.create_progressive_streaming(example)
        
        # Convert streaming points to training examples
        for point in points:
            streaming_example = {
                'text': point.text,
                'entities': point.entities,
                'relations': point.relations,
                'domain': example.get('domain', 'general'),
                'source': example.get('source', 'streaming'),
                'confidence': point.confidence,
                'is_streaming': True,
                'is_complete': point.is_complete,
                'streaming_type': point.streaming_type,
                'position': point.position,
                'original_text': example['text'],
                'original_entities': example['entities'],
                'original_relations': example['relations']
            }
            streaming_examples.append(streaming_example)
        
        return streaming_examples
    
    def augment_dataset(self, dataset: List[Dict[str, Any]], 
                       augmentation_factor: float = 2.0) -> List[Dict[str, Any]]:
        """Augment entire dataset with streaming examples"""
        logger.info(f"Augmenting dataset with streaming examples (factor: {augmentation_factor})")
        
        augmented_data = []
        
        # Keep all original examples
        augmented_data.extend(dataset)
        
        # Add streaming variants
        num_to_augment = int(len(dataset) * augmentation_factor)
        selected_examples = random.sample(dataset, min(num_to_augment, len(dataset)))
        
        for example in selected_examples:
            streaming_examples = self.augment_example(example)
            augmented_data.extend(streaming_examples)
        
        logger.info(f"Original dataset: {len(dataset)} examples")
        logger.info(f"Augmented dataset: {len(augmented_data)} examples")
        logger.info(f"Augmentation ratio: {len(augmented_data)/len(dataset):.2f}x")
        
        return augmented_data
    
    def save_augmented_data(self, data: List[Dict[str, Any]], filename: str):
        """Save augmented data to file"""
        filepath = Path(f"./dataset_cache/{filename}")
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved augmented data to {filepath}")

def main():
    """Test the streaming augmentation system"""
    # Test example
    test_example = {
        'text': 'Steve Jobs founded Apple in Cupertino and later created the iPhone.',
        'entities': ['Steve Jobs', 'Apple', 'Cupertino', 'iPhone'],
        'relations': [
            {'subject': 'Steve Jobs', 'predicate': 'founded', 'object': 'Apple'},
            {'subject': 'Apple', 'predicate': 'headquarters_in', 'object': 'Cupertino'},
            {'subject': 'Steve Jobs', 'predicate': 'created', 'object': 'iPhone'}
        ],
        'domain': 'general',
        'source': 'test'
    }
    
    # Create augmentor
    augmentor = StreamingAugmentor(
        word_probability=0.8,
        phrase_probability=0.9,
        clause_probability=1.0,
        max_streaming_points=8
    )
    
    # Augment the example
    streaming_examples = augmentor.augment_example(test_example)
    
    print(f"Generated {len(streaming_examples)} streaming examples:")
    for i, example in enumerate(streaming_examples[:3]):  # Show first 3
        print(f"\nExample {i+1}:")
        print(f"  Text: {example['text']}")
        print(f"  Entities: {example['entities']}")
        print(f"  Relations: {len(example['relations'])}")
        print(f"  Complete: {example['is_complete']}")
        print(f"  Confidence: {example['confidence']:.2f}")
        print(f"  Type: {example['streaming_type']}")

if __name__ == "__main__":
    main()