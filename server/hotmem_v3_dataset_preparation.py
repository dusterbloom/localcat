"""
HotMem V3 Dataset Preparation
Following the exact recipe from HOT MEM V3 document

Dataset Mix:
- 40% REBEL - General knowledge graphs (40K samples)
- 25% DialogRE - Conversational relations (30K samples)  
- 15% ConvQuestions - Coreference resolution (15K samples)
- 10% Wizard of Wikipedia - Natural dialogue (10K samples)
- 10% MultiWOZ - Task-oriented dialogue (10K samples)
"""

import json
import logging
from typing import List, Dict, Any, Set, Tuple
from datasets import load_dataset, Dataset
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotMemDatasetPreparer:
    """Prepare dataset mix exactly per HOT MEM V3 recipe"""
    
    def __init__(self, cache_dir: str = "./dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_rebel_dataset(self, limit: int = 40000) -> List[Dict]:
        """Load REBEL dataset - 40% of training mix"""
        logger.info(f"Loading REBEL dataset (limit: {limit})")
        
        dataset = load_dataset("Babelscape/rebel-dataset", split="train")
        processed_data = []
        
        for item in dataset[:limit]:
            triples = item.get('triples', [])
            text = item.get('text', '')
            
            if not text or not triples:
                continue
                
            # Extract unique entities from triples
            entities = set()
            relations = []
            
            for triple in triples:
                if isinstance(triple, dict):
                    head = triple.get('head', '').strip()
                    tail = triple.get('tail', '').strip()
                    rel_type = triple.get('type', '').strip()
                    
                    if head and tail and rel_type:
                        entities.add(head)
                        entities.add(tail)
                        relations.append({
                            'subject': head,
                            'predicate': rel_type,
                            'object': tail
                        })
            
            if entities and relations:
                processed_data.append({
                    'text': text,
                    'entities': list(entities),
                    'relations': relations,
                    'domain': 'general',
                    'source': 'rebel',
                    'confidence': 0.95
                })
        
        logger.info(f"Processed {len(processed_data)} REBEL examples")
        return processed_data
    
    def load_dialogre_dataset(self, limit: int = 30000) -> List[Dict]:
        """Load DialogRE dataset - 25% of training mix"""
        logger.info(f"Loading DialogRE dataset (limit: {limit})")
        
        dataset = load_dataset("dialogre", split="train")
        processed_data = []
        
        for item in dataset[:limit]:
            dialogue = item.get('dialogue', [])
            relations = item.get('relations', [])
            
            if not dialogue or not relations:
                continue
                
            # Extract entities from relations
            entities = set()
            formatted_relations = []
            
            for rel in relations:
                if len(rel) >= 3:
                    subject = rel[0].strip() if rel[0] else ''
                    predicate = rel[1].strip() if rel[1] else ''
                    obj = rel[2].strip() if rel[2] else ''
                    
                    if subject and predicate and obj:
                        entities.add(subject)
                        entities.add(obj)
                        formatted_relations.append({
                            'subject': subject,
                            'predicate': predicate,
                            'object': obj
                        })
            
            # Join dialogue turns into single text
            text = ' '.join(dialogue)
            
            if entities and formatted_relations:
                processed_data.append({
                    'text': text,
                    'entities': list(entities),
                    'relations': formatted_relations,
                    'domain': 'conversation',
                    'source': 'dialogre',
                    'confidence': 0.90
                })
        
        logger.info(f"Processed {len(processed_data)} DialogRE examples")
        return processed_data
    
    def load_convquestions_dataset(self, limit: int = 15000) -> List[Dict]:
        """Load ConvQuestions dataset - 15% of training mix"""
        logger.info(f"Loading ConvQuestions dataset (limit: {limit})")
        
        dataset = load_dataset("convquestions", split="train")
        processed_data = []
        
        for item in dataset[:limit]:
            conversation = item.get('conversation', [])
            entities_tracked = item.get('entities_tracked', [])
            coreferences = item.get('coreferences', [])
            
            if not conversation or not entities_tracked:
                continue
                
            # Build text from conversation
            text = ' '.join(conversation)
            
            # Extract relations from coreferences
            relations = []
            for coref in coreferences:
                if len(coref) == 2:
                    pronoun, entity = coref
                    relations.append({
                        'subject': pronoun,
                        'predicate': 'refers_to',
                        'object': entity,
                        'confidence': 0.85
                    })
            
            # Add co-occurrence relations
            for i, entity1 in enumerate(entities_tracked):
                for entity2 in entities_tracked[i+1:]:
                    if entity1 in text and entity2 in text:
                        relations.append({
                            'subject': entity1,
                            'predicate': 'co_occurs_with',
                            'object': entity2,
                            'confidence': 0.70
                        })
            
            processed_data.append({
                'text': text,
                'entities': entities_tracked,
                'relations': relations,
                'domain': 'conversation',
                'source': 'convquestions',
                'confidence': 0.80
            })
        
        logger.info(f"Processed {len(processed_data)} ConvQuestions examples")
        return processed_data
    
    def load_wizard_dataset(self, limit: int = 10000) -> List[Dict]:
        """Load Wizard of Wikipedia dataset - 10% of training mix"""
        logger.info(f"Loading Wizard of Wikipedia dataset (limit: {limit})")
        
        dataset = load_dataset("wizard_of_wikipedia", split="train")
        processed_data = []
        
        for item in dataset[:limit]:
            dialog = item.get('dialog', [])
            knowledge = item.get('knowledge', [])
            chosen_topic = item.get('chosen_topic', '')
            
            if not dialog:
                continue
                
            # Build text from dialogue
            text = ' '.join(dialog)
            
            # Extract entities from knowledge
            entities = set()
            relations = []
            
            # Add topic as main entity
            if chosen_topic:
                entities.add(chosen_topic)
            
            # Process knowledge statements
            for stmt in knowledge:
                # Simple entity extraction (can be enhanced)
                words = stmt.split()
                for i, word in enumerate(words):
                    if word[0].isupper() and len(word) > 2:
                        entities.add(word)
                        
                        # Add topic relation
                        if chosen_topic and chosen_topic.lower() in stmt.lower():
                            relations.append({
                                'subject': word,
                                'predicate': 'related_to',
                                'object': chosen_topic,
                                'confidence': 0.75
                            })
            
            processed_data.append({
                'text': text,
                'entities': list(entities),
                'relations': relations,
                'domain': 'dialogue',
                'source': 'wizard',
                'confidence': 0.75
            })
        
        logger.info(f"Processed {len(processed_data)} Wizard examples")
        return processed_data
    
    def load_multiwoz_dataset(self, limit: int = 10000) -> List[Dict]:
        """Load MultiWOZ dataset - 10% of training mix"""
        logger.info(f"Loading MultiWOZ dataset (limit: {limit})")
        
        dataset = load_dataset("multi_woz_v22", split="train")
        processed_data = []
        
        for item in dataset[:limit]:
            dialogue = item.get('dialogue', [])
            state = item.get('state', {})
            
            if not dialogue:
                continue
                
            # Extract entities and relations from state
            entities = set()
            relations = []
            
            # Process dialogue state
            for domain, domain_state in state.items():
                if isinstance(domain_state, dict):
                    entities.add(domain)
                    
                    for slot, value in domain_state.items():
                        if value and value != 'none':
                            entities.add(value)
                            relations.append({
                                'subject': domain,
                                'predicate': f'has_{slot}',
                                'object': value,
                                'confidence': 0.90
                            })
            
            # Add dialogue text
            text = ' '.join(dialogue) if isinstance(dialogue, list) else dialogue
            
            processed_data.append({
                'text': text,
                'entities': list(entities),
                'relations': relations,
                'domain': 'task_oriented',
                'source': 'multiwoz',
                'confidence': 0.85
            })
        
        logger.info(f"Processed {len(processed_data)} MultiWOZ examples")
        return processed_data
    
    def create_training_mix(self) -> List[Dict]:
        """Create the exact training mix per HOT MEM V3 recipe"""
        logger.info("Creating HotMem V3 training mix...")
        
        # Load all datasets with exact proportions
        rebel_data = self.load_rebel_dataset(40000)      # 40%
        dialogre_data = self.load_dialogre_dataset(30000) # 25%  
        convquestions_data = self.load_convquestions_dataset(15000) # 15%
        wizard_data = self.load_wizard_dataset(10000)     # 10%
        multiwoz_data = self.load_multiwoz_dataset(10000) # 10%
        
        # Combine all data
        all_data = (
            rebel_data + 
            dialogre_data + 
            convquestions_data + 
            wizard_data + 
            multiwoz_data
        )
        
        logger.info(f"Total training examples: {len(all_data)}")
        logger.info("Dataset distribution:")
        logger.info(f"  - REBEL: {len(rebel_data)} ({len(rebel_data)/len(all_data)*100:.1f}%)")
        logger.info(f"  - DialogRE: {len(dialogre_data)} ({len(dialogre_data)/len(all_data)*100:.1f}%)")
        logger.info(f"  - ConvQuestions: {len(convquestions_data)} ({len(convquestions_data)/len(all_data)*100:.1f}%)")
        logger.info(f"  - Wizard: {len(wizard_data)} ({len(wizard_data)/len(all_data)*100:.1f}%)")
        logger.info(f"  - MultiWOZ: {len(multiwoz_data)} ({len(multiwoz_data)/len(all_data)*100:.1f}%)")
        
        return all_data
    
    def save_training_data(self, data: List[Dict], filename: str):
        """Save training data to file"""
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved training data to {filepath}")
        
        # Also save as HuggingFace dataset
        hf_dataset = Dataset.from_list(data)
        hf_path = self.cache_dir / f"{filename}.hf"
        hf_dataset.save_to_disk(str(hf_path))
        logger.info(f"Saved HuggingFace dataset to {hf_path}")
    
    def load_cached_data(self, filename: str) -> List[Dict]:
        """Load cached training data if exists"""
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            logger.info(f"Loading cached data from {filepath}")
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None

def main():
    """Main execution function"""
    preparer = HotMemDatasetPreparer()
    
    # Check if cached data exists
    cached_data = preparer.load_cached_data("hotmem_v3_training_mix.json")
    
    if cached_data:
        logger.info("Using cached training data")
        training_data = cached_data
    else:
        logger.info("Preparing fresh training mix")
        training_data = preparer.create_training_mix()
        preparer.save_training_data(training_data, "hotmem_v3_training_mix.json")
    
    # Print statistics
    logger.info(f"Final training dataset size: {len(training_data)}")
    
    # Analyze domain distribution
    domains = {}
    for item in training_data:
        domain = item.get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1
    
    logger.info("Domain distribution:")
    for domain, count in domains.items():
        logger.info(f"  - {domain}: {count} ({count/len(training_data)*100:.1f}%)")
    
    return training_data

if __name__ == "__main__":
    training_data = main()