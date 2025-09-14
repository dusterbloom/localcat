"""
RoBERTa-based Relation Extraction Fallback for GLiREL
Uses roberta-base-mnli for zero-shot relation classification (fixed batching & indentation)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
from loguru import logger

class RoBERTaRelationExtractor:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            self.model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
            self.device = "cpu"  # Force CPU to avoid MPS bus error
            self.model.to(self.device)
            logger.info(f"[RoBERTa RE] Initialized on {self.device} (base model)")
        except Exception as e:
            logger.error(f"[RoBERTa RE] Init failed: {e}")
            self.tokenizer = None
            self.model = None

    def extract_relations(self, text: str, entities: List[Dict[str, Any]], threshold: float = 0.7) -> List[Dict[str, Any]]:
        if not self.tokenizer or not self.model:
            logger.warning("[RoBERTa RE] Not available")
            return []
        
        relations = []
        relation_labels = [
            "works_at", "founded", "ceo_of", "acquired", "located_in", "born_in",
            "moved_to", "from", "manufactured_by", "costs", "competes_with"
        ]
        
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities[i+1:], i+1):
                hypothesis = [f"{e1['text']} {rel} {e2['text']}." for rel in relation_labels]
                
                try:
                    # Fixed pair encoding: premise str, hypothesis list of str
                    inputs = self.tokenizer([text] * len(hypothesis), hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        entail_scores = probs[:, 2].cpu().numpy()  # Entailment index 2
                    
                    for k, score in enumerate(entail_scores):
                        if score > threshold:
                            relations.append({
                                'subject': e1['text'],
                                'predicate': relation_labels[k],
                                'object': e2['text'],
                                'confidence': float(score)
                            })
                            logger.debug(f"[RoBERTa RE] Found {e1['text']} {relation_labels[k]} {e2['text']} (conf: {score:.2f})")
                except Exception as e:
                    logger.debug(f"[RoBERTa RE] Error for {e1['text']}-{e2['text']}: {e}")
                    continue
        
        return relations

# Global instance
roberta_re = RoBERTaRelationExtractor()

if __name__ == "__main__":
    # Test
    text = "Maria moved to Paris from Barcelona."
    entities = [{'text': 'Maria', 'label': 'PERSON'}, {'text': 'Paris', 'label': 'LOC'}, {'text': 'Barcelona', 'label': 'LOC'}]
    rels = roberta_re.extract_relations(text, entities)
    print(rels)
    print(rels)
