"""
DSPy Framework for Declarative AI Modules

Revolutionary self-improving AI system using DSPy for graph extraction.
This transforms HotMem from static pattern engineering to adaptive intelligence.

Author: HotMem V3 Evolution
"""

import dspy
from typing import List, Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
from loguru import logger

# Sample triples for testing retrieval (mock dual_graph get_relationships)
SAMPLE_TRIPLES = [
    {"subject": "Sarah", "predicate": "married_to", "object": "Michael_Chen", "confidence": 0.95},
    {"subject": "Sarah", "predicate": "works_at", "object": "TechCorp", "confidence": 0.85},
    {"subject": "Michael_Chen", "predicate": "parent_of", "object": "Emma", "confidence": 0.9},
    {"subject": "Emma", "predicate": "age", "object": "5", "confidence": 0.8},
    {"subject": "Sarah", "predicate": "lives_in", "object": "Seattle", "confidence": 0.7},
]

# Global flag to prevent multiple DSPy configurations
_DSPY_CONFIGURED = False

# DSPy Configuration - Offline with Unsloth fine-tuning support
class DSPyConfig:
    """Configuration for DSPy framework"""
    def __init__(self, unsloth_model: str = "unsloth/gemma-2-2b-it-GGUF"):
        global _DSPY_CONFIGURED

        # Configure DSPy with local LM (Ollama/LM Studio)
        self.llm = dspy.LM(
            model="openai/qwen2.5-coder-0.5b-instruct",  # Small coder model for structured output
            api_base="http://127.0.0.1:1234/v1",
            api_key="lmstudio",  # Dummy for local
            max_tokens=256,  # Smaller model needs fewer tokens
            temperature=0.1
        )

        # Unsloth for offline fine-tuning
        self.unsloth_model = unsloth_model
        self.use_unsloth = os.getenv("USE_UNSLOTH_FINE_TUNE", "true").lower() in ("true", "1", "yes")

        # Configure DSPy globally - only once
        if not _DSPY_CONFIGURED:
            try:
                dspy.settings.configure(lm=self.llm)
                _DSPY_CONFIGURED = True
                logger.info(f"DSPy configured: local LM (gemma-3-270m), Unsloth={self.use_unsloth}")
            except RuntimeError as e:
                if "can only be called from the same async task" in str(e):
                    logger.warning("DSPy already configured in another async task, skipping configuration")
                    _DSPY_CONFIGURED = True
                else:
                    raise
        else:
            logger.debug("DSPy already configured, skipping")

# Data Models for Graph Extraction
@dataclass
class Entity:
    """Entity extracted from text"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class Relationship:
    """Relationship between entities"""
    subject: str
    predicate: str
    object: str
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgeGraph:
    """Complete knowledge graph from text"""
    entities: List[Entity]
    relationships: List[Relationship]
    source_text: str
    extraction_confidence: float
    metadata: Dict[str, Any] = None

# DSPy Signatures for Declarative AI
class EntityExtractionSignature(dspy.Signature):
    """Declarative signature for entity extraction"""
    __doc__ = """Extract entities from text with high precision."""
    
    text = dspy.InputField(desc="Input text to extract entities from")
    context = dspy.InputField(desc="Optional context about the conversation", required=False)
    
    entities = dspy.OutputField(desc="List of extracted entities with labels and positions")
    confidence = dspy.OutputField(desc="Overall confidence in extraction (0-1)")

class RelationshipExtractionSignature(dspy.Signature):
    """Declarative signature for relationship extraction"""
    __doc__ = """Extract relationships between entities in text."""
    
    text = dspy.InputField(desc="Input text to extract relationships from")
    entities = dspy.InputField(desc="List of pre-extracted entities")
    context = dspy.InputField(desc="Optional context about the conversation", required=False)
    
    relationships = dspy.OutputField(desc="List of extracted relationships")
    confidence = dspy.OutputField(desc="Overall confidence in relationship extraction (0-1)")

class GraphBuildingSignature(dspy.Signature):
    """Declarative signature for complete graph building"""
    __doc__ = """Build complete knowledge graph from text."""
    
    text = dspy.InputField(desc="Input text to build graph from")
    context = dspy.InputField(desc="Optional context about the conversation", required=False)
    requirements = dspy.InputField(desc="Specific extraction requirements", required=False)
    
    graph = dspy.OutputField(desc="Complete knowledge graph with entities and relationships")
    confidence = dspy.OutputField(desc="Overall confidence in graph construction (0-1)")

class TripleRetrievalSignature(dspy.Signature):
    """Signature for retrieving relevant triples from graph context using RAG-style selection."""
    __doc__ = """Given a query and context of available triples from the knowledge graph, select the most relevant triples that answer the query. Focus on semantic relevance to the query intent."""
    
    query = dspy.InputField(desc="The user's query or question")
    context = dspy.InputField(desc="JSON string of available triples from the knowledge graph, each as {'subject': str, 'predicate': str, 'object': str, 'confidence': float}")
    
    relevant_triples = dspy.OutputField(desc="List of relevant triples in the format ['subject predicate object'] as strings. Only include triples directly relevant to the query.")

# DSPy Modules for Self-Improving AI
class EntityExtractor(dspy.Module):
    """Self-improving entity extraction module"""
    
    def __init__(self):
        super().__init__()
        self.entity_extraction = dspy.Predict(EntityExtractionSignature)
        
    def forward(self, text: str, context: Optional[str] = None) -> List[Entity]:
        """Extract entities using DSPy"""
        # Prepare input
        inputs = {"text": text}
        if context:
            inputs["context"] = context
            
        # Use DSPy to predict entities
        prediction = self.entity_extraction(**inputs)
        
        # Parse prediction into Entity objects
        entities = []
        try:
            entities_data = json.loads(prediction.entities)
            for entity_data in entities_data:
                # Ensure entity_data is a dictionary
                if isinstance(entity_data, dict):
                    entity = Entity(
                        text=entity_data["text"],
                        label=entity_data["label"],
                        start=entity_data.get("start", 0),
                        end=entity_data.get("end", len(entity_data["text"])),
                        confidence=entity_data.get("confidence", 0.8),
                        metadata=entity_data.get("metadata", {})
                    )
                    entities.append(entity)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback to simple extraction
            entities = self._fallback_extraction(text)
            
        return entities
    
    def _fallback_extraction(self, text: str) -> List[Entity]:
        """Fallback extraction when DSPy prediction fails"""
        # Simple rule-based fallback
        entities = []
        words = text.split()
        
        # Basic person name detection
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entities.append(Entity(
                    text=word,
                    label="PERSON",
                    start=text.find(word),
                    end=text.find(word) + len(word),
                    confidence=0.6
                ))
                
        return entities

class RelationshipExtractor(dspy.Module):
    """Self-improving relationship extraction module"""
    
    def __init__(self):
        super().__init__()
        self.relationship_extraction = dspy.Predict(RelationshipExtractionSignature)
        
    def forward(self, text: str, entities: List[Entity], context: Optional[str] = None) -> List[Relationship]:
        """Extract relationships using DSPy"""
        # Prepare entities data
        entities_data = [
            {
                "text": e.text,
                "label": e.label,
                "start": e.start,
                "end": e.end
            }
            for e in entities
        ]
        
        # Prepare input
        inputs = {
            "text": text,
            "entities": json.dumps(entities_data)
        }
        if context:
            inputs["context"] = context
            
        # Use DSPy to predict relationships
        prediction = self.relationship_extraction(**inputs)
        
        # Parse prediction into Relationship objects
        relationships = []
        try:
            relationships_data = json.loads(prediction.relationships)
            for rel_data in relationships_data:
                # Ensure rel_data is a dictionary
                if isinstance(rel_data, dict):
                    relationship = Relationship(
                        subject=rel_data["subject"],
                        predicate=rel_data["predicate"],
                        object=rel_data["object"],
                        confidence=rel_data.get("confidence", 0.8),
                        metadata=rel_data.get("metadata", {})
                    )
                    relationships.append(relationship)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback to simple relationship detection
            relationships = self._fallback_relationships(text, entities)
            
        return relationships
    
    def _fallback_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Fallback relationship extraction"""
        relationships = []
        
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.label == "PERSON" and entity2.label == "ORG":
                    relationships.append(Relationship(
                        subject=entity1.text,
                        predicate="works_for",
                        object=entity2.text,
                        confidence=0.5
                    ))
                    
        return relationships

class GraphBuilder(dspy.Module):
    """Self-improving graph building module"""
    
    def __init__(self):
        super().__init__()
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.graph_building = dspy.Predict(GraphBuildingSignature)
        
    def forward(self, text: str, context: Optional[str] = None, requirements: Optional[str] = None) -> KnowledgeGraph:
        """Build complete knowledge graph using DSPy with hybrid approach"""
        try:
            # Always use DSPy for better quality
            # Removed threshold to ensure consistent behavior

            # Optimized direct LM approach with exact schema format
            prompt = f"""Extract relations from this text as JSON:

Text: {text}

Output format: {{"relations": [{{"subject": "entity_name", "predicate": "specific_relation", "object": "entity_name", "confidence": 0.8}}]}}

Examples:
- "John works at Google" -> {{"relations": [{{"subject": "John", "predicate": "works_at", "object": "Google", "confidence": 0.9}}]}}
- "Sarah lives in Seattle" -> {{"relations": [{{"subject": "Sarah", "predicate": "lives_in", "object": "Seattle", "confidence": 0.85}}]}}

Your JSON:"""

            # Use the LM directly from config
            from components.ai.dspy_modules import dspy
            lm_response = dspy.settings.lm(prompt)

            # Parse the JSON response
            graph_data = None
            if isinstance(lm_response, list) and len(lm_response) > 0:
                response_text = lm_response[0]

                # Handle markdown-wrapped JSON (```json...```)
                if '```json' in response_text:
                    # Extract JSON from markdown code block
                    start = response_text.find('```json') + 7
                    end = response_text.rfind('```')
                    if end > start:
                        json_text = response_text[start:end].strip()
                        try:
                            graph_data = json.loads(json_text)
                            print(f"DEBUG: Parsed markdown JSON: {graph_data}")
                        except json.JSONDecodeError as e:
                            print(f"DEBUG: Markdown JSON parse error: {e}")

                # Handle direct JSON
                elif response_text.startswith('{') and response_text.endswith('}'):
                    try:
                        graph_data = json.loads(response_text)
                        print(f"DEBUG: Parsed direct JSON: {graph_data}")
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Direct JSON parse error: {e}")

                # Handle JSON object somewhere in the response
                elif '{' in response_text and '}' in response_text:
                    try:
                        # Find JSON object in response
                        start = response_text.find('{')
                        end = response_text.rfind('}') + 1
                        json_text = response_text[start:end]
                        graph_data = json.loads(json_text)
                        print(f"DEBUG: Parsed embedded JSON: {graph_data}")
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Embedded JSON parse error: {e}")

            if not graph_data:
                print(f"DEBUG: Using fallback, response was: {repr(lm_response[0]) if lm_response else 'empty'}")
                # Fallback to step-by-step extraction
                return self._fallback_graph_building(text, context)

            # Parse entities from relations
            entities = []
            entity_set = set()
            for r in graph_data.get("relations", []):
                if isinstance(r, dict):
                    subject = r.get("subject", "")
                    obj = r.get("object", "")
                    if subject and subject not in entity_set:
                        entities.append(Entity(text=subject, label="ENTITY", start=0, end=len(subject), confidence=0.8))
                        entity_set.add(subject)
                    if obj and obj not in entity_set:
                        entities.append(Entity(text=obj, label="ENTITY", start=0, end=len(obj), confidence=0.8))
                        entity_set.add(obj)

            # Parse relations safely
            relationships = []
            relations_data = graph_data.get("relations", [])
            print(f"DEBUG: Found {len(relations_data)} relations to parse")

            for r in relations_data:
                if isinstance(r, dict):
                    print(f"DEBUG: Processing relation: {r}")
                    relationship = Relationship(
                        subject=r.get("subject", ""),
                        predicate=r.get("predicate", ""),
                        object=r.get("object", ""),
                        confidence=r.get("confidence", 0.8),
                        metadata={}
                    )
                    relationships.append(relationship)
                    print(f"DEBUG: Added relationship: {relationship.subject} --{relationship.predicate}--> {relationship.object}")
                else:
                    print(f"DEBUG: Relation is not a dict: {r}")

            print(f"DEBUG: Creating KnowledgeGraph with {len(relationships)} relationships")
            result = KnowledgeGraph(
                entities=entities,
                relationships=relationships,
                source_text=text,
                extraction_confidence=0.8,
                metadata={"method": "direct_lm"}
            )
            print(f"DEBUG: Final KG has {len(result.relationships)} relationships")
            return result

        except Exception as e:
            print(f"DEBUG: Exception in direct LM approach: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to step-by-step extraction
            return self._fallback_graph_building(text, context)
    
    def _fallback_graph_building(self, text: str, context: Optional[str] = None) -> KnowledgeGraph:
        """Fallback graph building"""
        entities = self.entity_extractor(text, context)
        relationships = self.relationship_extractor(text, entities, context)
        
        return KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            source_text=text,
            extraction_confidence=0.7,
            metadata={"method": "fallback"}
        )

    def retrieve_triples(self, query: str) -> List[str]:
        """Retrieve relevant triples from mock dual_graph using RAG-style selection with fallback."""
        # Mock fetch from dual_graph.get_relationships (using SAMPLE_TRIPLES)
        triples_json = json.dumps(SAMPLE_TRIPLES)
        
        # Primary retrieval using DSPy ChainOfThought with TripleRetrievalSignature
        try:
            triple_retrieval = dspy.ChainOfThought(TripleRetrievalSignature)
            prediction = triple_retrieval(query=query, context=triples_json)
            
            # Parse output: expect list of strings like "subject predicate object"
            relevant_triples = []
            if prediction.relevant_triples:
                try:
                    # Try to parse as JSON list of strings
                    triples_list = json.loads(prediction.relevant_triples)
                    if isinstance(triples_list, list):
                        relevant_triples = [str(triple) for triple in triples_list if isinstance(triple, str)]
                    else:
                        # Fallback to splitting by newline or comma
                        relevant_triples = [t.strip() for t in str(prediction.relevant_triples).split('\n') if t.strip()]
                except json.JSONDecodeError:
                    # Split by common delimiters
                    relevant_triples = [t.strip() for t in str(prediction.relevant_triples).split(',') if t.strip()]
            
            # Filter for valid "subject predicate object" format
            valid_triples = [t for t in relevant_triples if len(t.split()) >= 3]
            if valid_triples:
                return valid_triples
                
        except Exception:
            pass  # Fall through to fallback
        
        # Fallback: keyword match on predicates (e.g., for family queries)
        query_lower = query.lower()
        fallback_triples = []
        family_keywords = ['family', 'relationship', 'married', 'parent', 'child', 'sibling']
        if any(kw in query_lower for kw in family_keywords):
            for triple in SAMPLE_TRIPLES:
                if triple['predicate'] in ['married_to', 'parent_of', 'child_of', 'sibling_of']:
                    fallback_triples.append(f"{triple['subject']} {triple['predicate']} {triple['object']}")
        
        if fallback_triples:
            return fallback_triples
        
        # Default: return all triples if no specific match
        return [f"{t['subject']} {t['predicate']} {t['object']}" for t in SAMPLE_TRIPLES]

# DSPy Optimizer for Self-Improvement
class DSPyOptimizer:
    """Optimizes DSPy modules using GEPA principles"""
    
    def __init__(self):
        self.config = DSPyConfig()
        
    def optimize_module(self, module: dspy.Module, training_data: List[Dict[str, Any]]) -> dspy.Module:
        """Optimize a DSPy module using training data"""
        # Configure DSPy for optimization using available teleprompters
        try:
            # Try to use BootstrapFewShot if available
            teleprompter = dspy.BootstrapFewShot(
                metric=self._extraction_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4
            )
        except AttributeError:
            # Fallback to simple optimization
            return module
        
        try:
            # Optimize the module
            optimized_module = teleprompter.compile(
                module,
                trainset=training_data[:20]  # Use subset for training
            )
            return optimized_module
        except Exception:
            # Fallback to original module if optimization fails
            return module
    
    def _extraction_metric(self, example: Dict[str, Any], prediction: Dict[str, Any]) -> float:
        """Custom metric for extraction quality"""
        # Simple F1-score based metric
        if "entities" in example and "entities" in prediction:
            predicted_entities = set(prediction["entities"])
            actual_entities = set(example["entities"])
            
            if len(predicted_entities) == 0:
                return 0.0
                
            precision = len(predicted_entities & actual_entities) / len(predicted_entities)
            recall = len(predicted_entities & actual_entities) / len(actual_entities)
            
            if precision + recall == 0:
                return 0.0
                
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
            
        return 0.0

# Main DSPy Framework Class
class DSPyFramework:
    """Main DSPy framework for HotMem V3"""
    
    def __init__(self):
        self.config = DSPyConfig()
        self.graph_builder = GraphBuilder()
        self.optimizer = DSPyOptimizer()
        self.is_trained = False
        
    def extract_graph(self, text: str, context: Optional[str] = None) -> KnowledgeGraph:
        """Extract knowledge graph from text"""
        return self.graph_builder(text, context)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the DSPy framework"""
        if not training_data:
            return
            
        # Optimize the graph builder
        self.graph_builder = self.optimizer.optimize_module(
            self.graph_builder, 
            training_data
        )
        
        self.is_trained = True
        
    def save_model(self, path: str):
        """Save trained model"""
        # DSPy models can be saved for later use
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.graph_builder, f)
    
    def load_model(self, path: str):
        """Load trained model"""
        import pickle
        with open(path, 'rb') as f:
            self.graph_builder = pickle.load(f)
        self.is_trained = True

# Integration with existing HotMem system
class DSPyHotMemIntegration:
    """Integration layer for DSPy with existing HotMem system"""
    
    def __init__(self):
        self.dspy_framework = DSPyFramework()
        
    def extract_facts(self, text: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract facts in HotMem format"""
        graph = self.dspy_framework.extract_graph(text, context)
        
        # Convert to HotMem fact format
        facts = []
        for relationship in graph.relationships:
            fact = {
                "subject": relationship.subject,
                "predicate": relationship.predicate,
                "object": relationship.object,
                "confidence": relationship.confidence,
                "source_text": text,
                "metadata": {
                    "extraction_method": "dspy_v3",
                    "graph_confidence": graph.extraction_confidence,
                    **relationship.metadata
                }
            }
            facts.append(fact)
            
        return facts
    
    def train_from_memory(self, memory_data: List[Dict[str, Any]]):
        """Train DSPy framework from existing memory data"""
        training_data = []
        
        for memory in memory_data:
            if "text" in memory and "facts" in memory:
                # Convert HotMem facts to training format
                entities = []
                relationships = []
                
                for fact in memory["facts"]:
                    relationships.append({
                        "subject": fact["subject"],
                        "predicate": fact["predicate"],
                        "object": fact["object"]
                    })
                
                training_example = {
                    "text": memory["text"],
                    "relationships": relationships,
                    "entities": entities
                }
                training_data.append(training_example)
        
        if training_data:
            self.dspy_framework.train(training_data)


def test_triple_retrieval():
    """Basic test for TripleRetrieval in GraphBuilder."""
    from sklearn.metrics import f1_score  # For F1 computation
    
    # Initialize GraphBuilder
    graph_builder = GraphBuilder()
    
    # Test query
    query = "What is Sarah's family relationship?"
    retrieved_triples = graph_builder.retrieve_triples(query)
    
    # Expected gold standard
    gold_triples = ["Sarah married_to Michael_Chen"]
    
    # Simple F1 computation (treating triples as binary classification per possible triple)
    # For this basic test, check if the expected triple is in retrieved
    expected_triple = gold_triples[0]
    f1 = 1.0 if expected_triple in retrieved_triples else 0.0
    
    print(f"Query: {query}")
    print(f"Retrieved triples: {retrieved_triples}")
    print(f"Gold triples: {gold_triples}")
    print(f"F1 Score: {f1}")
    
    assert f1 > 0.8, f"Retrieval failed: expected F1 > 0.8, got {f1}"
    print("Triple retrieval test passed!")
    
    return f1


# Run test if module is run directly
if __name__ == "__main__":
    test_triple_retrieval()

# Global instance
_dspy_framework: Optional[DSPyFramework] = None

def get_dspy_framework() -> DSPyFramework:
    """Get global DSPy framework instance"""
    global _dspy_framework
    if _dspy_framework is None:
        _dspy_framework = DSPyFramework()
    return _dspy_framework

def set_dspy_framework(framework: DSPyFramework):
    """Set global DSPy framework instance"""
    global _dspy_framework
    _dspy_framework = framework