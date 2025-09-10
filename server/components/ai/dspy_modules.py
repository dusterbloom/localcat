"""
DSPy Framework for Declarative AI Modules

Revolutionary self-improving AI system using DSPy for graph extraction.
This transforms HotMem from static pattern engineering to adaptive intelligence.

Author: HotMem V3 Evolution
"""

import dspy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

# DSPy Configuration
class DSPyConfig:
    """Configuration for DSPy framework"""
    def __init__(self):
        # Configure DSPy with local LLM
        self.llm = dspy.LM(
            model="gpt-3.5-turbo",  # Will be overridden with local model
            api_base="http://localhost:1234/v1",  # Local LLM server
            api_key="not-needed",  # Local server doesn't require key
            max_tokens=1000,
            temperature=0.1
        )
        
        # Configure DSPy
        dspy.settings.configure(
            lm=self.llm,
            trace=[],  # For learning and optimization
        )

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
        """Build complete knowledge graph using DSPy"""
        # Prepare input
        inputs = {"text": text}
        if context:
            inputs["context"] = context
        if requirements:
            inputs["requirements"] = requirements
            
        # Use DSPy to predict complete graph
        prediction = self.graph_building(**inputs)
        
        # Parse prediction into KnowledgeGraph
        try:
            graph_data = json.loads(prediction.graph)
            
            # Parse entities safely
            entities = []
            for e in graph_data.get("entities", []):
                if isinstance(e, dict):
                    entity = Entity(
                        text=e.get("text", ""),
                        label=e.get("label", ""),
                        start=e.get("start", 0),
                        end=e.get("end", 0),
                        confidence=e.get("confidence", 0.8),
                        metadata=e.get("metadata", {})
                    )
                    entities.append(entity)
            
            # Parse relationships safely
            relationships = []
            for r in graph_data.get("relationships", []):
                if isinstance(r, dict):
                    relationship = Relationship(
                        subject=r.get("subject", ""),
                        predicate=r.get("predicate", ""),
                        object=r.get("object", ""),
                        confidence=r.get("confidence", 0.8),
                        metadata=r.get("metadata", {})
                    )
                    relationships.append(relationship)
            
            return KnowledgeGraph(
                entities=entities,
                relationships=relationships,
                source_text=text,
                extraction_confidence=float(prediction.confidence),
                metadata=graph_data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
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