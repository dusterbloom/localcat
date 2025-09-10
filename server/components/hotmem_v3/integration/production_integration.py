"""
HotMem v3 Production Integration for localcat
Integrates the trained HotMem v3 model with the existing voice agent pipeline

This script provides the integration layer between:
- HotMem v3 streaming extraction
- Existing localcat voice agent (Pipecat framework)
- Real-time knowledge graph construction
- Voice UI components
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import time
import threading
from dataclasses import dataclass, field

# Set up logging
logger = logging.getLogger(__name__)

# Import existing localcat components
try:
    from bot import LocalCatBot
    from components.ai.dspy_modules import DSPyFramework
except ImportError:
    print("Warning: Could not import localcat components - will create standalone integration")

# Import HotMem v3 components
from ..extraction.streaming_extraction import StreamingExtractor, StreamingVoiceProcessor

# Try to import inference module, fallback to basic implementation
try:
    from ..core.inference import HotMemInference
except ImportError:
    # Create basic inference class
    class HotMemInference:
        def __init__(self, model_path=None):
            self.model_path = model_path
            self.initialized = False
        
        def initialize(self):
            self.initialized = True
            
        def extract(self, text):
            return {'entities': [], 'relations': []}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VoiceEvent:
    """Represents a voice event in the pipeline"""
    event_type: str  # 'speech_start', 'speech_end', 'transcription', 'extraction'
    data: Dict[str, Any]
    timestamp: float
    session_id: str

@dataclass
class MemoryUpdate:
    """Represents a memory update to be stored"""
    entities: List[str]
    relations: List[Dict[str, Any]]
    context: str
    confidence: float
    timestamp: float
    session_id: str

class HotMemIntegration:
    """Main integration class for HotMem v3 in localcat"""
    
    def __init__(self, 
                 model_path: str = "./optimized_models/hotmem_v3_package",
                 enable_real_time: bool = True,
                 confidence_threshold: float = 0.7):
        """
        Initialize HotMem v3 integration
        
        Args:
            model_path: Path to the optimized HotMem v3 model
            enable_real_time: Whether to enable real-time streaming extraction
            confidence_threshold: Minimum confidence for accepting extractions
        """
        self.model_path = Path(model_path) if model_path else None
        self.enable_real_time = enable_real_time
        self.confidence_threshold = confidence_threshold
        
        # Initialize HotMem components
        self.inference_engine = None
        self.streaming_extractor = None
        self.voice_processor = None
        
        # Integration state
        self.session_id = f"session_{int(time.time())}"
        self.current_transcript = ""
        self.knowledge_graph = {"entities": set(), "relations": []}
        self.memory_updates = []
        
        # Event callbacks
        self.event_callbacks = {
            'extraction_complete': [],
            'memory_update': [],
            'graph_updated': []
        }
        
        # Performance tracking
        self.extraction_count = 0
        self.total_extraction_time = 0.0
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize HotMem v3 components"""
        
        try:
            # Load inference engine
            self.inference_engine = HotMemInference(str(self.model_path))
            logger.info("✅ HotMem inference engine loaded")
            
            # Initialize streaming extractor
            if self.enable_real_time:
                self.streaming_extractor = StreamingExtractor(
                    model_path=str(self.model_path),
                    confidence_threshold=self.confidence_threshold,
                    enable_async=True
                )
                
                # Initialize voice processor
                self.voice_processor = StreamingVoiceProcessor(self.streaming_extractor)
                
                # Override the result handler
                self.voice_processor._on_extraction_result = self._on_extraction_result
                
                logger.info("✅ Streaming extraction initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize HotMem components: {e}")
            raise
    
    def _on_extraction_result(self, result, chunk):
        """Handle extraction result from streaming processor"""
        
        # Create extraction event
        event = VoiceEvent(
            event_type='extraction',
            data={
                'entities': result.entities,
                'relations': result.relations,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'is_complete': result.is_complete,
                'chunk_id': chunk.chunk_id
            },
            timestamp=time.time(),
            session_id=self.session_id
        )
        
        # Update knowledge graph
        self._update_knowledge_graph(result)
        
        # Create memory update
        memory_update = MemoryUpdate(
            entities=result.entities,
            relations=result.relations,
            context=self.current_transcript,
            confidence=result.confidence,
            timestamp=time.time(),
            session_id=self.session_id
        )
        
        self.memory_updates.append(memory_update)
        
        # Trigger callbacks
        self._trigger_callbacks('extraction_complete', event)
        self._trigger_callbacks('memory_update', memory_update)
        
        # Update performance tracking
        self.extraction_count += 1
        self.total_extraction_time += result.processing_time
        
        logger.info(f"Extraction {self.extraction_count}: {len(result.entities)} entities, "
                   f"{len(result.relations)} relations (confidence: {result.confidence:.2f})")
    
    def _update_knowledge_graph(self, result):
        """Update the internal knowledge graph"""
        
        # Add new entities
        for entity in result.entities:
            self.knowledge_graph['entities'].add(entity)
        
        # Add new relations (avoid duplicates)
        existing_relations = {json.dumps(r, sort_keys=True) for r in self.knowledge_graph['relations']}
        
        for relation in result.relations:
            relation_key = json.dumps(relation, sort_keys=True)
            if relation_key not in existing_relations:
                self.knowledge_graph['relations'].append(relation)
                existing_relations.add(relation_key)
        
        # Trigger graph update callback
        self._trigger_callbacks('graph_updated', self.knowledge_graph)
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event type"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_callbacks(self, event_type: str, data):
        """Trigger callbacks for specific event type"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
    
    def process_transcription(self, text: str, is_final: bool = False):
        """
        Process transcription text through HotMem v3
        
        Args:
            text: Transcribed text
            is_final: Whether this is the final transcription
        """
        
        # Update current transcript
        if is_final:
            self.current_transcript = text
        else:
            self.current_transcript += " " + text
        
        # Process through streaming extractor if available
        if self.streaming_extractor:
            from ..extraction.streaming_extraction import StreamingChunk
            
            chunk = StreamingChunk(
                text=text,
                timestamp=time.time(),
                chunk_id=len(self.current_transcript.split()),
                is_final=is_final
            )
            
            # Process in background
            threading.Thread(
                target=self.streaming_extractor.process_chunk,
                args=(chunk,),
                daemon=True
            ).start()
        
        else:
            # Fallback to direct inference
            self._process_direct_inference(text, is_final)
    
    def _process_direct_inference(self, text: str, is_final: bool):
        """Process text through direct inference (non-streaming)"""
        
        if not self.inference_engine:
            return
        
        try:
            start_time = time.time()
            
            # Extract graph
            result = self.inference_engine.extract_graph(text)
            
            processing_time = time.time() - start_time
            
            # Create result object
            from ..extraction.streaming_extraction import ExtractionResult
            extraction_result = ExtractionResult(
                entities=result.get('entities', []),
                relations=result.get('relations', []),
                confidence=result.get('confidence', 0.5),
                processing_time=processing_time,
                is_complete=is_final
            )
            
            # Update knowledge graph
            self._update_knowledge_graph(extraction_result)
            
            # Create memory update
            memory_update = MemoryUpdate(
                entities=extraction_result.entities,
                relations=extraction_result.relations,
                context=self.current_transcript,
                confidence=extraction_result.confidence,
                timestamp=time.time(),
                session_id=self.session_id
            )
            
            self.memory_updates.append(memory_update)
            
            logger.info(f"Direct inference: {len(extraction_result.entities)} entities, "
                       f"{len(extraction_result.relations)} relations")
            
        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get current knowledge graph"""
        return {
            'entities': list(self.knowledge_graph['entities']),
            'relations': self.knowledge_graph['relations'],
            'stats': {
                'entity_count': len(self.knowledge_graph['entities']),
                'relation_count': len(self.knowledge_graph['relations']),
                'extraction_count': self.extraction_count,
                'avg_extraction_time': self.total_extraction_time / max(self.extraction_count, 1),
                'session_id': self.session_id
            }
        }
    
    def query_graph(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Query the knowledge graph"""
        
        if query_type == 'entities':
            # Get all entities
            return [{'name': entity, 'type': 'entity'} for entity in self.knowledge_graph['entities']]
        
        elif query_type == 'relations':
            # Get all relations
            return self.knowledge_graph['relations']
        
        elif query_type == 'entity_relations':
            # Get relations for specific entity
            entity = kwargs.get('entity')
            if not entity:
                return []
            
            return [
                rel for rel in self.knowledge_graph['relations']
                if rel.get('subject') == entity or rel.get('object') == entity
            ]
        
        elif query_type == 'related_entities':
            # Get entities related to specific entity
            entity = kwargs.get('entity')
            if not entity:
                return []
            
            related = set()
            for rel in self.knowledge_graph['relations']:
                if rel.get('subject') == entity:
                    related.add(rel.get('object'))
                elif rel.get('object') == entity:
                    related.add(rel.get('subject'))
            
            return [{'name': e, 'type': 'entity'} for e in related if e != entity]
        
        else:
            logger.warning(f"Unknown query type: {query_type}")
            return []
    
    def enhance_llm_context(self, base_context: str) -> str:
        """Enhance LLM context with knowledge graph information"""
        
        graph = self.get_knowledge_graph()
        
        if not graph['entities'] and not graph['relations']:
            return base_context
        
        # Add relevant knowledge to context
        enhanced_context = base_context + "\n\nRelevant Knowledge:\n"
        
        # Add recent entities
        if graph['entities']:
            recent_entities = list(graph['entities'])[-10:]  # Last 10 entities
            enhanced_context += f"Known entities: {', '.join(recent_entities)}\n"
        
        # Add recent relations
        if graph['relations']:
            recent_relations = graph['relations'][-5:]  # Last 5 relations
            enhanced_context += "Known relationships:\n"
            for rel in recent_relations:
                enhanced_context += f"- {rel.get('subject', 'Unknown')} {rel.get('predicate', 'related to')} {rel.get('object', 'Unknown')}\n"
        
        return enhanced_context
    
    def get_memory_updates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memory updates"""
        return [
            {
                'entities': update.entities,
                'relations': update.relations,
                'context': update.context,
                'confidence': update.confidence,
                'timestamp': update.timestamp,
                'session_id': update.session_id
            }
            for update in self.memory_updates[-limit:]
        ]
    
    def export_knowledge_graph(self, filepath: str):
        """Export knowledge graph to file"""
        
        graph_data = {
            'knowledge_graph': self.get_knowledge_graph(),
            'memory_updates': self.get_memory_updates(),
            'export_timestamp': time.time(),
            'session_id': self.session_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Knowledge graph exported to {filepath}")
    
    def import_knowledge_graph(self, filepath: str):
        """Import knowledge graph from file"""
        
        try:
            with open(filepath, 'r') as f:
                graph_data = json.load(f)
            
            # Load knowledge graph
            imported_graph = graph_data.get('knowledge_graph', {})
            self.knowledge_graph['entities'] = set(imported_graph.get('entities', []))
            self.knowledge_graph['relations'] = imported_graph.get('relations', [])
            
            logger.info(f"Knowledge graph imported from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import knowledge graph: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {
            'extraction_count': self.extraction_count,
            'total_extraction_time': self.total_extraction_time,
            'avg_extraction_time': self.total_extraction_time / max(self.extraction_count, 1),
            'memory_updates_count': len(self.memory_updates),
            'knowledge_graph_size': {
                'entities': len(self.knowledge_graph['entities']),
                'relations': len(self.knowledge_graph['relations'])
            },
            'session_id': self.session_id
        }
        
        # Add streaming stats if available
        if self.streaming_extractor:
            streaming_stats = self.streaming_extractor.get_performance_stats()
            stats['streaming'] = streaming_stats
        
        return stats
    
    def reset_session(self, new_session_id: Optional[str] = None):
        """Reset session state"""
        
        self.session_id = new_session_id or f"session_{int(time.time())}"
        self.current_transcript = ""
        self.memory_updates = []
        
        # Reset streaming extractor
        if self.streaming_extractor:
            self.streaming_extractor.reset()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.streaming_extractor:
            await self.streaming_extractor.cleanup()
        
        self.inference_engine = None
        self.streaming_extractor = None
        logger.info("HotMemIntegration cleanup completed")

class LocalCatHotMemIntegration:
    """Integration with existing LocalCat bot"""
    
    def __init__(self, localcat_bot=None, hotmem_integration=None):
        """
        Initialize integration with LocalCat bot
        
        Args:
            localcat_bot: Existing LocalCat bot instance
            hotmem_integration: HotMem integration instance
        """
        self.localcat_bot = localcat_bot
        self.hotmem = hotmem_integration or HotMemIntegration()
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for LocalCat integration"""
        
        # Handle extraction events
        self.hotmem.add_event_callback('extraction_complete', self._on_extraction_complete)
        
        # Handle memory updates
        self.hotmem.add_event_callback('memory_update', self._on_memory_update)
        
        # Handle graph updates
        self.hotmem.add_event_callback('graph_updated', self._on_graph_updated)
    
    def _on_extraction_complete(self, event):
        """Handle extraction complete event"""
        # This could update LocalCat's internal state
        logger.info(f"Extraction complete: {len(event.data['entities'])} entities extracted")
    
    def _on_memory_update(self, memory_update):
        """Handle memory update event"""
        # This could store memory in LocalCat's memory system
        logger.info(f"Memory update: {len(memory_update.entities)} entities stored")
    
    def _on_graph_updated(self, knowledge_graph):
        """Handle graph updated event"""
        # This could trigger UI updates or other actions
        logger.info(f"Graph updated: {len(knowledge_graph['entities'])} entities total")
    
    def process_user_input(self, text: str, is_final: bool = False):
        """Process user input through HotMem integration"""
        self.hotmem.process_transcription(text, is_final)
    
    def get_enhanced_context(self, base_context: str) -> str:
        """Get enhanced context with HotMem knowledge"""
        return self.hotmem.enhance_llm_context(base_context)
    
    def get_knowledge_summary(self) -> str:
        """Get summary of current knowledge"""
        graph = self.hotmem.get_knowledge_graph()
        
        summary = f"I know about {graph['stats']['entity_count']} entities "
        summary += f"and {graph['stats']['relation_count']} relationships. "
        
        if graph['entities']:
            recent_entities = list(graph['entities'])[-5:]
            summary += f"Recent entities include: {', '.join(recent_entities)}."
        
        return summary

# Example usage and testing
def main():
    """Test the HotMem integration"""
    
    print("🚀 HotMem v3 Production Integration Test")
    print("=" * 50)
    
    # Initialize integration (without model for demo)
    try:
        integration = HotMemIntegration(
            model_path="./optimized_models/hotmem_v3_package",
            enable_real_time=True,
            confidence_threshold=0.7
        )
        print("✅ HotMem integration initialized")
    except Exception as e:
        print(f"⚠️ Could not load model, using fallback: {e}")
        integration = HotMemIntegration(model_path=None)
    
    # Test conversation
    test_conversation = [
        ("Hi, I'm Sarah and I work at Google", False),
        ("I'm a software engineer in the AI department", True),
        ("My manager is John who leads the machine learning team", False),
        ("We're working on new language models", True)
    ]
    
    print("\nTesting conversation processing...")
    
    for i, (text, is_final) in enumerate(test_conversation):
        print(f"\nTurn {i+1}: '{text}' (final: {is_final})")
        
        # Process transcription
        integration.process_transcription(text, is_final)
        
        # Give time for processing
        time.sleep(0.5)
        
        # Show current graph
        graph = integration.get_knowledge_graph()
        print(f"Current knowledge: {graph['stats']['entity_count']} entities, "
              f"{graph['stats']['relation_count']} relations")
    
    # Show final results
    print(f"\n{'='*50}")
    print("Final Knowledge Graph:")
    final_graph = integration.get_knowledge_graph()
    
    print(f"\nEntities: {final_graph['entities']}")
    print(f"\nRelations:")
    for rel in final_graph['relations']:
        print(f"  - {rel['subject']} --{rel.get('predicate', 'unknown')}--> {rel['object']}")
    
    # Test queries
    print(f"\n{'='*50}")
    print("Testing Graph Queries:")
    
    # Query entities related to Sarah
    related = integration.query_graph('related_entities', entity='Sarah')
    print(f"Entities related to Sarah: {[e['name'] for e in related]}")
    
    # Query relations for Google
    google_relations = integration.query_graph('entity_relations', entity='Google')
    print(f"Relations for Google: {len(google_relations)} found")
    
    # Performance stats
    print(f"\n{'='*50}")
    print("Performance Stats:")
    stats = integration.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test context enhancement
    print(f"\n{'='*50}")
    print("Testing Context Enhancement:")
    base_context = "User is asking about their work situation."
    enhanced = integration.enhance_llm_context(base_context)
    print(f"Base: {base_context}")
    print(f"Enhanced: {enhanced}")

if __name__ == "__main__":
    main()