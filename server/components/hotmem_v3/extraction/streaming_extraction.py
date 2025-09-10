"""
HotMem V3 Streaming Extraction
Real-time graph extraction for voice conversations

This module implements the streaming extraction capability that allows HotMem v3
to extract entities and relations from partial sentences as they come in from
voice input, enabling real-time knowledge graph construction.
"""

import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingChunk:
    """Represents a chunk of streaming text"""
    text: str
    timestamp: float
    chunk_id: int
    is_final: bool = False
    confidence: float = 1.0

@dataclass
class ExtractionResult:
    """Result of graph extraction from streaming text"""
    entities: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    processing_time: float = 0.0
    is_complete: bool = False
    partial_results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StreamingState:
    """Maintains state during streaming extraction"""
    current_text: str = ""
    extracted_entities: Set[str] = field(default_factory=set)
    extracted_relations: List[Dict[str, Any]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    chunk_count: int = 0
    last_update: float = 0.0

class StreamingExtractor:
    """Real-time streaming graph extractor for HotMem v3"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 max_history_length: int = 100,
                 confidence_threshold: float = 0.7,
                 enable_async: bool = True):
        """
        Initialize the streaming extractor
        
        Args:
            model_path: Path to the trained model
            max_history_length: Maximum number of chunks to keep in history
            confidence_threshold: Minimum confidence for accepting extractions
            enable_async: Whether to use async processing
        """
        self.model_path = model_path
        self.max_history_length = max_history_length
        self.confidence_threshold = confidence_threshold
        self.enable_async = enable_async
        
        # Streaming state
        self.current_state = StreamingState()
        self.chunk_history = deque(maxlen=max_history_length)
        self.extraction_buffer = deque(maxlen=10)
        
        # Model components (will be loaded when needed)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Threading for async processing
        self.processing_lock = threading.Lock()
        self.extraction_queue = asyncio.Queue() if enable_async else None
        
        # Performance tracking
        self.total_chunks_processed = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        
        # Initialize if model path provided
        if model_path:
            self.load_model()
    
    def configure(self, model_path: Optional[str] = None, confidence_threshold: float = 0.7, 
                 max_history_length: int = 100, enable_async: bool = True):
        """Configure the streaming extractor"""
        if model_path is not None:
            self.model_path = model_path
        
        self.confidence_threshold = confidence_threshold
        self.max_history_length = max_history_length
        self.enable_async = enable_async
        
        # Update chunk history size
        self.chunk_history = deque(maxlen=max_history_length)
        
        # Load model if path provided and not already loaded
        if model_path and not self.model_loaded:
            self.load_model()
        
        logger.info(f"Streaming extractor configured: model_path={model_path}, "
                    f"confidence_threshold={confidence_threshold}")
    
    def load_model(self):
        """Load the trained model"""
        if self.model_loaded:
            return
        
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Try to load optimized model first
            from ..core.inference import HotMemInference
            self.inference_engine = HotMemInference(self.model_path)
            self.model_loaded = True
            logger.info("✅ Loaded optimized inference engine")
            return
            
        except ImportError:
            logger.warning("Optimized inference engine not available, using fallback")
        
        try:
            # Fallback to direct model loading
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model_loaded = True
            logger.info("✅ Loaded model directly")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_chunk(self, chunk: StreamingChunk) -> ExtractionResult:
        """
        Process a single streaming chunk
        
        Args:
            chunk: StreamingChunk containing text and metadata
            
        Returns:
            ExtractionResult with entities and relations found in this chunk
        """
        start_time = time.time()
        
        with self.processing_lock:
            # Update streaming state
            self.current_state.chunk_count += 1
            self.current_state.last_update = chunk.timestamp
            
            # Add to history
            self.chunk_history.append(chunk)
            
            # Update current text
            if chunk.is_final:
                self.current_state.current_text = chunk.text
            else:
                # Append to current text
                self.current_state.current_text += " " + chunk.text
            
            # Perform extraction
            result = self._extract_from_text(self.current_state.current_text, chunk.is_final)
            
            # Update state with new extractions
            self._update_state(result)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update performance metrics
            self.total_chunks_processed += 1
            self.total_processing_time += processing_time
            self.last_processing_time = processing_time
            
            # Add to extraction buffer
            self.extraction_buffer.append(result)
            
            logger.debug(f"Processed chunk {chunk.chunk_id} in {processing_time:.3f}s")
            
            return result
    
    def _extract_from_text(self, text: str, is_final: bool) -> ExtractionResult:
        """Extract entities and relations from text"""
        
        # Clean text
        text = text.strip()
        if not text:
            return ExtractionResult()
        
        try:
            if hasattr(self, 'inference_engine') and self.inference_engine:
                # Use optimized inference engine
                result_dict = self.inference_engine.extract_graph(text)
                
                return ExtractionResult(
                    entities=result_dict.get('entities', []),
                    relations=result_dict.get('relations', []),
                    confidence=result_dict.get('confidence', 0.5),
                    is_complete=is_final
                )
            
            elif self.model and self.tokenizer:
                # Use direct model inference
                return self._model_inference(text, is_final)
            
            else:
                # Fallback to rule-based extraction
                return self._rule_based_extraction(text, is_final)
                
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult()
    
    def _model_inference(self, text: str, is_final: bool) -> ExtractionResult:
        """Perform inference using the loaded model"""
        
        # Create prompt
        if is_final:
            prompt = f"Extract entities and relations from the text. Output in JSON format.\n\nText: {text}\n\nOutput JSON:"
        else:
            prompt = f"Extract entities and relations from this partial speech. Output in JSON format.\n\nText: {text}\n\nOutput JSON:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to device if available
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON portion
        try:
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = generated_text[json_start:json_end]
                result_dict = json.loads(json_str)
                
                return ExtractionResult(
                    entities=result_dict.get('entities', []),
                    relations=result_dict.get('relations', []),
                    confidence=result_dict.get('confidence', 0.7),
                    is_complete=is_final
                )
        except:
            pass
        
        return ExtractionResult()
    
    def _rule_based_extraction(self, text: str, is_final: bool) -> ExtractionResult:
        """Fallback rule-based extraction"""
        import re
        
        # Simple entity extraction (capitalized words)
        entities = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)))
        
        # Simple relation extraction (pattern-based)
        relations = []
        
        # Look for common relation patterns
        patterns = [
            (r'(\b[A-Z][a-z]+\b)\s+(founded|created|established)\s+(\b[A-Z][a-z]+\b)', 'founded'),
            (r'(\b[A-Z][a-z]+\b)\s+(works for|is employed by)\s+(\b[A-Z][a-z]+\b)', 'works_for'),
            (r'(\b[A-Z][a-z]+\b)\s+(is located in| headquartered in)\s+(\b[A-Z][a-z]+\b)', 'located_in'),
        ]
        
        for pattern, relation_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 3:
                    subject, predicate, obj = match
                    relations.append({
                        'subject': subject.strip(),
                        'predicate': relation_type,
                        'object': obj.strip(),
                        'confidence': 0.6
                    })
        
        confidence = 0.5 if is_final else 0.3
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            confidence=confidence,
            is_complete=is_final
        )
    
    def _update_state(self, result: ExtractionResult):
        """Update streaming state with new extraction results"""
        
        # Update entities
        for entity in result.entities:
            if entity not in self.current_state.extracted_entities:
                self.current_state.extracted_entities.add(entity)
        
        # Update relations (avoid duplicates)
        existing_relations = {json.dumps(r, sort_keys=True) for r in self.current_state.extracted_relations}
        
        for relation in result.relations:
            relation_key = json.dumps(relation, sort_keys=True)
            if relation_key not in existing_relations:
                self.current_state.extracted_relations.append(relation)
                existing_relations.add(relation_key)
        
        # Update confidence history
        self.current_state.confidence_history.append(result.confidence)
        
        # Update processing times
        self.current_state.processing_times.append(result.processing_time)
    
    def get_current_graph(self) -> Dict[str, Any]:
        """Get the current knowledge graph state"""
        
        return {
            'entities': list(self.current_state.extracted_entities),
            'relations': self.current_state.extracted_relations,
            'current_text': self.current_state.current_text,
            'stats': {
                'chunks_processed': self.current_state.chunk_count,
                'avg_confidence': sum(self.current_state.confidence_history) / len(self.current_state.confidence_history) if self.current_state.confidence_history else 0,
                'avg_processing_time': sum(self.current_state.processing_times) / len(self.current_state.processing_times) if self.current_state.processing_times else 0,
                'last_update': self.current_state.last_update
            }
        }
    
    def reset(self):
        """Reset the streaming state"""
        self.current_state = StreamingState()
        self.chunk_history.clear()
        self.extraction_buffer.clear()
        logger.info("Streaming state reset")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        if self.total_chunks_processed == 0:
            return {'status': 'no_chunks_processed'}
        
        return {
            'total_chunks_processed': self.total_chunks_processed,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / self.total_chunks_processed,
            'last_processing_time': self.last_processing_time,
            'chunks_per_second': self.total_chunks_processed / self.total_processing_time,
            'buffer_size': len(self.extraction_buffer),
            'model_loaded': self.model_loaded
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.reset()
        logger.info("StreamingExtractor cleanup completed")

class StreamingVoiceProcessor:
    """Integrates streaming extraction with voice input"""
    
    def __init__(self, extractor: StreamingExtractor):
        self.extractor = extractor
        self.voice_buffer = deque(maxlen=50)
        self.is_processing = False
        self.processing_thread = None
    
    def add_voice_chunk(self, audio_data: bytes, text: str, is_final: bool = False):
        """Add voice chunk for processing"""
        
        chunk = StreamingChunk(
            text=text,
            timestamp=time.time(),
            chunk_id=len(self.voice_buffer),
            is_final=is_final
        )
        
        self.voice_buffer.append(chunk)
        
        # Process immediately if not already processing
        if not self.is_processing:
            self._process_voice_buffer()
    
    def _process_voice_buffer(self):
        """Process accumulated voice chunks"""
        
        if not self.voice_buffer:
            return
        
        self.is_processing = True
        
        try:
            while self.voice_buffer:
                chunk = self.voice_buffer.popleft()
                result = self.extractor.process_chunk(chunk)
                
                # Log significant extractions
                if result.entities or result.relations:
                    logger.info(f"Extracted: {len(result.entities)} entities, {len(result.relations)} relations")
                
                # Here you could trigger callbacks or update UI
                self._on_extraction_result(result, chunk)
                
        finally:
            self.is_processing = False
    
    def _on_extraction_result(self, result: ExtractionResult, chunk: StreamingChunk):
        """Handle extraction result (override for custom behavior)"""
        # This method can be overridden to handle results
        # e.g., update UI, send to other systems, etc.
        pass
    
    def start_listening(self):
        """Start continuous voice processing"""
        logger.info("Starting voice processing...")
        # This would integrate with your voice input system
        pass
    
    def stop_listening(self):
        """Stop voice processing"""
        logger.info("Stopping voice processing...")
        self.is_processing = False

# Example usage and testing
def main():
    """Test the streaming extraction system"""
    
    print("🎯 HotMem v3 Streaming Extraction Test")
    print("=" * 50)
    
    # Initialize extractor (without model for demo)
    extractor = StreamingExtractor(
        model_path=None,  # Will use rule-based fallback
        confidence_threshold=0.5
    )
    
    # Simulate streaming conversation
    conversation_chunks = [
        "Hi, I'm Steve",
        "and I work at Apple",
        "which is located in Cupertino",
        "I founded the company",
        "with my friend Woz"
    ]
    
    print("Simulating streaming conversation...")
    
    for i, chunk_text in enumerate(conversation_chunks):
        print(f"\nChunk {i+1}: '{chunk_text}'")
        
        # Create streaming chunk
        chunk = StreamingChunk(
            text=chunk_text,
            timestamp=time.time(),
            chunk_id=i,
            is_final=(i == len(conversation_chunks) - 1)
        )
        
        # Process chunk
        result = extractor.process_chunk(chunk)
        
        # Show results
        if result.entities:
            print(f"  Entities: {result.entities}")
        if result.relations:
            print(f"  Relations: {result.relations}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Processing time: {result.processing_time:.3f}s")
    
    # Show final graph
    print(f"\n{'='*50}")
    print("Final Knowledge Graph:")
    final_graph = extractor.get_current_graph()
    
    print(f"Total entities: {len(final_graph['entities'])}")
    print(f"Total relations: {len(final_graph['relations'])}")
    
    print(f"\nEntities: {final_graph['entities']}")
    print(f"\nRelations:")
    for rel in final_graph['relations']:
        print(f"  - {rel['subject']} --{rel.get('predicate', 'unknown')}--> {rel['object']}")
    
    # Performance stats
    print(f"\n{'='*50}")
    print("Performance Stats:")
    stats = extractor.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()