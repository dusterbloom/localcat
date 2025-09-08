"""
Quality Processor - Validates and ensures quality of extracted information
"""

import time
import re
from typing import List, Optional, Dict, Any, Union, Tuple
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class QualityLevel(Enum):
    """Quality levels for extracted information"""
    HIGH = "high"      # Reliable, well-structured information
    MEDIUM = "medium"  # Acceptable quality with some uncertainty
    LOW = "low"        # Questionable quality, needs verification
    REJECTED = "rejected"  # Poor quality, should be discarded


@dataclass
class QualityMetrics:
    """Quality metrics for extracted information"""
    confidence: float
    completeness: float
    accuracy: float
    relevance: float
    consistency: float
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return (self.confidence * 0.3 + 
                self.completeness * 0.2 + 
                self.accuracy * 0.3 + 
                self.relevance * 0.1 + 
                self.consistency * 0.1)


@dataclass
class QualityProcessorConfig:
    """Configuration for quality processor"""
    min_confidence_threshold: float = 0.6
    min_overall_quality_threshold: float = 0.5
    enable_correction: bool = True
    enable_validation: bool = True
    max_validation_time: float = 1.0  # seconds
    enable_metrics: bool = True
    quality_rules_file: Optional[str] = None


class QualityProcessor(FrameProcessor):
    """
    Quality processor that validates and ensures quality of extracted information.
    Provides quality assessment, validation, and correction capabilities.
    """
    
    def __init__(self, config: QualityProcessorConfig):
        super().__init__()
        self.config = config
        
        # Quality rules and patterns
        self._quality_rules = self._load_quality_rules()
        
        # Performance tracking
        self._metrics = {
            'total_processed': 0,
            'passed_quality': 0,
            'failed_quality': 0,
            'corrected_items': 0,
            'quality_scores': [],
            'validation_times': []
        }
        
        logger.info(f"✅ Quality processor initialized with thresholds: "
                   f"confidence={config.min_confidence_threshold}, "
                   f"overall={config.min_overall_quality_threshold}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames"""
        await super().process_frame(frame, direction)
        
        start_time = time.time()
        
        if isinstance(frame, (TranscriptionFrame, TextFrame)):
            await self._handle_text_frame(frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
        
        # Update metrics
        if self.config.enable_metrics:
            self._update_metrics(time.time() - start_time)
    
    async def _handle_text_frame(self, frame: Union[TranscriptionFrame, TextFrame], direction: FrameDirection):
        """Handle text frames for quality validation"""
        try:
            # Check if frame has extraction metadata
            extraction_result = getattr(frame, 'metadata', {}).get('extraction_result', None)
            
            if extraction_result:
                # Validate and improve extraction quality
                validated_result = await self._validate_extraction_quality(extraction_result)
                
                if validated_result:
                    # Update frame with validated result
                    frame.metadata['extraction_result'] = validated_result
                    frame.metadata['quality_validated'] = True
                    frame.metadata['quality_level'] = validated_result.get('quality_level', 'unknown')
                    
                    await self.push_frame(frame, direction)
                else:
                    # Reject low-quality extraction
                    logger.debug("✅ Extraction rejected due to low quality")
                    await self.push_frame(frame, direction)
            else:
                # Perform basic text quality validation
                text_quality = await self._validate_text_quality(frame.text)
                
                if text_quality.overall_score >= self.config.min_overall_quality_threshold:
                    frame.metadata['text_quality'] = text_quality.overall_score
                    await self.push_frame(frame, direction)
                else:
                    logger.debug("✅ Text rejected due to low quality")
                    await self.push_frame(frame, direction)
                    
        except Exception as e:
            logger.error(f"✅ Error processing text frame: {e}")
            await self.push_frame(frame, direction)
    
    async def _validate_extraction_quality(self, extraction_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate quality of extraction result"""
        try:
            # Calculate quality metrics
            quality_metrics = self._calculate_extraction_quality(extraction_result)
            
            # Determine quality level
            quality_level = self._determine_quality_level(quality_metrics)
            
            # Apply corrections if enabled and needed
            if self.config.enable_correction and quality_level in [QualityLevel.LOW, QualityLevel.MEDIUM]:
                extraction_result = await self._apply_corrections(extraction_result, quality_metrics)
                # Recalculate quality after corrections
                quality_metrics = self._calculate_extraction_quality(extraction_result)
                quality_level = self._determine_quality_level(quality_metrics)
            
            # Check if meets minimum thresholds
            if (quality_level == QualityLevel.REJECTED or 
                quality_metrics.overall_score < self.config.min_overall_quality_threshold):
                return None
            
            # Add quality metadata
            extraction_result['quality_metrics'] = quality_metrics
            extraction_result['quality_level'] = quality_level.value
            extraction_result['quality_score'] = quality_metrics.overall_score
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"✅ Error validating extraction quality: {e}")
            return None
    
    def _calculate_extraction_quality(self, extraction_result: Dict[str, Any]) -> QualityMetrics:
        """Calculate quality metrics for extraction result"""
        # Base confidence from extraction
        confidence = extraction_result.get('confidence', 0.5)
        
        # Calculate completeness based on extracted elements
        entities = extraction_result.get('entities', [])
        facts = extraction_result.get('facts', [])
        relations = extraction_result.get('relations', [])
        
        completeness = min(1.0, (len(entities) * 0.3 + len(facts) * 0.4 + len(relations) * 0.3) / 5.0)
        
        # Calculate accuracy based on validation rules
        accuracy = self._validate_extraction_accuracy(extraction_result)
        
        # Calculate relevance based on text coherence
        relevance = self._calculate_relevance(extraction_result)
        
        # Calculate consistency between extracted elements
        consistency = self._calculate_consistency(extraction_result)
        
        return QualityMetrics(
            confidence=confidence,
            completeness=completeness,
            accuracy=accuracy,
            relevance=relevance,
            consistency=consistency
        )
    
    def _validate_extraction_accuracy(self, extraction_result: Dict[str, Any]) -> float:
        """Validate accuracy of extracted information"""
        accuracy_score = 0.5  # Base score
        
        # Validate entities
        entities = extraction_result.get('entities', [])
        if entities:
            valid_entities = sum(1 for entity in entities if self._is_valid_entity(entity))
            accuracy_score += (valid_entities / len(entities)) * 0.3
        
        # Validate facts
        facts = extraction_result.get('facts', [])
        if facts:
            valid_facts = sum(1 for fact in facts if self._is_valid_fact(fact))
            accuracy_score += (valid_facts / len(facts)) * 0.2
        
        return min(1.0, accuracy_score)
    
    def _is_valid_entity(self, entity: Dict[str, Any]) -> bool:
        """Check if entity is valid"""
        text = entity.get('text', '').strip()
        label = entity.get('label', '').strip()
        
        # Basic validation
        if not text or len(text) < 2:
            return False
        
        if not label:
            return False
        
        # Check for common patterns
        if self._matches_invalid_pattern(text):
            return False
        
        return True
    
    def _is_valid_fact(self, fact: Dict[str, Any]) -> bool:
        """Check if fact is valid"""
        text = fact.get('text', '').strip()
        
        if not text or len(text) < 10:
            return False
        
        # Check for meaningful content
        if not re.search(r'[a-zA-Z]{3,}', text):
            return False
        
        return True
    
    def _matches_invalid_pattern(self, text: str) -> bool:
        """Check if text matches invalid patterns"""
        invalid_patterns = [
            r'^\d+$',  # Just numbers
            r'^[^\w\s]*$',  # Just special characters
            r'^\s*$',  # Just whitespace
            r'.*\b(undefined|null|none|nan)\b.*',  # Undefined values
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_relevance(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate relevance of extracted information"""
        text = extraction_result.get('text', '')
        entities = extraction_result.get('entities', [])
        facts = extraction_result.get('facts', [])
        
        if not text:
            return 0.0
        
        relevance_score = 0.5
        
        # Check entity relevance
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            if entity_text in text.lower():
                relevance_score += 0.1
        
        # Check fact relevance
        for fact in facts:
            fact_text = fact.get('text', '').lower()
            if any(word in text.lower() for word in fact_text.split()[:3]):
                relevance_score += 0.1
        
        return min(1.0, relevance_score)
    
    def _calculate_consistency(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate consistency between extracted elements"""
        entities = extraction_result.get('entities', [])
        facts = extraction_result.get('facts', [])
        
        consistency_score = 0.7  # Base consistency
        
        # Check for entity conflicts
        entity_texts = [e.get('text', '').lower() for e in entities]
        if len(entity_texts) != len(set(entity_texts)):
            consistency_score -= 0.2  # Duplicate entities
        
        # Check for fact conflicts
        fact_texts = [f.get('text', '').lower() for f in facts]
        if len(fact_texts) != len(set(fact_texts)):
            consistency_score -= 0.2  # Duplicate facts
        
        return max(0.0, consistency_score)
    
    def _determine_quality_level(self, quality_metrics: QualityMetrics) -> QualityLevel:
        """Determine quality level based on metrics"""
        overall_score = quality_metrics.overall_score
        
        if overall_score >= 0.8:
            return QualityLevel.HIGH
        elif overall_score >= 0.6:
            return QualityLevel.MEDIUM
        elif overall_score >= 0.4:
            return QualityLevel.LOW
        else:
            return QualityLevel.REJECTED
    
    async def _apply_corrections(self, extraction_result: Dict[str, Any], 
                               quality_metrics: QualityMetrics) -> Dict[str, Any]:
        """Apply corrections to improve extraction quality"""
        corrected_result = extraction_result.copy()
        
        # Correct entities
        if quality_metrics.accuracy < 0.7:
            corrected_result['entities'] = self._correct_entities(corrected_result.get('entities', []))
        
        # Correct facts
        if quality_metrics.completeness < 0.6:
            corrected_result['facts'] = self._correct_facts(corrected_result.get('facts', []))
        
        # Update confidence after corrections
        corrected_result['confidence'] = min(1.0, corrected_result.get('confidence', 0.5) + 0.1)
        
        self._metrics['corrected_items'] += 1
        
        return corrected_result
    
    def _correct_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correct and filter entities"""
        corrected_entities = []
        
        for entity in entities:
            if self._is_valid_entity(entity):
                # Apply standardization
                entity['text'] = entity['text'].strip()
                entity['label'] = entity['label'].strip().title()
                corrected_entities.append(entity)
        
        return corrected_entities
    
    def _correct_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correct and filter facts"""
        corrected_facts = []
        
        for fact in facts:
            if self._is_valid_fact(fact):
                # Apply standardization
                fact['text'] = fact['text'].strip()
                if fact['text'].endswith('.'):
                    fact['text'] = fact['text'][:-1]
                corrected_facts.append(fact)
        
        return corrected_facts
    
    async def _validate_text_quality(self, text: str) -> QualityMetrics:
        """Validate quality of plain text"""
        if not text or not text.strip():
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate basic text quality metrics
        confidence = 0.7  # Default confidence for user input
        
        # Completeness based on text length and structure
        completeness = min(1.0, len(text) / 100.0)  # Normalize to 100 chars
        
        # Accuracy based on language patterns
        accuracy = 0.8 if re.search(r'[a-zA-Z]{3,}', text) else 0.3
        
        # Relevance (always relevant for direct user input)
        relevance = 1.0
        
        # Consistency (check for obvious contradictions)
        consistency = 0.9 if not self._has_contradictions(text) else 0.5
        
        return QualityMetrics(
            confidence=confidence,
            completeness=completeness,
            accuracy=accuracy,
            relevance=relevance,
            consistency=consistency
        )
    
    def _has_contradictions(self, text: str) -> bool:
        """Check for obvious contradictions in text"""
        contradiction_patterns = [
            r'\b(always|never)\b.*\b(sometimes|maybe)\b',
            r'\b(yes|no)\b.*\b(no|yes)\b',
            r'\b(good|bad)\b.*\b(bad|good)\b'
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _load_quality_rules(self) -> Dict[str, Any]:
        """Load quality validation rules"""
        default_rules = {
            'min_entity_length': 2,
            'max_entity_length': 50,
            'min_fact_length': 10,
            'max_fact_length': 200,
            'required_entity_labels': ['PERSON', 'ORG', 'GPE', 'PRODUCT'],
            'invalid_words': ['undefined', 'null', 'none', 'nan', 'error'],
            'quality_weights': {
                'confidence': 0.3,
                'completeness': 0.2,
                'accuracy': 0.3,
                'relevance': 0.1,
                'consistency': 0.1
            }
        }
        
        return default_rules
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self._metrics['total_processed'] += 1
        self._metrics['validation_times'].append(processing_time)
        
        # Keep only recent validation times
        if len(self._metrics['validation_times']) > 100:
            self._metrics['validation_times'] = self._metrics['validation_times'][-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self._metrics.copy()
        
        # Calculate average validation time
        if metrics['validation_times']:
            metrics['avg_validation_time_ms'] = (
                sum(metrics['validation_times']) / len(metrics['validation_times']) * 1000
            )
        else:
            metrics['avg_validation_time_ms'] = 0
        
        # Calculate pass rate
        if metrics['total_processed'] > 0:
            metrics['pass_rate'] = metrics['passed_quality'] / metrics['total_processed']
        else:
            metrics['pass_rate'] = 0
        
        return metrics
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("✅ Quality processor cleanup completed")