"""
Extraction Processor - Handles text extraction and processing
"""

import time
from typing import List, Optional, Dict, Any, Union
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from extraction_registry import ExtractionRegistry
from components.extraction.extraction_strategies import ExtractionStrategyBase


class ExtractionMode(Enum):
    """Extraction processing modes"""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    ADAPTIVE = "adaptive"


@dataclass
class ExtractionProcessorConfig:
    """Configuration for extraction processor"""
    default_strategy: str = "enhanced_hotmem"
    fallback_strategy: str = "lightweight"
    enable_multi_strategy: bool = True
    quality_threshold: float = 0.7
    max_extraction_time: float = 2.0  # seconds
    enable_metrics: bool = True


class ExtractionProcessor(FrameProcessor):
    """
    Extraction processor that handles text extraction using multiple strategies.
    Provides a unified interface for different extraction methods.
    """
    
    def __init__(self, config: ExtractionProcessorConfig):
        super().__init__()
        self.config = config
        
        # Initialize extraction registry
        self.extraction_registry = ExtractionRegistry()
        
        # Performance tracking
        self._metrics = {
            'total_processed': 0,
            'successful_extractions': 0,
            'strategy_usage': {},
            'avg_extraction_time_ms': 0,
            'quality_scores': []
        }
        
        logger.info(f"🔍 Extraction processor initialized with default strategy: {config.default_strategy}")
    
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
        """Handle text frames for extraction"""
        try:
            text = frame.text
            if not text or not text.strip():
                await self.push_frame(frame, direction)
                return
            
            # Extract information using configured strategy
            extraction_result = await self._extract_information(text)
            
            if extraction_result:
                # Create enhanced frame with extraction metadata
                enhanced_frame = self._create_enhanced_frame(frame, extraction_result)
                await self.push_frame(enhanced_frame, direction)
            else:
                # Pass through original frame
                await self.push_frame(frame, direction)
                
        except Exception as e:
            logger.error(f"🔍 Error processing text frame: {e}")
            await self.push_frame(frame, direction)
    
    async def _extract_information(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract information from text using configured strategies"""
        start_time = time.time()
        
        # Try primary strategy first
        result = await self._try_extraction_strategy(text, self.config.default_strategy)
        
        if result and result.get('quality_score', 0) >= self.config.quality_threshold:
            logger.debug(f"🔍 Primary strategy succeeded with quality: {result['quality_score']}")
            return result
        
        # Try fallback strategy if primary failed
        if self.config.fallback_strategy and self.config.fallback_strategy != self.config.default_strategy:
            logger.debug(f"🔍 Trying fallback strategy: {self.config.fallback_strategy}")
            result = await self._try_extraction_strategy(text, self.config.fallback_strategy)
            if result:
                logger.debug(f"🔍 Fallback strategy succeeded with quality: {result.get('quality_score', 0)}")
                return result
        
        # Try multi-strategy approach if enabled
        if self.config.enable_multi_strategy:
            logger.debug("🔍 Trying multi-strategy extraction")
            result = await self._try_multi_strategy_extraction(text)
            if result:
                logger.debug(f"🔍 Multi-strategy extraction succeeded with quality: {result.get('quality_score', 0)}")
                return result
        
        logger.warning(f"🔍 All extraction strategies failed for text: {text[:100]}...")
        return None
    
    async def _try_extraction_strategy(self, text: str, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Try extraction with a specific strategy"""
        try:
            strategy = self.extraction_registry.get_strategy(strategy_name)
            if not strategy:
                logger.warning(f"🔍 Strategy not found: {strategy_name}")
                return None
            
            # Perform extraction
            result = await strategy.extract(text)
            
            if result:
                # Add metadata
                result['strategy_used'] = strategy_name
                result['extraction_time'] = time.time()
                
                # Update metrics
                self._update_strategy_metrics(strategy_name, result.get('quality_score', 0))
                
                return result
            
        except Exception as e:
            logger.error(f"🔍 Error with strategy {strategy_name}: {e}")
        
        return None
    
    async def _try_multi_strategy_extraction(self, text: str) -> Optional[Dict[str, Any]]:
        """Try extraction using multiple strategies and combine results"""
        available_strategies = self.extraction_registry.get_available_strategies()
        
        if not available_strategies:
            return None
        
        # Run extractions in parallel
        extraction_tasks = []
        for strategy_name in available_strategies[:3]:  # Limit to top 3 strategies
            task = self._try_extraction_strategy(text, strategy_name)
            extraction_tasks.append(task)
        
        # Wait for all extractions to complete
        import asyncio
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, dict) and result.get('quality_score', 0) > 0:
                successful_results.append(result)
        
        if not successful_results:
            return None
        
        # Combine results
        combined_result = self._combine_extraction_results(successful_results)
        return combined_result
    
    def _combine_extraction_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple extraction results"""
        # Sort by quality score
        sorted_results = sorted(results, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Take the best result as base
        best_result = sorted_results[0]
        
        # Merge entities from all results
        all_entities = []
        for result in results:
            entities = result.get('entities', [])
            all_entities.extend(entities)
        
        # Remove duplicates
        unique_entities = self._deduplicate_entities(all_entities)
        
        # Create combined result
        combined = {
            'text': best_result.get('text', ''),
            'entities': unique_entities,
            'facts': best_result.get('facts', []),
            'relations': best_result.get('relations', []),
            'quality_score': max(r.get('quality_score', 0) for r in results),
            'strategies_used': [r.get('strategy_used', 'unknown') for r in results],
            'extraction_time': best_result.get('extraction_time', time.time()),
            'metadata': {
                'combined_from': len(results),
                'best_strategy': best_result.get('strategy_used', 'unknown')
            }
        }
        
        return combined
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            entity_key = (entity.get('text', '').lower(), entity.get('label', ''))
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _create_enhanced_frame(self, original_frame: Union[TranscriptionFrame, TextFrame], 
                             extraction_result: Dict[str, Any]) -> Union[TranscriptionFrame, TextFrame]:
        """Create enhanced frame with extraction metadata"""
        # Add extraction metadata to frame
        if hasattr(original_frame, 'metadata'):
            original_frame.metadata.update({
                'extraction_result': extraction_result,
                'extraction_quality': extraction_result.get('quality_score', 0),
                'extraction_strategy': extraction_result.get('strategy_used', 'unknown')
            })
        else:
            original_frame.metadata = {
                'extraction_result': extraction_result,
                'extraction_quality': extraction_result.get('quality_score', 0),
                'extraction_strategy': extraction_result.get('strategy_used', 'unknown')
            }
        
        return original_frame
    
    def _update_strategy_metrics(self, strategy_name: str, quality_score: float):
        """Update strategy usage metrics"""
        if strategy_name not in self._metrics['strategy_usage']:
            self._metrics['strategy_usage'][strategy_name] = {
                'count': 0,
                'total_quality': 0,
                'avg_quality': 0
            }
        
        usage = self._metrics['strategy_usage'][strategy_name]
        usage['count'] += 1
        usage['total_quality'] += quality_score
        usage['avg_quality'] = usage['total_quality'] / usage['count']
        
        # Track quality scores
        self._metrics['quality_scores'].append(quality_score)
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self._metrics['total_processed'] += 1
        
        if processing_time > 0:
            self._metrics['avg_extraction_time_ms'] = (
                (self._metrics['avg_extraction_time_ms'] * (self._metrics['total_processed'] - 1) + 
                 processing_time * 1000) / self._metrics['total_processed']
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self._metrics.copy()
        
        # Calculate overall average quality
        if metrics['quality_scores']:
            metrics['avg_quality_score'] = sum(metrics['quality_scores']) / len(metrics['quality_scores'])
        else:
            metrics['avg_quality_score'] = 0
        
        return metrics
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available extraction strategies"""
        return self.extraction_registry.get_available_strategies()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.extraction_registry.cleanup()
        logger.info("🔍 Extraction processor cleanup completed")