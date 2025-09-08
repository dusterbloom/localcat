"""
Pipeline Builder - Composable pipeline construction for LocalCat
"""

import asyncio
from typing import List, Dict, Any, Optional, Type, Union, Callable
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame

from processors.memory_processor import MemoryProcessor, MemoryProcessorConfig
from processors.extraction_processor import ExtractionProcessor, ExtractionProcessorConfig
from processors.quality_processor import QualityProcessor, QualityProcessorConfig
from processors.context_processor import ContextProcessor, ContextProcessorConfig


class PipelineStage(Enum):
    """Pipeline stages"""
    INPUT = "input"
    EXTRACTION = "extraction"
    QUALITY = "quality"
    MEMORY = "memory"
    CONTEXT = "context"
    OUTPUT = "output"


@dataclass
class PipelineNode:
    """Node in the pipeline graph"""
    name: str
    processor: FrameProcessor
    stage: PipelineStage
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0


@dataclass
class PipelineConfig:
    """Configuration for pipeline builder"""
    enable_memory: bool = True
    enable_extraction: bool = True
    enable_quality: bool = True
    enable_context: bool = True
    default_user_id: str = "default-user"
    max_pipeline_latency: float = 2.0  # seconds
    enable_metrics: bool = True
    parallel_processing: bool = True


class PipelineBuilder:
    """
    Pipeline builder for creating composable processing pipelines.
    Provides configuration-driven pipeline assembly and dependency injection.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nodes: Dict[str, PipelineNode] = {}
        self.pipeline_order: List[str] = []
        self.built_pipeline: Optional[List[FrameProcessor]] = None
        
        # Initialize default processors
        self._initialize_default_processors()
        
        logger.info("🏗️ Pipeline builder initialized")
    
    def _initialize_default_processors(self):
        """Initialize default pipeline processors"""
        # Memory processor
        if self.config.enable_memory:
            memory_config = MemoryProcessorConfig(
                user_id=self.config.default_user_id,
                enable_metrics=self.config.enable_metrics
            )
            self.add_processor(
                name="memory",
                processor=MemoryProcessor(memory_config),
                stage=PipelineStage.MEMORY,
                config={"user_id": self.config.default_user_id}
            )
        
        # Extraction processor
        if self.config.enable_extraction:
            extraction_config = ExtractionProcessorConfig(
                enable_metrics=self.config.enable_metrics
            )
            self.add_processor(
                name="extraction",
                processor=ExtractionProcessor(extraction_config),
                stage=PipelineStage.EXTRACTION,
                config={}
            )
        
        # Quality processor
        if self.config.enable_quality:
            quality_config = QualityProcessorConfig(
                enable_metrics=self.config.enable_metrics
            )
            self.add_processor(
                name="quality",
                processor=QualityProcessor(quality_config),
                stage=PipelineStage.QUALITY,
                config={}
            )
        
        # Context processor
        if self.config.enable_context:
            context_config = ContextProcessorConfig(
                enable_metrics=self.config.enable_metrics
            )
            self.add_processor(
                name="context",
                processor=ContextProcessor(context_config),
                stage=PipelineStage.CONTEXT,
                config={}
            )
    
    def add_processor(self, name: str, processor: FrameProcessor, stage: PipelineStage, 
                     config: Dict[str, Any] = None, dependencies: List[str] = None,
                     enabled: bool = True, priority: int = 0):
        """Add a processor to the pipeline"""
        if config is None:
            config = {}
        if dependencies is None:
            dependencies = []
        
        node = PipelineNode(
            name=name,
            processor=processor,
            stage=stage,
            config=config,
            dependencies=dependencies,
            enabled=enabled,
            priority=priority
        )
        
        self.nodes[name] = node
        logger.debug(f"🏗️ Added processor: {name} at stage {stage.value}")
        
        return self
    
    def remove_processor(self, name: str):
        """Remove a processor from the pipeline"""
        if name in self.nodes:
            del self.nodes[name]
            logger.debug(f"🏗️ Removed processor: {name}")
        
        return self
    
    def enable_processor(self, name: str):
        """Enable a processor"""
        if name in self.nodes:
            self.nodes[name].enabled = True
            logger.debug(f"🏗️ Enabled processor: {name}")
        
        return self
    
    def disable_processor(self, name: str):
        """Disable a processor"""
        if name in self.nodes:
            self.nodes[name].enabled = False
            logger.debug(f"🏗️ Disabled processor: {name}")
        
        return self
    
    def set_processor_config(self, name: str, config: Dict[str, Any]):
        """Update processor configuration"""
        if name in self.nodes:
            self.nodes[name].config.update(config)
            logger.debug(f"🏗️ Updated config for processor: {name}")
        
        return self
    
    def add_dependency(self, processor_name: str, depends_on: str):
        """Add dependency between processors"""
        if processor_name in self.nodes and depends_on in self.nodes:
            if depends_on not in self.nodes[processor_name].dependencies:
                self.nodes[processor_name].dependencies.append(depends_on)
                logger.debug(f"🏗️ Added dependency: {processor_name} -> {depends_on}")
        
        return self
    
    def build_pipeline(self) -> List[FrameProcessor]:
        """Build the pipeline with proper ordering"""
        try:
            # Validate pipeline configuration
            self._validate_pipeline()
            
            # Resolve dependencies and determine order
            self.pipeline_order = self._resolve_dependencies()
            
            # Create processor list
            pipeline = []
            for name in self.pipeline_order:
                node = self.nodes[name]
                if node.enabled:
                    pipeline.append(node.processor)
                    logger.debug(f"🏗️ Added to pipeline: {name}")
            
            self.built_pipeline = pipeline
            
            logger.info(f"🏗️ Pipeline built successfully with {len(pipeline)} processors")
            return pipeline
            
        except Exception as e:
            logger.error(f"🏗️ Error building pipeline: {e}")
            raise
    
    def _validate_pipeline(self):
        """Validate pipeline configuration"""
        # Check for circular dependencies
        if self._has_circular_dependencies():
            raise ValueError("Pipeline has circular dependencies")
        
        # Check for missing dependencies
        missing_deps = self._get_missing_dependencies()
        if missing_deps:
            raise ValueError(f"Missing dependencies: {missing_deps}")
        
        logger.debug("🏗️ Pipeline validation passed")
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            
            for dep in self.nodes[node_name].dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node_name)
            return False
        
        for node_name in self.nodes:
            if node_name not in visited:
                if has_cycle(node_name):
                    return True
        
        return False
    
    def _get_missing_dependencies(self) -> List[str]:
        """Get list of missing dependencies"""
        missing = []
        
        for node_name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    missing.append(dep)
        
        return missing
    
    def _resolve_dependencies(self) -> List[str]:
        """Resolve dependencies using topological sort"""
        # Group by stage
        stage_order = [
            PipelineStage.INPUT,
            PipelineStage.EXTRACTION,
            PipelineStage.QUALITY,
            PipelineStage.MEMORY,
            PipelineStage.CONTEXT,
            PipelineStage.OUTPUT
        ]
        
        # Get enabled nodes
        enabled_nodes = [node for node in self.nodes.values() if node.enabled]
        
        # Sort by stage first, then by priority, then by dependencies
        sorted_nodes = []
        
        for stage in stage_order:
            stage_nodes = [node for node in enabled_nodes if node.stage == stage]
            
            # Topological sort within stage
            stage_sorted = self._topological_sort(stage_nodes)
            
            # Add to final list
            sorted_nodes.extend(stage_sorted)
        
        return [node.name for node in sorted_nodes]
    
    def _topological_sort(self, nodes: List[PipelineNode]) -> List[PipelineNode]:
        """Topological sort of nodes within a stage"""
        # Create adjacency list
        adj = {node.name: [] for node in nodes}
        in_degree = {node.name: 0 for node in nodes}
        
        for node in nodes:
            for dep in node.dependencies:
                if dep in adj:  # Only consider dependencies within this stage
                    adj[dep].append(node.name)
                    in_degree[node.name] += 1
        
        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda name: self.nodes[name].priority, reverse=True)
            current = queue.pop(0)
            result.append(self.nodes[current])
            
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration"""
        return {
            'total_processors': len(self.nodes),
            'enabled_processors': len([n for n in self.nodes.values() if n.enabled]),
            'processors': {
                name: {
                    'stage': node.stage.value,
                    'enabled': node.enabled,
                    'dependencies': node.dependencies,
                    'priority': node.priority,
                    'config': node.config
                }
                for name, node in self.nodes.items()
            },
            'pipeline_order': self.pipeline_order if self.pipeline_order else [],
            'built': self.built_pipeline is not None
        }
    
    def get_processor(self, name: str) -> Optional[FrameProcessor]:
        """Get a specific processor by name"""
        if name in self.nodes:
            return self.nodes[name].processor
        return None
    
    async def process_through_pipeline(self, frame: Frame) -> Frame:
        """Process a frame through the entire pipeline"""
        if not self.built_pipeline:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        current_frame = frame
        
        for processor in self.built_pipeline:
            try:
                # Create a simple wrapper to handle the async processing
                processed_frames = []
                
                class FrameCollector:
                    async def push_frame(self, frame, direction=None):
                        processed_frames.append(frame)
                
                collector = FrameCollector()
                processor.push_frame = collector.push_frame
                
                # Process the frame
                await processor.process_frame(current_frame, None)
                
                # Get the result
                if processed_frames:
                    current_frame = processed_frames[-1]
                
            except Exception as e:
                logger.error(f"🏗️ Error processing frame in pipeline: {e}")
                # Continue with original frame
                break
        
        return current_frame
    
    async def cleanup(self):
        """Cleanup all processors"""
        for node in self.nodes.values():
            if hasattr(node.processor, 'cleanup'):
                try:
                    await node.processor.cleanup()
                except Exception as e:
                    logger.error(f"🏗️ Error cleaning up processor {node.name}: {e}")
        
        logger.info("🏗️ Pipeline cleanup completed")
    
    def create_default_pipeline(self) -> 'PipelineBuilder':
        """Create a default pipeline configuration"""
        # Set up default dependencies
        if self.config.enable_extraction and self.config.enable_quality:
            self.add_dependency("quality", "extraction")
        
        if self.config.enable_quality and self.config.enable_memory:
            self.add_dependency("memory", "quality")
        
        if self.config.enable_memory and self.config.enable_context:
            self.add_dependency("context", "memory")
        
        return self


def create_pipeline_builder(config: Optional[PipelineConfig] = None) -> PipelineBuilder:
    """Create a pipeline builder with default configuration"""
    if config is None:
        config = PipelineConfig()
    
    return PipelineBuilder(config)


def create_minimal_pipeline() -> PipelineBuilder:
    """Create a minimal pipeline with only essential processors"""
    config = PipelineConfig(
        enable_memory=True,
        enable_extraction=False,
        enable_quality=False,
        enable_context=True
    )
    
    return PipelineBuilder(config)


def create_full_pipeline() -> PipelineBuilder:
    """Create a full pipeline with all processors"""
    config = PipelineConfig(
        enable_memory=True,
        enable_extraction=True,
        enable_quality=True,
        enable_context=True
    )
    
    return PipelineBuilder(config)