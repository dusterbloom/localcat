"""Extraction registry for managing extraction strategies"""

from typing import Dict, List, Type, Any
from dataclasses import dataclass

from components.extraction.extraction_strategies import (
    HotMemExtractionStrategy,
    EnhancedHotMemExtractionStrategy,
    UDExtractionStrategy,
    HybridExtractionStrategy,
    LightweightExtractionStrategy,
    MultilingualExtractionStrategy,
    PatternBasedExtractionStrategy
)


@dataclass
class ExtractionStrategyInfo:
    """Information about an extraction strategy"""
    name: str
    description: str
    strategy_class: Type[Any]
    enabled: bool = True
    priority: int = 0


class ExtractionRegistry:
    """Registry for managing extraction strategies"""
    
    def __init__(self):
        self._strategies: Dict[str, ExtractionStrategyInfo] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default extraction strategies"""
        strategies = [
            ExtractionStrategyInfo(
                name="hotmem",
                description="Basic hotmem extraction",
                strategy_class=HotMemExtractionStrategy,
                priority=1
            ),
            ExtractionStrategyInfo(
                name="enhanced_hotmem",
                description="Enhanced hotmem extraction with improved accuracy",
                strategy_class=EnhancedHotMemExtractionStrategy,
                priority=2
            ),
            ExtractionStrategyInfo(
                name="ud",
                description="Universal Dependencies extraction",
                strategy_class=UDExtractionStrategy,
                priority=3
            ),
            ExtractionStrategyInfo(
                name="hybrid",
                description="Hybrid extraction combining multiple methods",
                strategy_class=HybridExtractionStrategy,
                priority=4
            ),
            ExtractionStrategyInfo(
                name="lightweight",
                description="Lightweight extraction for performance",
                strategy_class=LightweightExtractionStrategy,
                priority=5
            ),
            ExtractionStrategyInfo(
                name="multilingual",
                description="Multilingual extraction support",
                strategy_class=MultilingualExtractionStrategy,
                priority=6
            ),
            ExtractionStrategyInfo(
                name="pattern",
                description="Pattern-based extraction",
                strategy_class=PatternBasedExtractionStrategy,
                priority=7
            )
        ]
        
        for strategy in strategies:
            self.register_strategy(strategy)
    
    def register_strategy(self, strategy: ExtractionStrategyInfo):
        """Register an extraction strategy"""
        self._strategies[strategy.name] = strategy
    
    def get_strategy(self, name: str) -> ExtractionStrategyInfo:
        """Get a strategy by name"""
        if name not in self._strategies:
            raise ValueError(f"Unknown extraction strategy: {name}")
        return self._strategies[name]
    
    def get_enabled_strategies(self) -> List[ExtractionStrategyInfo]:
        """Get all enabled strategies, sorted by priority"""
        enabled = [s for s in self._strategies.values() if s.enabled]
        return sorted(enabled, key=lambda x: x.priority)
    
    def get_all_strategies(self) -> List[ExtractionStrategyInfo]:
        """Get all strategies"""
        return list(self._strategies.values())
    
    def enable_strategy(self, name: str):
        """Enable a strategy"""
        if name in self._strategies:
            self._strategies[name].enabled = True
    
    def disable_strategy(self, name: str):
        """Disable a strategy"""
        if name in self._strategies:
            self._strategies[name].enabled = False
    
    def list_strategy_names(self) -> List[str]:
        """List all strategy names"""
        return list(self._strategies.keys())


# Global registry instance
_extraction_registry: ExtractionRegistry = None


def get_extraction_registry() -> ExtractionRegistry:
    """Get the global extraction registry instance"""
    global _extraction_registry
    if _extraction_registry is None:
        _extraction_registry = ExtractionRegistry()
    return _extraction_registry


def register_extraction_strategy(name: str, description: str, strategy_class: Type[Any], enabled: bool = True, priority: int = 0):
    """Register an extraction strategy with the global registry"""
    strategy = ExtractionStrategyInfo(
        name=name,
        description=description,
        strategy_class=strategy_class,
        enabled=enabled,
        priority=priority
    )
    get_extraction_registry().register_strategy(strategy)