"""
Extraction Registry Module

Plugin-based extraction registry for dynamic strategy management.
Provides runtime registration, configuration, and A/B testing framework.

Author: SOLID Refactoring
"""

import os
import importlib
import inspect
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass
import threading
from pathlib import Path
from loguru import logger

from memory_interfaces import IExtractionStrategy
from extraction_strategies import ExtractionStrategyBase

@dataclass
class PluginMetadata:
    """Metadata for extraction plugins."""
    name: str
    version: str
    description: str
    author: str
    strategy_class: Type[IExtractionStrategy]
    config_schema: Dict[str, Any] = None
    dependencies: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.config_schema is None:
            self.config_schema = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

@dataclass
class RegistryEntry:
    """Registry entry for an extraction strategy."""
    plugin_name: str
    strategy_class: Type[IExtractionStrategy]
    metadata: PluginMetadata
    factory_func: Optional[Callable] = None
    instance_cache: Optional[Dict[str, IExtractionStrategy]] = None
    
    def __post_init__(self):
        if self.instance_cache is None:
            self.instance_cache = {}

class ExtractionRegistry:
    """
    Plugin-based extraction registry.
    
    Responsibilities:
    - Dynamic strategy registration
    - Plugin discovery and loading
    - Strategy lifecycle management
    - A/B testing framework
    - Configuration management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Registry storage
        self.strategies: Dict[str, RegistryEntry] = {}
        self.plugins: Dict[str, PluginMetadata] = {}
        
        # Plugin directories
        self.plugin_directories = self.config.get('plugin_directories', [
            './plugins',
            './extractors',
            './server/plugins'
        ])
        
        # A/B testing configuration
        self.ab_test_config = self.config.get('ab_test_config', {})
        self.active_ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Strategy groups
        self.strategy_groups: Dict[str, List[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Discovery and loading
        self.auto_discover = self.config.get('auto_discover', True)
        self.discovered_plugins: List[str] = []
        
        # Initialize registry
        self._initialize_registry()
        
        logger.info("ExtractionRegistry initialized")
    
    def _initialize_registry(self) -> None:
        """Initialize the registry with built-in strategies."""
        # Register built-in strategies
        from extraction_strategies import (
            HotMemExtractionStrategy,
            UDExtractionStrategy,
            HybridExtractionStrategy,
            LightweightExtractionStrategy,
            MultilingualExtractionStrategy,
            EnhancedHotMemExtractionStrategy,
            PatternBasedExtractionStrategy
        )
        
        built_in_strategies = [
            ('hotmem', HotMemExtractionStrategy, 'HotMem-based extraction'),
            ('ud', UDExtractionStrategy, 'Universal Dependencies extraction'),
            ('hybrid', HybridExtractionStrategy, 'Hybrid spaCy-LLM extraction'),
            ('lightweight', LightweightExtractionStrategy, 'Lightweight relation extraction'),
            ('multilingual', MultilingualExtractionStrategy, 'Multilingual graph extraction'),
            ('enhanced_hotmem', EnhancedHotMemExtractionStrategy, 'Enhanced HotMem extraction'),
            ('pattern', PatternBasedExtractionStrategy, 'Pattern-based extraction'),
        ]
        
        for name, strategy_class, description in built_in_strategies:
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                description=description,
                author="LocalCat Team",
                strategy_class=strategy_class,
                tags=['built-in']
            )
            
            self._register_strategy(name, strategy_class, metadata)
        
        # Auto-discover plugins if enabled
        if self.auto_discover:
            self.discover_plugins()
    
    def _register_strategy(self, name: str, strategy_class: Type[IExtractionStrategy], 
                          metadata: PluginMetadata) -> None:
        """Register a strategy in the registry."""
        entry = RegistryEntry(
            plugin_name=metadata.name,
            strategy_class=strategy_class,
            metadata=metadata
        )
        
        self.strategies[name] = entry
        self.plugins[metadata.name] = metadata
        
        logger.info(f"Registered strategy: {name} ({metadata.name} v{metadata.version})")
    
    def register_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a plugin from a file path.
        
        Args:
            plugin_path: Path to plugin file
            config: Plugin configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load plugin module
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if not spec or not spec.loader:
                logger.error(f"Could not load plugin spec: {plugin_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin metadata
            if not hasattr(module, 'PLUGIN_METADATA'):
                logger.error(f"Plugin missing PLUGIN_METADATA: {plugin_path}")
                return False
            
            metadata = PluginMetadata(**module.PLUGIN_METADATA)
            
            # Validate plugin
            if not self._validate_plugin(metadata, module):
                return False
            
            # Register plugin
            self._register_strategy(metadata.name.lower(), metadata.strategy_class, metadata)
            
            # Apply configuration
            if config:
                self.update_plugin_config(metadata.name, config)
            
            logger.info(f"Registered plugin: {metadata.name} from {plugin_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_path}: {e}")
            return False
    
    def _validate_plugin(self, metadata: PluginMetadata, module) -> bool:
        """Validate a plugin."""
        try:
            # Check if strategy class exists
            if not hasattr(module, metadata.strategy_class.__name__):
                logger.error(f"Strategy class not found: {metadata.strategy_class.__name__}")
                return False
            
            # Check if strategy class implements IExtractionStrategy
            if not issubclass(metadata.strategy_class, IExtractionStrategy):
                logger.error(f"Strategy class does not implement IExtractionStrategy")
                return False
            
            # Check if strategy class can be instantiated
            try:
                strategy = metadata.strategy_class()
                if not hasattr(strategy, 'extract'):
                    logger.error("Strategy missing required extract method")
                    return False
            except Exception as e:
                logger.error(f"Cannot instantiate strategy: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            return False
    
    def discover_plugins(self) -> List[str]:
        """Discover and register plugins from plugin directories."""
        discovered = []
        
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Look for Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                plugin_path = str(py_file)
                if self.register_plugin(plugin_path):
                    discovered.append(plugin_path)
        
        self.discovered_plugins.extend(discovered)
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def get_strategy(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[IExtractionStrategy]:
        """
        Get a strategy instance by name.
        
        Args:
            name: Strategy name
            config: Strategy configuration
            
        Returns:
            Strategy instance or None
        """
        with self._lock:
            entry = self.strategies.get(name)
            if not entry:
                logger.warning(f"Strategy not found: {name}")
                return None
            
            # Create cache key for config
            cache_key = str(sorted((config or {}).items()))
            
            # Return cached instance if available
            if cache_key in entry.instance_cache:
                return entry.instance_cache[cache_key]
            
            # Create new instance
            try:
                if entry.factory_func:
                    strategy = entry.factory_func(config or {})
                else:
                    strategy = entry.strategy_class(config or {})
                
                # Cache instance
                entry.instance_cache[cache_key] = strategy
                
                logger.debug(f"Created strategy instance: {name}")
                return strategy
                
            except Exception as e:
                logger.error(f"Failed to create strategy {name}: {e}")
                return None
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all registered strategies."""
        with self._lock:
            strategies = []
            
            for name, entry in self.strategies.items():
                strategy_info = {
                    'name': name,
                    'plugin': entry.plugin_name,
                    'version': entry.metadata.version,
                    'description': entry.metadata.description,
                    'author': entry.metadata.author,
                    'tags': entry.metadata.tags,
                    'dependencies': entry.metadata.dependencies,
                    'available': self.is_strategy_available(name),
                    'cached_instances': len(entry.instance_cache)
                }
                strategies.append(strategy_info)
            
            return strategies
    
    def is_strategy_available(self, name: str) -> bool:
        """Check if a strategy is available."""
        with self._lock:
            entry = self.strategies.get(name)
            if not entry:
                return False
            
            try:
                strategy = self.get_strategy(name)
                return strategy is not None and strategy.is_available()
            except Exception:
                return False
    
    def get_strategy_config_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration schema for a strategy."""
        with self._lock:
            entry = self.strategies.get(name)
            if not entry:
                return None
            
            return entry.metadata.config_schema
    
    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        with self._lock:
            # Clear instance cache for affected strategies
            for name, entry in self.strategies.items():
                if entry.plugin_name == plugin_name:
                    entry.instance_cache.clear()
            
            logger.info(f"Updated configuration for plugin: {plugin_name}")
            return True
    
    def create_strategy_group(self, group_name: str, strategies: List[str]) -> bool:
        """Create a group of strategies."""
        with self._lock:
            # Validate all strategies exist
            for strategy in strategies:
                if strategy not in self.strategies:
                    logger.warning(f"Strategy not found: {strategy}")
                    return False
            
            self.strategy_groups[group_name] = strategies
            logger.info(f"Created strategy group: {group_name} with {len(strategies)} strategies")
            return True
    
    def get_strategy_group(self, group_name: str) -> List[str]:
        """Get strategies in a group."""
        return self.strategy_groups.get(group_name, [])
    
    def start_ab_test(self, test_name: str, group_a: List[str], group_b: List[str], 
                     traffic_split: float = 0.5) -> bool:
        """
        Start an A/B test.
        
        Args:
            test_name: Test name
            group_a: Control group strategies
            group_b: Test group strategies
            traffic_split: Traffic split for group B (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Validate strategies exist
            all_strategies = group_a + group_b
            for strategy in all_strategies:
                if strategy not in self.strategies:
                    logger.warning(f"Strategy not found: {strategy}")
                    return False
            
            self.active_ab_tests[test_name] = {
                'group_a': group_a,
                'group_b': group_b,
                'traffic_split': traffic_split,
                'started_at': time.time(),
                'impressions': 0,
                'conversions': {'a': 0, 'b': 0}
            }
            
            logger.info(f"Started A/B test: {test_name}")
            return True
    
    def get_ab_test_group(self, test_name: str, session_id: str) -> Optional[str]:
        """Get A/B test group for a session."""
        with self._lock:
            test = self.active_ab_tests.get(test_name)
            if not test:
                return None
            
            # Simple hash-based assignment
            hash_value = hash(session_id + test_name) % 100
            test['impressions'] += 1
            
            if hash_value < test['traffic_split'] * 100:
                test['conversions']['b'] += 1
                return 'b'
            else:
                test['conversions']['a'] += 1
                return 'a'
    
    def get_ab_test_stats(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test statistics."""
        with self._lock:
            test = self.active_ab_tests.get(test_name)
            if not test:
                return None
            
            return {
                'test_name': test_name,
                'group_a': test['group_a'],
                'group_b': test['group_b'],
                'traffic_split': test['traffic_split'],
                'started_at': test['started_at'],
                'duration_seconds': time.time() - test['started_at'],
                'impressions': test['impressions'],
                'conversions': test['conversions'],
                'conversion_rates': {
                    'a': test['conversions']['a'] / max(test['impressions'], 1),
                    'b': test['conversions']['b'] / max(test['impressions'], 1)
                }
            }
    
    def stop_ab_test(self, test_name: str) -> bool:
        """Stop an A/B test."""
        with self._lock:
            if test_name in self.active_ab_tests:
                del self.active_ab_tests[test_name]
                logger.info(f"Stopped A/B test: {test_name}")
                return True
            return False
    
    def reload_strategy(self, name: str) -> bool:
        """Reload a strategy."""
        with self._lock:
            entry = self.strategies.get(name)
            if not entry:
                return False
            
            # Clear instance cache
            entry.instance_cache.clear()
            
            logger.info(f"Reloaded strategy: {name}")
            return True
    
    def unregister_strategy(self, name: str) -> bool:
        """Unregister a strategy."""
        with self._lock:
            if name not in self.strategies:
                return False
            
            entry = self.strategies[name]
            
            # Remove from groups
            for group_name, strategies in self.strategy_groups.items():
                if name in strategies:
                    strategies.remove(name)
            
            # Remove from registry
            del self.strategies[name]
            if entry.plugin_name in self.plugins:
                del self.plugins[entry.plugin_name]
            
            logger.info(f"Unregistered strategy: {name}")
            return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            total_instances = sum(len(entry.instance_cache) for entry in self.strategies.values())
            
            return {
                'total_strategies': len(self.strategies),
                'total_plugins': len(self.plugins),
                'total_instances': total_instances,
                'strategy_groups': len(self.strategy_groups),
                'active_ab_tests': len(self.active_ab_tests),
                'discovered_plugins': len(self.discovered_plugins),
                'plugin_directories': self.plugin_directories
            }
    
    def export_registry_config(self) -> Dict[str, Any]:
        """Export registry configuration."""
        with self._lock:
            config = {
                'strategies': {},
                'groups': self.strategy_groups,
                'ab_tests': self.active_ab_tests
            }
            
            for name, entry in self.strategies.items():
                config['strategies'][name] = {
                    'plugin': entry.plugin_name,
                    'metadata': {
                        'version': entry.metadata.version,
                        'description': entry.metadata.description,
                        'author': entry.metadata.author,
                        'tags': entry.metadata.tags
                    }
                }
            
            return config
    
    def clear_cache(self) -> None:
        """Clear all strategy instance caches."""
        with self._lock:
            for entry in self.strategies.values():
                entry.instance_cache.clear()
            
            logger.info("Cleared all strategy caches")
    
    def shutdown(self) -> None:
        """Shutdown the registry."""
        with self._lock:
            self.clear_cache()
            self.strategies.clear()
            self.plugins.clear()
            self.strategy_groups.clear()
            self.active_ab_tests.clear()
            
            logger.info("ExtractionRegistry shutdown complete")

# Global registry instance
_registry_instance = None
_registry_lock = threading.Lock()

def get_registry(config: Optional[Dict[str, Any]] = None) -> ExtractionRegistry:
    """Get the global registry instance."""
    global _registry_instance
    
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = ExtractionRegistry(config)
        elif config:
            # Update config if provided
            _registry_instance.config.update(config)
        
        return _registry_instance