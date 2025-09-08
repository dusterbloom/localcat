"""
Memory Service Factory Module

Centralized service creation and dependency injection for the HotMem system.
Provides singleton management, lifecycle control, and configuration-driven initialization.

Author: SOLID Refactoring
"""

import os
import time
from typing import Dict, Any, Optional, Type, List, Callable
from dataclasses import dataclass
import threading
from loguru import logger

from components.memory.memory_interfaces import (
    IMemoryExtractor, IMemoryStorage, IMemoryRetriever, 
    IMemoryQuality, IMemoryMetrics, IMemoryService,
    IExtractionStrategy, IRetrievalStrategy, IConfigurationProvider
)
from components.memory.memory_store import MemoryStore

@dataclass
class ServiceConfig:
    """Configuration for service creation."""
    service_type: Type
    dependencies: List[str] = None
    singleton: bool = True
    factory_func: Callable = None
    config_section: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class MemoryServiceFactory:
    """
    Centralized service factory for the HotMem system.
    
    Responsibilities:
    - Service creation and dependency injection
    - Singleton management and lifecycle control
    - Configuration-driven initialization
    - Service health monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Service registry
        self.services: Dict[str, Any] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.singletons: Dict[str, Any] = {}
        
        # Lifecycle management
        self.initialized = False
        self.startup_time = time.time()
        self.shutdown_hooks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register default services
        self._register_default_services()
        
        logger.info("MemoryServiceFactory initialized")
    
    def _register_default_services(self) -> None:
        """Register default service configurations."""
        from components.memory.memory_extractor import MemoryExtractor
        from components.memory.memory_storage import MemoryStorage
        from components.memory.memory_retriever import MemoryRetriever
        from components.memory.memory_quality import MemoryQuality
        from components.memory.memory_metrics import MemoryMetrics
        
        # Core services
        self.register_service('memory_store', MemoryStore, singleton=True)
        self.register_service('memory_extractor', MemoryExtractor, 
                            dependencies=['memory_store'], singleton=True)
        self.register_service('memory_storage', MemoryStorage, 
                            dependencies=['memory_store'], singleton=True)
        self.register_service('memory_retriever', MemoryRetriever, 
                            dependencies=['memory_store'], singleton=True)
        self.register_service('memory_quality', MemoryQuality, singleton=True)
        self.register_service('memory_metrics', MemoryMetrics, singleton=True)
    
    def register_service(self, name: str, service_type: Type, 
                        dependencies: List[str] = None, 
                        singleton: bool = True,
                        factory_func: Callable = None,
                        config_section: str = None) -> None:
        """
        Register a service configuration.
        
        Args:
            name: Service name
            service_type: Service class type
            dependencies: List of dependency service names
            singleton: Whether this service should be a singleton
            factory_func: Optional factory function for creation
            config_section: Configuration section name
        """
        with self._lock:
            service_config = ServiceConfig(
                service_type=service_type,
                dependencies=dependencies or [],
                singleton=singleton,
                factory_func=factory_func,
                config_section=config_section
            )
            
            self.service_configs[name] = service_config
            logger.info(f"Registered service: {name} -> {service_type.__name__}")
    
    def get_service(self, name: str) -> Any:
        """
        Get a service instance, creating it if necessary.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        with self._lock:
            if name not in self.service_configs:
                raise ValueError(f"Service '{name}' is not registered")
            
            # Check if singleton already exists
            if self.service_configs[name].singleton and name in self.singletons:
                return self.singletons[name]
            
            # Create service instance
            service = self._create_service(name)
            
            # Store singleton if applicable
            if self.service_configs[name].singleton:
                self.singletons[name] = service
            
            return service
    
    def _create_service(self, name: str) -> Any:
        """Create a service instance with dependency injection."""
        config = self.service_configs[name]
        
        # Resolve dependencies
        dependencies = {}
        for dep_name in config.dependencies:
            dependencies[dep_name] = self.get_service(dep_name)
        
        # Get service configuration
        service_config = self._get_service_config(name, config.config_section)
        
        # Create service instance
        if config.factory_func:
            instance = config.factory_func(dependencies, service_config)
        else:
            if dependencies:
                instance = config.service_type(**dependencies, config=service_config)
            else:
                instance = config.service_type(config=service_config)
        
        logger.info(f"Created service instance: {name}")
        return instance
    
    def _get_service_config(self, service_name: str, config_section: str = None) -> Dict[str, Any]:
        """Get configuration for a service."""
        section_name = config_section or service_name
        
        # Get from main config
        if section_name in self.config:
            return self.config[section_name]
        
        # Return empty config if not found
        return {}
    
    def initialize_services(self) -> None:
        """Initialize all registered services."""
        with self._lock:
            if self.initialized:
                logger.warning("Services already initialized")
                return
            
            logger.info("Initializing all services...")
            
            # Initialize all services
            for service_name in self.service_configs:
                try:
                    self.get_service(service_name)
                except Exception as e:
                    logger.error(f"Failed to initialize service '{service_name}': {e}")
            
            self.initialized = True
            logger.info("All services initialized successfully")
    
    def get_memory_service(self) -> IMemoryService:
        """Get the main memory service instance."""
        return self.get_service('memory_service')
    
    def get_extractor(self) -> IMemoryExtractor:
        """Get the memory extractor instance."""
        return self.get_service('memory_extractor')
    
    def get_storage(self) -> IMemoryStorage:
        """Get the memory storage instance."""
        return self.get_service('memory_storage')
    
    def get_retriever(self) -> IMemoryRetriever:
        """Get the memory retriever instance."""
        return self.get_service('memory_retriever')
    
    def get_quality(self) -> IMemoryQuality:
        """Get the memory quality instance."""
        return self.get_service('memory_quality')
    
    def get_metrics(self) -> IMemoryMetrics:
        """Get the memory metrics instance."""
        return self.get_service('memory_metrics')
    
    def get_store(self) -> MemoryStore:
        """Get the memory store instance."""
        return self.get_service('memory_store')
    
    def register_shutdown_hook(self, hook: Callable) -> None:
        """Register a shutdown hook."""
        self.shutdown_hooks.append(hook)
    
    def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        logger.info("Shutting down MemoryServiceFactory...")
        
        # Call shutdown hooks
        for hook in self.shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Shutdown hook failed: {e}")
        
        # Clear singletons
        self.singletons.clear()
        
        logger.info("MemoryServiceFactory shutdown complete")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            'initialized': self.initialized,
            'startup_time': self.startup_time,
            'uptime_seconds': time.time() - self.startup_time,
            'registered_services': list(self.service_configs.keys()),
            'singleton_services': list(self.singletons.keys()),
            'service_count': len(self.service_configs),
            'singleton_count': len(self.singletons),
        }
        
        return status
    
    def reload_service(self, service_name: str) -> bool:
        """
        Reload a service instance.
        
        Args:
            service_name: Name of service to reload
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if service_name not in self.service_configs:
                logger.error(f"Service '{service_name}' is not registered")
                return False
            
            config = self.service_configs[service_name]
            
            if not config.singleton:
                logger.warning(f"Service '{service_name}' is not a singleton, cannot reload")
                return False
            
            # Remove existing singleton
            if service_name in self.singletons:
                del self.singletons[service_name]
            
            # Create new instance
            try:
                self.get_service(service_name)
                logger.info(f"Reloaded service: {service_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to reload service '{service_name}': {e}")
                return False
    
    def create_memory_service(self, config: Dict[str, Any] = None) -> IMemoryService:
        """
        Create a complete memory service with all components.
        
        Args:
            config: Configuration for the memory service
            
        Returns:
            Configured memory service instance
        """
        if config:
            self.config.update(config)
        
        # Initialize all services if not already done
        if not self.initialized:
            self.initialize_services()
        
        # Create and return the main memory service
        return self.get_memory_service()
    
    def register_extraction_strategy(self, name: str, strategy: IExtractionStrategy) -> None:
        """Register an extraction strategy."""
        self.register_service(f'extraction_strategy_{name}', 
                            type(strategy), 
                            singleton=True,
                            factory_func=lambda deps, cfg: strategy)
    
    def register_retrieval_strategy(self, name: str, strategy: IRetrievalStrategy) -> None:
        """Register a retrieval strategy."""
        self.register_service(f'retrieval_strategy_{name}', 
                            type(strategy), 
                            singleton=True,
                            factory_func=lambda deps, cfg: strategy)
    
    def get_extraction_strategies(self) -> Dict[str, IExtractionStrategy]:
        """Get all registered extraction strategies."""
        strategies = {}
        
        for service_name, config in self.service_configs.items():
            if service_name.startswith('extraction_strategy_'):
                strategy_name = service_name.replace('extraction_strategy_', '')
                try:
                    strategies[strategy_name] = self.get_service(service_name)
                except Exception as e:
                    logger.warning(f"Failed to get extraction strategy '{strategy_name}': {e}")
        
        return strategies
    
    def get_retrieval_strategies(self) -> Dict[str, IRetrievalStrategy]:
        """Get all registered retrieval strategies."""
        strategies = {}
        
        for service_name, config in self.service_configs.items():
            if service_name.startswith('retrieval_strategy_'):
                strategy_name = service_name.replace('retrieval_strategy_', '')
                try:
                    strategies[strategy_name] = self.get_service(service_name)
                except Exception as e:
                    logger.warning(f"Failed to get retrieval strategy '{strategy_name}': {e}")
        
        return strategies
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

# Global factory instance
_factory_instance = None
_factory_lock = threading.Lock()

def get_factory(config: Optional[Dict[str, Any]] = None) -> MemoryServiceFactory:
    """Get the global factory instance."""
    global _factory_instance
    
    with _factory_lock:
        if _factory_instance is None:
            _factory_instance = MemoryServiceFactory(config)
        elif config:
            _factory_instance.config.update(config)
        
        return _factory_instance

def create_memory_service(config: Optional[Dict[str, Any]] = None) -> IMemoryService:
    """Create a memory service using the global factory."""
    factory = get_factory(config)
    return factory.create_memory_service(config)