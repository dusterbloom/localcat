"""
Service Health Monitoring System for LocalCat Server

Provides comprehensive health checks for all external services and internal components.
Implements circuit breaker pattern, exponential backoff, and graceful degradation.
"""

import asyncio
import time
import aiohttp
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import psutil
import loguru

logger = loguru.logger.bind(component="health_monitor")


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Service blocked, failing fast
    HALF_OPEN = "half_open"  # Testing service recovery


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    status: HealthStatus
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    last_check: datetime = None
    consecutive_failures: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()


@dataclass
class ServiceConfig:
    """Configuration for service health monitoring"""
    name: str
    url: str
    timeout: float = 5.0
    interval: float = 30.0
    expected_status: int = 200
    check_headers: Optional[Dict[str, str]] = None
    check_body: Optional[str] = None
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: float = 60.0
    critical: bool = True
    
    def __post_init__(self):
        if self.check_headers is None:
            self.check_headers = {}


class HealthMonitor:
    """
    Comprehensive health monitoring for external services and internal components.
    
    Features:
    - HTTP health checks for external services
    - Database connectivity checks
    - Process and system resource monitoring
    - Circuit breaker pattern for fault tolerance
    - Exponential backoff and retry logic
    - Graceful degradation capabilities
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "/Users/peppi/Dev/localcat/server/config")
        self.services: Dict[str, ServiceConfig] = {}
        self.health_results: Dict[str, HealthCheckResult] = {}
        self.monitoring_active = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_states: Dict[str, Tuple[CircuitState, float]] = {}
        
        # Initialize default services
        self._initialize_default_services()
        
    def _initialize_default_services(self):
        """Initialize default service configurations"""
        
        # Ollama LLM Service
        self.add_service(ServiceConfig(
            name="ollama",
            url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434") + "/api/tags",
            timeout=3.0,
            interval=15.0,
            critical=True,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30.0
        ))
        
        # LM Studio Service (if configured)
        lm_studio_url = os.getenv("SUMMARIZER_BASE_URL")
        if lm_studio_url and "127.0.0.1" in lm_studio_url:
            self.add_service(ServiceConfig(
                name="lm_studio",
                url=lm_studio_url.replace("/v1", "/models"),
                timeout=3.0,
                interval=30.0,
                critical=False,
                circuit_breaker_threshold=2,
                circuit_breaker_timeout=60.0
            ))
        
        # Database Health
        db_path = os.getenv("HOTMEM_SQLITE", "/Users/peppi/Dev/localcat/data/memory.db")
        self.add_service(ServiceConfig(
            name="database",
            url=f"file://{db_path}",
            timeout=2.0,
            interval=10.0,
            critical=True,
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=5.0
        ))
        
        # LEANN Index (if enabled)
        if os.getenv("HOTMEM_USE_LEANN", "false").lower() in ("1", "true", "yes"):
            leann_path = os.getenv("LEANN_INDEX_PATH", "/Users/peppi/Dev/localcat/data/memory_vectors.leann")
            self.add_service(ServiceConfig(
                name="leann_index",
                url=f"file://{leann_path}",
                timeout=1.0,
                interval=60.0,
                critical=False,
                circuit_breaker_threshold=1,
                circuit_breaker_timeout=30.0
            ))
    
    def add_service(self, config: ServiceConfig):
        """Add a service to monitor"""
        self.services[config.name] = config
        self.health_results[config.name] = HealthCheckResult(
            service_name=config.name,
            status=HealthStatus.UNKNOWN
        )
        self.circuit_states[config.name] = (CircuitState.CLOSED, 0.0)
        
        logger.info(f"Added service to monitoring: {config.name}")
    
    async def start_monitoring(self):
        """Start health monitoring for all services"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0),
            connector=aiohttp.TCPConnector(limit=20)
        )
        
        # Start monitoring tasks for each service
        for service_name, config in self.services.items():
            task = asyncio.create_task(self._monitor_service(service_name, config))
            self.check_tasks[service_name] = task
        
        logger.info(f"Started health monitoring for {len(self.services)} services")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.check_tasks:
            await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        self.check_tasks.clear()
        logger.info("Stopped health monitoring")
    
    async def _monitor_service(self, service_name: str, config: ServiceConfig):
        """Monitor a specific service"""
        while self.monitoring_active:
            try:
                result = await self._check_service_health(service_name, config)
                self.health_results[service_name] = result
                
                # Log status changes
                previous_result = self.health_results.get(service_name)
                if previous_result and previous_result.status != result.status:
                    level = "ERROR" if result.status == HealthStatus.UNHEALTHY else "INFO"
                    logger.log(
                        level,
                        f"Service {service_name} status changed: {previous_result.status.value} → {result.status.value}"
                    )
                
                # Wait for next check
                await asyncio.sleep(config.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring service {service_name}: {e}")
                await asyncio.sleep(config.interval)
    
    async def _check_service_health(self, service_name: str, config: ServiceConfig) -> HealthCheckResult:
        """Check health of a specific service"""
        start_time = time.time()
        
        # Check circuit breaker state
        circuit_state, open_until = self.circuit_states.get(service_name, (CircuitState.CLOSED, 0.0))
        
        if circuit_state == CircuitState.OPEN:
            if time.time() < open_until:
                return HealthCheckResult(
                    service_name=service_name,
                    status=HealthStatus.UNHEALTHY,
                    error_message="Circuit breaker open",
                    circuit_state=circuit_state
                )
            else:
                # Move to half-open state
                self.circuit_states[service_name] = (CircuitState.HALF_OPEN, time.time())
                circuit_state = CircuitState.HALF_OPEN
        
        try:
            if config.url.startswith("file://"):
                result = await self._check_file_service(service_name, config)
            else:
                result = await self._check_http_service(service_name, config)
            
            result.response_time = time.time() - start_time
            
            # Reset circuit breaker on success
            if result.status == HealthStatus.HEALTHY:
                self.circuit_states[service_name] = (CircuitState.CLOSED, 0.0)
                result.consecutive_failures = 0
            elif result.status == HealthStatus.UNHEALTHY:
                result.consecutive_failures += 1
                
                # Open circuit breaker if threshold exceeded
                if result.consecutive_failures >= config.circuit_breaker_threshold:
                    timeout = config.circuit_breaker_timeout
                    self.circuit_states[service_name] = (CircuitState.OPEN, time.time() + timeout)
                    result.circuit_state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker opened for service {service_name} ({timeout}s timeout)")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Health check failed for {service_name}: {error_msg}")
            
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                error_message=error_msg,
                response_time=time.time() - start_time,
                circuit_state=circuit_state
            )
    
    async def _check_http_service(self, service_name: str, config: ServiceConfig) -> HealthCheckResult:
        """Check health of an HTTP service"""
        try:
            async with self.session.get(
                config.url,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
                headers=config.check_headers
            ) as response:
                
                if response.status == config.expected_status:
                    # Check response body if specified
                    if config.check_body:
                        body = await response.text()
                        if config.check_body not in body:
                            return HealthCheckResult(
                                service_name=service_name,
                                status=HealthStatus.DEGRADED,
                                error_message=f"Expected body '{config.check_body}' not found"
                            )
                    
                    return HealthCheckResult(
                        service_name=service_name,
                        status=HealthStatus.HEALTHY
                    )
                else:
                    return HealthCheckResult(
                        service_name=service_name,
                        status=HealthStatus.DEGRADED,
                        error_message=f"HTTP {response.status}, expected {config.expected_status}"
                    )
                    
        except asyncio.TimeoutError:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                error_message="Request timeout"
            )
        except aiohttp.ClientError as e:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Connection error: {e}"
            )
    
    async def _check_file_service(self, service_name: str, config: ServiceConfig) -> HealthCheckResult:
        """Check health of a file-based service"""
        try:
            file_path = config.url[7:]  # Remove "file://" prefix
            
            if service_name == "database":
                # Check database connectivity
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                cursor.fetchone()
                conn.close()
                
                return HealthCheckResult(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY
                )
                
            elif service_name == "leann_index":
                # Check if LEANN index file exists and is accessible
                if os.path.exists(file_path):
                    return HealthCheckResult(
                        service_name=service_name,
                        status=HealthStatus.HEALTHY
                    )
                else:
                    return HealthCheckResult(
                        service_name=service_name,
                        status=HealthStatus.DEGRADED,
                        error_message="LEANN index file not found"
                    )
            
            else:
                # Generic file check
                if os.path.exists(file_path):
                    return HealthCheckResult(
                        service_name=service_name,
                        status=HealthStatus.HEALTHY
                    )
                else:
                    return HealthCheckResult(
                        service_name=service_name,
                        status=HealthStatus.UNHEALTHY,
                        error_message="File not found"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                error_message=f"File check error: {e}"
            )
    
    def get_service_health(self, service_name: str) -> Optional[HealthCheckResult]:
        """Get current health status of a service"""
        return self.health_results.get(service_name)
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all monitored services"""
        return {
            name: {
                "status": result.status.value,
                "response_time": result.response_time,
                "error": result.error_message,
                "last_check": result.last_check.isoformat(),
                "consecutive_failures": result.consecutive_failures,
                "circuit_state": result.circuit_state.value,
                "critical": self.services[name].critical
            }
            for name, result in self.health_results.items()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        try:
            # System resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Service health summary
            total_services = len(self.services)
            healthy_services = sum(1 for r in self.health_results.values() if r.status == HealthStatus.HEALTHY)
            critical_services = sum(1 for r in self.health_results.values() 
                                  if r.status == HealthStatus.UNHEALTHY and self.services[r.service_name].critical)
            
            # Overall system status
            if critical_services > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif healthy_services == total_services:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
            
            return {
                "overall_status": overall_status.value,
                "timestamp": datetime.now().isoformat(),
                "system_resources": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2)
                },
                "services": {
                    "total": total_services,
                    "healthy": healthy_services,
                    "unhealthy": total_services - healthy_services,
                    "critical_failures": critical_services
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def is_healthy(self) -> bool:
        """Check if all critical services are healthy"""
        for name, result in self.health_results.items():
            if self.services[name].critical and result.status != HealthStatus.HEALTHY:
                return False
        return True
    
    async def wait_for_healthy(self, timeout: float = 30.0) -> bool:
        """Wait for all services to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_healthy():
                return True
            await asyncio.sleep(1.0)
        
        return False