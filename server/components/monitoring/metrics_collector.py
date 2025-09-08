"""
Metrics Collection System for LocalCat Server

Collects, aggregates, and provides access to performance metrics across all system components.
Implements time-series data collection, aggregation, and query capabilities.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import sqlite3
import os
import psutil
import loguru

logger = loguru.logger.bind(component="metrics_collector")


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags
        }


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected"""
    name: str
    type: str  # "gauge", "counter", "histogram", "timer"
    description: str
    tags: Dict[str, str]
    aggregation: str = "avg"  # "avg", "sum", "min", "max", "count"
    retention_hours: int = 24  # How long to keep this metric


class MetricsCollector:
    """
    Comprehensive metrics collection system for LocalCat components.
    
    Features:
    - Time-series metric collection
    - Multiple metric types (gauge, counter, histogram, timer)
    - Automatic aggregation and rollup
    - Persistent storage with SQLite
    - Query and export capabilities
    - Performance monitoring with minimal overhead
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "/Users/peppi/Dev/localcat/data/metrics.db"
        self.metrics: Dict[str, MetricDefinition] = {}
        self.data_points: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collectors: Dict[str, Callable] = {}
        self.collection_active = False
        self.collection_interval = 5.0  # seconds
        self.collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Register default metrics
        self._register_default_metrics()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    aggregation TEXT DEFAULT 'avg',
                    retention_hours INTEGER DEFAULT 24,
                    created_at REAL
                )
            ''')
            
            # Create metric data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_id INTEGER,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    tags TEXT,
                    FOREIGN KEY (metric_id) REFERENCES metrics (id)
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_data_name_time ON metric_data(metric_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_data_timestamp ON metric_data(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Metrics database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
    
    def _register_default_metrics(self):
        """Register default system and application metrics"""
        
        # System metrics
        self.register_metric(MetricDefinition(
            name="system.cpu_percent",
            type="gauge",
            description="CPU utilization percentage",
            tags={"component": "system"},
            aggregation="avg",
            retention_hours=168  # 7 days
        ))
        
        self.register_metric(MetricDefinition(
            name="system.memory_percent",
            type="gauge",
            description="Memory utilization percentage",
            tags={"component": "system"},
            aggregation="avg",
            retention_hours=168
        ))
        
        self.register_metric(MetricDefinition(
            name="system.memory_available_gb",
            type="gauge",
            description="Available memory in GB",
            tags={"component": "system"},
            aggregation="avg",
            retention_hours=168
        ))
        
        self.register_metric(MetricDefinition(
            name="system.disk_percent",
            type="gauge",
            description="Disk utilization percentage",
            tags={"component": "system"},
            aggregation="avg",
            retention_hours=168
        ))
        
        # Application metrics
        self.register_metric(MetricDefinition(
            name="app.memory_entities",
            type="gauge",
            description="Number of entities in memory database",
            tags={"component": "memory"},
            aggregation="avg",
            retention_hours=72
        ))
        
        self.register_metric(MetricDefinition(
            name="app.memory_edges",
            type="gauge",
            description="Number of edges in memory database",
            tags={"component": "memory"},
            aggregation="avg",
            retention_hours=72
        ))
        
        self.register_metric(MetricDefinition(
            name="app.memory_mentions",
            type="gauge",
            description="Number of mentions in memory database",
            tags={"component": "memory"},
            aggregation="avg",
            retention_hours=72
        ))
        
        # Performance metrics
        self.register_metric(MetricDefinition(
            name="perf.extraction_latency",
            type="timer",
            description="Memory extraction latency in milliseconds",
            tags={"component": "extraction"},
            aggregation="avg",
            retention_hours=48
        ))
        
        self.register_metric(MetricDefinition(
            name="perf.retrieval_latency",
            type="timer",
            description="Memory retrieval latency in milliseconds",
            tags={"component": "retrieval"},
            aggregation="avg",
            retention_hours=48
        ))
        
        self.register_metric(MetricDefinition(
            name="perf.tts_latency",
            type="timer",
            description="TTS processing latency in milliseconds",
            tags={"component": "tts"},
            aggregation="avg",
            retention_hours=48
        ))
        
        # Register system collectors
        self.register_collector("system_metrics", self._collect_system_metrics)
        self.register_collector("memory_metrics", self._collect_memory_metrics)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition"""
        with self._lock:
            self.metrics[metric_def.name] = metric_def
            
            # Store in database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO metrics 
                    (name, type, description, tags, aggregation, retention_hours, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_def.name,
                    metric_def.type,
                    metric_def.description,
                    json.dumps(metric_def.tags),
                    metric_def.aggregation,
                    metric_def.retention_hours,
                    time.time()
                ))
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Failed to register metric {metric_def.name}: {e}")
        
        logger.debug(f"Registered metric: {metric_def.name}")
    
    def register_collector(self, name: str, collector_func: Callable):
        """Register a metrics collector function"""
        self.collectors[name] = collector_func
        logger.debug(f"Registered collector: {name}")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        if name not in self.metrics:
            logger.warning(f"Unknown metric: {name}")
            return
        
        metric_def = self.metrics[name]
        combined_tags = {**metric_def.tags, **(tags or {})}
        
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=combined_tags
        )
        
        with self._lock:
            self.data_points[name].append(point)
        
        # Store in database asynchronously
        asyncio.create_task(self._store_metric_async(name, point))
    
    async def _store_metric_async(self, name: str, point: MetricPoint):
        """Store metric point in database asynchronously"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get metric ID
            cursor.execute("SELECT id FROM metrics WHERE name = ?", (name,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Metric {name} not found in database")
                return
            
            metric_id = result[0]
            
            # Store data point
            cursor.execute('''
                INSERT INTO metric_data (metric_id, timestamp, value, tags)
                VALUES (?, ?, ?, ?)
            ''', (metric_id, point.timestamp, point.value, json.dumps(point.tags)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store metric {name}: {e}")
    
    def record_timer(self, name: str, start_time: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric (duration since start_time)"""
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.record_metric(name, duration, tags)
    
    async def start_collection(self):
        """Start automatic metrics collection"""
        if self.collection_active:
            logger.warning("Metrics collection already active")
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop automatic metrics collection"""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.collection_active:
            try:
                # Run all collectors
                for name, collector in self.collectors.items():
                    try:
                        await collector()
                    except Exception as e:
                        logger.error(f"Error in collector {name}: {e}")
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric("system.cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_percent", memory.percent)
            self.record_metric("system.memory_available_gb", round(memory.available / (1024**3), 2))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk_percent", disk.percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_memory_metrics(self):
        """Collect memory database metrics"""
        try:
            db_path = os.getenv("HOTMEM_SQLITE", "/Users/peppi/Dev/localcat/data/memory.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Count entities
                cursor.execute("SELECT COUNT(*) FROM entity")
                entity_count = cursor.fetchone()[0]
                self.record_metric("app.memory_entities", entity_count)
                
                # Count edges
                cursor.execute("SELECT COUNT(*) FROM edge")
                edge_count = cursor.fetchone()[0]
                self.record_metric("app.memory_edges", edge_count)
                
                # Count mentions
                cursor.execute("SELECT COUNT(*) FROM mention")
                mention_count = cursor.fetchone()[0]
                self.record_metric("app.memory_mentions", mention_count)
                
                conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metric data based on retention policies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get retention periods
            cursor.execute("SELECT name, retention_hours FROM metrics")
            retention_periods = dict(cursor.fetchall())
            
            # Delete old data
            cutoff_time = time.time()
            for metric_name, retention_hours in retention_periods.items():
                cutoff = cutoff_time - (retention_hours * 3600)
                cursor.execute('''
                    DELETE FROM metric_data 
                    WHERE metric_id = (SELECT id FROM metrics WHERE name = ?)
                    AND timestamp < ?
                ''', (metric_name, cutoff))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_metric_data(self, name: str, start_time: Optional[float] = None, 
                       end_time: Optional[float] = None, limit: int = 100) -> List[MetricPoint]:
        """Get metric data for a specific metric"""
        if name not in self.metrics:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT md.timestamp, md.value, md.tags
                FROM metric_data md
                JOIN metrics m ON md.metric_id = m.id
                WHERE m.name = ?
            '''
            params = [name]
            
            if start_time:
                query += ' AND md.timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND md.timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY md.timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            points = []
            for timestamp, value, tags_json in rows:
                tags = json.loads(tags_json) if tags_json else {}
                points.append(MetricPoint(timestamp, value, tags))
            
            conn.close()
            return points
            
        except Exception as e:
            logger.error(f"Error getting metric data for {name}: {e}")
            return []
    
    def get_aggregated_metrics(self, name: str, interval: str = "1h", 
                               start_time: Optional[float] = None, 
                               end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get aggregated metric data"""
        if name not in self.metrics:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Parse interval (e.g., "1h", "30m", "1d")
            interval_seconds = self._parse_interval(interval)
            
            query = f'''
                SELECT 
                    (md.timestamp - (md.timestamp % {interval_seconds})) as bucket,
                    AVG(md.value) as avg_value,
                    MIN(md.value) as min_value,
                    MAX(md.value) as max_value,
                    COUNT(md.value) as count
                FROM metric_data md
                JOIN metrics m ON md.metric_id = m.id
                WHERE m.name = ?
            '''
            params = [name]
            
            if start_time:
                query += ' AND md.timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND md.timestamp <= ?'
                params.append(end_time)
            
            query += f' GROUP BY (md.timestamp - (md.timestamp % {interval_seconds}))'
            query += ' ORDER BY bucket DESC LIMIT 100'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for bucket, avg_val, min_val, max_val, count in rows:
                results.append({
                    "timestamp": bucket,
                    "avg": round(avg_val, 3),
                    "min": round(min_val, 3),
                    "max": round(max_val, 3),
                    "count": count
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting aggregated metrics for {name}: {e}")
            return []
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds"""
        if interval.endswith('s'):
            return int(interval[:-1])
        elif interval.endswith('m'):
            return int(interval[:-1]) * 60
        elif interval.endswith('h'):
            return int(interval[:-1]) * 3600
        elif interval.endswith('d'):
            return int(interval[:-1]) * 86400
        else:
            return int(interval)  # Assume seconds
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current values of all metrics"""
        current_metrics = {}
        
        for name in self.metrics:
            recent_points = list(self.data_points[name])
            if recent_points:
                current_metrics[name] = recent_points[-1].value
            else:
                current_metrics[name] = None
        
        return current_metrics
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format"""
        data = {
            "timestamp": time.time(),
            "metrics": {}
        }
        
        for name, points in self.data_points.items():
            if points:
                data["metrics"][name] = [point.to_dict() for point in list(points)[-100:]]
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "total_metrics": len(self.metrics),
            "active_collectors": len(self.collectors),
            "collection_active": self.collection_active,
            "database_path": self.db_path,
            "metrics_by_type": defaultdict(int)
        }
        
        for metric in self.metrics.values():
            summary["metrics_by_type"][metric.type] += 1
        
        return dict(summary)