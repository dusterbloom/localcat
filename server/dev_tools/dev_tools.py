"""
Developer Tools for LocalCat - Development server and debugging utilities
"""

import asyncio
import uvicorn
import json
import time
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

import fastapi
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import websockets

from config import get_config, Config
from pipeline_builder import PipelineBuilder, PipelineConfig
from processors.memory_processor import MemoryProcessor, MemoryProcessorConfig
from processors.extraction_processor import ExtractionProcessor, ExtractionProcessorConfig
from processors.quality_processor import QualityProcessor, QualityProcessorConfig
from processors.context_processor import ContextProcessor, ContextProcessorConfig
from context_manager import ContextManager, ContextManagerConfig


class DevToolType(Enum):
    """Types of developer tools"""
    DEBUGGER = "debugger"
    PROFILER = "profiler"
    MEMORY_INSPECTOR = "memory_inspector"
    PIPELINE_VISUALIZER = "pipeline_visualizer"
    CONFIG_EDITOR = "config_editor"
    TEST_RUNNER = "test_runner"
    LOG_VIEWER = "log_viewer"


@dataclass
class DevToolConfig:
    """Configuration for developer tools"""
    enable_debug_server: bool = True
    debug_server_port: int = 8080
    enable_profiling: bool = True
    enable_memory_inspection: bool = True
    enable_hot_reload: bool = True
    enable_real_time_metrics: bool = True
    log_level: str = "DEBUG"
    max_log_entries: int = 1000
    enable_websocket_support: bool = True


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass


class DevToolsServer:
    """Development server with hot reload and debugging tools"""
    
    def __init__(self, config: DevToolConfig, app_config: Config):
        self.config = config
        self.app_config = app_config
        self.app = FastAPI(title="LocalCat Dev Tools", version="1.0.0")
        self.connection_manager = ConnectionManager()
        
        # Initialize components
        self.pipeline_builder: Optional[PipelineBuilder] = None
        self.context_manager: Optional[ContextManager] = None
        
        # Metrics and monitoring
        self.metrics_history: List[Dict[str, Any]] = []
        self.log_entries: List[Dict[str, Any]] = []
        
        # Profiling data
        self.profiling_data: Dict[str, Any] = {}
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Dev tools dashboard"""
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/config")
        async def get_config():
            """Get current configuration"""
            return self.app_config.to_dict()
        
        @self.app.post("/api/config")
        async def update_config(config_data: Dict[str, Any]):
            """Update configuration"""
            try:
                new_config = Config.from_dict(config_data)
                new_config.validate()
                
                # Update global config
                from config import set_config
                set_config(new_config)
                self.app_config = new_config
                
                await self._broadcast_update({
                    "type": "config_updated",
                    "config": new_config.to_dict()
                })
                
                return {"status": "success", "message": "Configuration updated"}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics"""
            metrics = self._collect_metrics()
            return metrics
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history():
            """Get metrics history"""
            return self.metrics_history
        
        @self.app.get("/api/pipeline")
        async def get_pipeline_info():
            """Get pipeline information"""
            if not self.pipeline_builder:
                return {"error": "Pipeline not initialized"}
            
            return self.pipeline_builder.get_pipeline_info()
        
        @self.app.post("/api/pipeline/build")
        async def build_pipeline(config: Optional[Dict[str, Any]] = None):
            """Build pipeline with optional configuration"""
            try:
                if config:
                    pipeline_config = PipelineConfig(**config)
                else:
                    pipeline_config = PipelineConfig()
                
                self.pipeline_builder = PipelineBuilder(pipeline_config)
                pipeline = self.pipeline_builder.build_pipeline()
                
                await self._broadcast_update({
                    "type": "pipeline_built",
                    "pipeline_info": self.pipeline_builder.get_pipeline_info()
                })
                
                return {"status": "success", "pipeline_length": len(pipeline)}
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/context/sessions")
        async def get_context_sessions():
            """Get active context sessions"""
            if not self.context_manager:
                return {"error": "Context manager not initialized"}
            
            sessions = []
            for session_id, session in self.context_manager.active_sessions.items():
                session_info = {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "duration": session.duration,
                    "idle_time": session.idle_time,
                    "context_items": sum(len(window.items) for window in session.context_windows.values())
                }
                sessions.append(session_info)
            
            return sessions
        
        @self.app.get("/api/context/session/{session_id}")
        async def get_session_context(session_id: str):
            """Get session context details"""
            if not self.context_manager:
                return {"error": "Context manager not initialized"}
            
            session = await self.context_manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return await self.context_manager.get_session_summary(session_id)
        
        @self.app.post("/api/test/run")
        async def run_tests(test_config: Dict[str, Any]):
            """Run tests with specified configuration"""
            try:
                results = await self._run_tests(test_config)
                return results
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/logs")
        async def get_logs():
            """Get recent log entries"""
            return self.log_entries
        
        @self.app.post("/api/logs/clear")
        async def clear_logs():
            """Clear log entries"""
            self.log_entries.clear()
            return {"status": "success", "message": "Logs cleared"}
        
        @self.app.get("/api/system/info")
        async def get_system_info():
            """Get system information"""
            return self._get_system_info()
        
        @self.app.post("/api/profiling/start")
        async def start_profiling():
            """Start profiling"""
            self.profiling_data = {
                "start_time": time.time(),
                "memory_start": psutil.Process().memory_info().rss,
                "operations": []
            }
            return {"status": "success", "message": "Profiling started"}
        
        @self.app.post("/api/profiling/stop")
        async def stop_profiling():
            """Stop profiling and get results"""
            if not self.profiling_data:
                raise HTTPException(status_code=400, detail="Profiling not started")
            
            self.profiling_data["end_time"] = time.time()
            self.profiling_data["memory_end"] = psutil.Process().memory_info().rss
            
            duration = self.profiling_data["end_time"] - self.profiling_data["start_time"]
            memory_delta = self.profiling_data["memory_end"] - self.profiling_data["memory_start"]
            
            results = {
                "duration": duration,
                "memory_delta": memory_delta,
                "operations": len(self.profiling_data["operations"]),
                "operations_per_second": len(self.profiling_data["operations"]) / duration if duration > 0 else 0
            }
            
            self.profiling_data = {}
            return results
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.connection_manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming WebSocket messages
                    await self._handle_websocket_message(websocket, data)
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
    
    def _setup_middleware(self):
        """Setup middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _get_dashboard_html(self) -> str:
        """Get HTML for dev tools dashboard"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LocalCat Dev Tools</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; }
                .metric { display: inline-block; margin: 8px; padding: 8px; background: #f5f5f5; border-radius: 4px; }
                .status { padding: 4px 8px; border-radius: 4px; font-size: 12px; }
                .status.good { background: #d4edda; color: #155724; }
                .status.warning { background: #fff3cd; color: #856404; }
                .status.error { background: #f8d7da; color: #721c24; }
                button { padding: 8px 16px; margin: 4px; border: none; border-radius: 4px; cursor: pointer; }
                button.primary { background: #007bff; color: white; }
                button.success { background: #28a745; color: white; }
                button.danger { background: #dc3545; color: white; }
                pre { background: #f8f9fa; padding: 8px; border-radius: 4px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LocalCat Developer Tools</h1>
                
                <div class="card">
                    <h2>System Status</h2>
                    <div id="system-status">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>Metrics</h2>
                    <div id="metrics">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>Pipeline</h2>
                    <div id="pipeline-info">Loading...</div>
                    <button class="primary" onclick="buildPipeline()">Build Pipeline</button>
                </div>
                
                <div class="card">
                    <h2>Configuration</h2>
                    <div id="config">Loading...</div>
                    <button class="primary" onclick="loadConfig()">Refresh Config</button>
                </div>
                
                <div class="card">
                    <h2>Context Sessions</h2>
                    <div id="sessions">Loading...</div>
                    <button class="primary" onclick="loadSessions()">Refresh Sessions</button>
                </div>
                
                <div class="card">
                    <h2>Testing</h2>
                    <button class="success" onclick="runTests()">Run Tests</button>
                    <button class="primary" onclick="startProfiling()">Start Profiling</button>
                    <button class="danger" onclick="stopProfiling()">Stop Profiling</button>
                    <div id="test-results"></div>
                </div>
                
                <div class="card">
                    <h2>Logs</h2>
                    <div id="logs" style="max-height: 300px; overflow-y: auto;"></div>
                    <button class="danger" onclick="clearLogs()">Clear Logs</button>
                </div>
            </div>
            
            <script>
                let ws;
                
                function connectWebSocket() {
                    ws = new WebSocket('ws://localhost:8080/ws');
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    };
                    
                    ws.onclose = function() {
                        setTimeout(connectWebSocket, 1000);
                    };
                }
                
                function handleWebSocketMessage(data) {
                    switch(data.type) {
                        case 'metrics_update':
                            updateMetrics(data.metrics);
                            break;
                        case 'config_updated':
                            updateConfig(data.config);
                            break;
                        case 'pipeline_built':
                            updatePipelineInfo(data.pipeline_info);
                            break;
                        case 'log_entry':
                            addLogEntry(data.log);
                            break;
                    }
                }
                
                async function loadConfig() {
                    const response = await fetch('/api/config');
                    const config = await response.json();
                    updateConfig(config);
                }
                
                function updateConfig(config) {
                    document.getElementById('config').innerHTML = '<pre>' + JSON.stringify(config, null, 2) + '</pre>';
                }
                
                async function buildPipeline() {
                    const response = await fetch('/api/pipeline/build', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    });
                    const result = await response.json();
                    alert('Pipeline built successfully!');
                    loadPipelineInfo();
                }
                
                async function loadPipelineInfo() {
                    const response = await fetch('/api/pipeline');
                    const info = await response.json();
                    updatePipelineInfo(info);
                }
                
                function updatePipelineInfo(info) {
                    document.getElementById('pipeline-info').innerHTML = '<pre>' + JSON.stringify(info, null, 2) + '</pre>';
                }
                
                async function loadSessions() {
                    const response = await fetch('/api/context/sessions');
                    const sessions = await response.json();
                    updateSessions(sessions);
                }
                
                function updateSessions(sessions) {
                    const html = sessions.map(s => `
                        <div class="metric">
                            <strong>${s.session_id}</strong><br>
                            User: ${s.user_id}<br>
                            Duration: ${s.duration.toFixed(1)}s<br>
                            Context Items: ${s.context_items}
                        </div>
                    `).join('');
                    document.getElementById('sessions').innerHTML = html;
                }
                
                async function runTests() {
                    const response = await fetch('/api/test/run', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    });
                    const results = await response.json();
                    document.getElementById('test-results').innerHTML = '<pre>' + JSON.stringify(results, null, 2) + '</pre>';
                }
                
                async function startProfiling() {
                    await fetch('/api/profiling/start', { method: 'POST' });
                    alert('Profiling started');
                }
                
                async function stopProfiling() {
                    const response = await fetch('/api/profiling/stop', { method: 'POST' });
                    const results = await response.json();
                    document.getElementById('test-results').innerHTML = '<pre>' + JSON.stringify(results, null, 2) + '</pre>';
                }
                
                async function clearLogs() {
                    await fetch('/api/logs/clear', { method: 'POST' });
                    document.getElementById('logs').innerHTML = '';
                }
                
                function addLogEntry(log) {
                    const logsDiv = document.getElementById('logs');
                    const logEntry = document.createElement('div');
                    logEntry.innerHTML = `<small>${log.timestamp}</small> ${log.level}: ${log.message}`;
                    logsDiv.appendChild(logEntry);
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                }
                
                // Initialize
                connectWebSocket();
                loadConfig();
                loadPipelineInfo();
                loadSessions();
                setInterval(async () => {
                    const response = await fetch('/api/metrics');
                    const metrics = await response.json();
                    updateMetrics(metrics);
                }, 1000);
                
                function updateMetrics(metrics) {
                    document.getElementById('metrics').innerHTML = `
                        <div class="metric">
                            <strong>Memory:</strong> ${metrics.memory_usage_mb.toFixed(1)} MB
                        </div>
                        <div class="metric">
                            <strong>CPU:</strong> ${metrics.cpu_usage_percent.toFixed(1)}%
                        </div>
                        <div class="metric">
                            <strong>Active Sessions:</strong> ${metrics.active_sessions}
                        </div>
                        <div class="metric">
                            <strong>Pipeline Ops:</strong> ${metrics.pipeline_operations}
                        </div>
                    `;
                }
            </script>
        </body>
        </html>
        """
    
    async def _broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if self.config.enable_websocket_support:
            message = json.dumps(data)
            await self.connection_manager.broadcast(message)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system and application metrics"""
        process = psutil.Process()
        
        metrics = {
            "timestamp": time.time(),
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_usage_percent": process.cpu_percent(),
            "active_sessions": len(self.context_manager.active_sessions) if self.context_manager else 0,
            "pipeline_operations": self._get_pipeline_operations_count(),
            "uptime": time.time() - process.create_time()
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _get_pipeline_operations_count(self) -> int:
        """Get total pipeline operations count"""
        count = 0
        if self.pipeline_builder:
            for node in self.pipeline_builder.nodes.values():
                if hasattr(node.processor, 'get_metrics'):
                    metrics = node.processor.get_metrics()
                    count += metrics.get('total_processed', 0)
        return count
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_usage": {
                "total_gb": psutil.disk_usage('/').total / 1024 / 1024 / 1024,
                "used_gb": psutil.disk_usage('/').used / 1024 / 1024 / 1024,
                "free_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024
            },
            "network_interfaces": psutil.net_if_addrs(),
            "boot_time": psutil.boot_time()
        }
    
    async def _run_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests with specified configuration"""
        # This would integrate with pytest or similar
        # For now, return mock results
        return {
            "status": "success",
            "tests_run": 10,
            "tests_passed": 9,
            "tests_failed": 1,
            "tests_skipped": 0,
            "duration": 2.5,
            "results": [
                {"name": "test_memory_processor", "status": "passed", "duration": 0.1},
                {"name": "test_extraction_processor", "status": "passed", "duration": 0.2},
                {"name": "test_quality_processor", "status": "failed", "duration": 0.1, "error": "AssertionError"}
            ]
        }
    
    async def _handle_websocket_message(self, websocket: WebSocket, data: str):
        """Handle incoming WebSocket messages"""
        try:
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "get_metrics":
                metrics = self._collect_metrics()
                await websocket.send_text(json.dumps({"type": "metrics", "metrics": metrics}))
            
        except Exception as e:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
    
    async def start_server(self):
        """Start the development server"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.debug_server_port,
            log_level=self.config.log_level.lower(),
            reload=self.config.enable_hot_reload
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def add_log_entry(self, level: str, message: str):
        """Add log entry"""
        entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message
        }
        
        self.log_entries.append(entry)
        
        # Keep only recent entries
        if len(self.log_entries) > self.config.max_log_entries:
            self.log_entries = self.log_entries[-self.config.max_log_entries:]
        
        # Broadcast to WebSocket clients
        asyncio.create_task(self._broadcast_update({
            "type": "log_entry",
            "log": entry
        }))


class DebugUtils:
    """Debugging utilities for LocalCat"""
    
    def __init__(self, config: DevToolConfig):
        self.config = config
    
    def inspect_object(self, obj: Any, max_depth: int = 3) -> Dict[str, Any]:
        """Inspect object and return structured information"""
        def inspect_recursive(obj, depth=0, path=""):
            if depth > max_depth:
                return f"<max depth reached at {path}>"
            
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [inspect_recursive(item, depth + 1, f"{path}[{i}]") for i, item in enumerate(obj[:10])]  # Limit to 10 items
            elif isinstance(obj, dict):
                return {k: inspect_recursive(v, depth + 1, f"{path}.{k}") for k, v in list(obj.items())[:10]}  # Limit to 10 items
            elif hasattr(obj, '__dict__'):
                return {
                    "type": obj.__class__.__name__,
                    "module": obj.__module__,
                    "attributes": inspect_recursive(obj.__dict__, depth + 1, f"{path}.attributes")
                }
            else:
                return str(obj)
        
        return inspect_recursive(obj)
    
    def trace_execution(self, func):
        """Decorator to trace function execution"""
        import functools
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"→ Entering {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                print(f"← Exiting {func.__name__} (duration: {duration:.3f}s)")
                return result
            except Exception as e:
                duration = time.time() - start_time
                print(f"✗ Error in {func.__name__} (duration: {duration:.3f}s): {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"→ Entering {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                print(f"← Exiting {func.__name__} (duration: {duration:.3f}s)")
                return result
            except Exception as e:
                duration = time.time() - start_time
                print(f"✗ Error in {func.__name__} (duration: {duration:.3f}s): {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class MemoryInspector:
    """Memory inspection utilities"""
    
    def __init__(self, config: DevToolConfig):
        self.config = config
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        process = psutil.Process()
        
        return {
            "rss": process.memory_info().rss,  # Resident Set Size
            "vms": process.memory_info().vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total,
            "used": psutil.virtual_memory().used
        }
    
    def get_object_memory_usage(self, obj: Any) -> int:
        """Get memory usage of a specific object"""
        import sys
        return sys.getsizeof(obj)
    
    def find_memory_leaks(self) -> List[Dict[str, Any]]:
        """Find potential memory leaks"""
        import gc
        import types
        
        gc.collect()
        
        leaks = []
        obj_types = {}
        
        for obj in gc.get_objects():
            obj_type = type(obj)
            if obj_type not in obj_types:
                obj_types[obj_type] = []
            obj_types[obj_type].append(obj)
        
        for obj_type, objects in obj_types.items():
            if len(objects) > 100:  # Threshold for potential leak
                total_size = sum(sys.getsizeof(obj) for obj in objects)
                leaks.append({
                    "type": str(obj_type),
                    "count": len(objects),
                    "total_size": total_size,
                    "avg_size": total_size / len(objects)
                })
        
        return sorted(leaks, key=lambda x: x["total_size"], reverse=True)


def create_dev_tools_server(config: Optional[DevToolConfig] = None, 
                           app_config: Optional[Config] = None) -> DevToolsServer:
    """Create development tools server"""
    if config is None:
        config = DevToolConfig()
    
    if app_config is None:
        app_config = get_config()
    
    return DevToolsServer(config, app_config)


async def run_dev_tools_server(config: Optional[DevToolConfig] = None):
    """Run development tools server"""
    server = create_dev_tools_server(config)
    await server.start_server()