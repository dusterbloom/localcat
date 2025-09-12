"""
Alerting System for LocalCat Server

Provides comprehensive alerting capabilities for system health and performance metrics.
Implements rule-based alerting, notification channels, and alert lifecycle management.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import loguru

logger = loguru.logger.bind(component="alerting_system")


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    description: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    duration: float = 0.0  # How long condition must persist
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if isinstance(self.triggered_at, float):
            self.triggered_at = datetime.fromtimestamp(self.triggered_at)


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    type: str  # "email", "webhook", "console", "log"
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class AlertingSystem:
    """
    Comprehensive alerting system for LocalCat monitoring.
    
    Features:
    - Rule-based alerting on metrics and health status
    - Multiple notification channels (email, webhook, console)
    - Alert lifecycle management (trigger, acknowledge, resolve)
    - Alert suppression and deduplication
    - Persistent alert storage and history
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "/Users/peppi/Dev/localcat/data/alerts.db"
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.running = False
        self.check_interval = 10.0  # seconds
        self.check_task: Optional[asyncio.Task] = None
        self.metrics_collector = None
        
        # Initialize database
        self._init_database()
        
        # Register default alert rules
        self._register_default_rules()
        
        # Register default notification channels
        self._register_default_channels()
    
    def _init_database(self):
        """Initialize SQLite database for alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL,
                    threshold REAL,
                    triggered_at REAL NOT NULL,
                    acknowledged_at REAL,
                    resolved_at REAL,
                    tags TEXT
                )
            ''')
            
            # Create alert rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    metric_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    severity TEXT NOT NULL,
                    duration REAL DEFAULT 0.0,
                    tags TEXT
                )
            ''')
            
            # Create notification channels table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_channels (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    config TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON alerts(triggered_at)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Alerting database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize alerting database: {e}")
    
    def _register_default_rules(self):
        """Register default alert rules"""
        
        # High CPU usage
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            description="CPU usage is consistently high",
            metric_name="system.cpu_percent",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=300.0,  # 5 minutes
            tags={"component": "system", "resource": "cpu"}
        ))
        
        # Critical CPU usage
        self.add_rule(AlertRule(
            name="critical_cpu_usage",
            description="CPU usage is critically high",
            metric_name="system.cpu_percent",
            condition="gt",
            threshold=95.0,
            severity=AlertSeverity.CRITICAL,
            duration=60.0,  # 1 minute
            tags={"component": "system", "resource": "cpu"}
        ))
        
        # High memory usage
        self.add_rule(AlertRule(
            name="high_memory_usage",
            description="Memory usage is high",
            metric_name="system.memory_percent",
            condition="gt",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            duration=300.0,
            tags={"component": "system", "resource": "memory"}
        ))
        
        # Low disk space
        self.add_rule(AlertRule(
            name="low_disk_space",
            description="Disk space is running low",
            metric_name="system.disk_percent",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.ERROR,
            duration=60.0,
            tags={"component": "system", "resource": "disk"}
        ))
        
        # Service health failure
        self.add_rule(AlertRule(
            name="service_unhealthy",
            description="Critical service is unhealthy",
            metric_name="service.health",
            condition="lt",
            threshold=1.0,  # 1 = healthy, 0 = unhealthy
            severity=AlertSeverity.ERROR,
            duration=60.0,
            tags={"component": "service", "health": "unhealthy"}
        ))
        
        # Memory extraction latency
        self.add_rule(AlertRule(
            name="slow_extraction",
            description="Memory extraction is slow",
            metric_name="perf.extraction_latency",
            condition="gt",
            threshold=500.0,  # 500ms
            severity=AlertSeverity.WARNING,
            duration=180.0,  # 3 minutes
            tags={"component": "performance", "operation": "extraction"}
        ))
        
        # TTS latency
        self.add_rule(AlertRule(
            name="slow_tts",
            description="TTS processing is slow",
            metric_name="perf.tts_latency",
            condition="gt",
            threshold=1000.0,  # 1 second
            severity=AlertSeverity.WARNING,
            duration=120.0,  # 2 minutes
            tags={"component": "performance", "operation": "tts"}
        ))
    
    def _register_default_channels(self):
        """Register default notification channels"""
        
        # Console channel (always available)
        self.add_channel(NotificationChannel(
            name="console",
            type="console",
            enabled=True,
            config={}
        ))
        
        # Log channel
        self.add_channel(NotificationChannel(
            name="log",
            type="log",
            enabled=True,
            config={}
        ))
        
        # Email channel (if configured)
        email_enabled = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() in ("1", "true", "yes")
        if email_enabled:
            self.add_channel(NotificationChannel(
                name="email",
                type="email",
                enabled=True,
                config={
                    "smtp_server": os.getenv("ALERT_SMTP_SERVER", "localhost"),
                    "smtp_port": int(os.getenv("ALERT_SMTP_PORT", "587")),
                    "username": os.getenv("ALERT_EMAIL_USERNAME", ""),
                    "password": os.getenv("ALERT_EMAIL_PASSWORD", ""),
                    "from_address": os.getenv("ALERT_EMAIL_FROM", "localcat@localhost"),
                    "to_addresses": os.getenv("ALERT_EMAIL_TO", "admin@localhost").split(","),
                    "use_tls": os.getenv("ALERT_EMAIL_TLS", "true").lower() in ("1", "true", "yes")
                }
            ))
        
        # Webhook channel (if configured)
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            self.add_channel(NotificationChannel(
                name="webhook",
                type="webhook",
                enabled=True,
                config={
                    "url": webhook_url,
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"}
                }
            ))
    
    def set_metrics_collector(self, collector):
        """Set the metrics collector for alert evaluation"""
        self.metrics_collector = collector
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.name] = rule
        
        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO alert_rules 
                (name, description, metric_name, condition, threshold, severity, duration, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.name, rule.description, rule.metric_name, rule.condition,
                rule.threshold, rule.severity.value, rule.duration, json.dumps(rule.tags)
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert rule {rule.name}: {e}")
        
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel"""
        self.channels[channel.name] = channel
        
        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO notification_channels 
                (name, type, enabled, config)
                VALUES (?, ?, ?, ?)
            ''', (channel.name, channel.type, channel.enabled, json.dumps(channel.config)))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store notification channel {channel.name}: {e}")
        
        logger.info(f"Added notification channel: {channel.name}")
    
    async def start_monitoring(self):
        """Start alert monitoring"""
        if self.running:
            logger.warning("Alert monitoring already active")
            return
        
        self.running = True
        self.check_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Started alert monitoring")
    
    async def stop_monitoring(self):
        """Stop alert monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped alert monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for alert evaluation"""
        while self.running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        if not self.metrics_collector:
            return
        
        for rule_name, rule in self.rules.items():
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        try:
            # Get current metric value
            if rule.metric_name == "service.health":
                # Special case for service health
                value = 1.0 if await self._check_service_health() else 0.0
            else:
                # Get from metrics collector
                current_metrics = self.metrics_collector.get_current_metrics()
                value = current_metrics.get(rule.metric_name)
                
                if value is None:
                    return  # No data available
            
            # Check condition
            condition_met = self._check_condition(value, rule.condition, rule.threshold)
            
            # Get existing alert for this rule
            existing_alert = self.active_alerts.get(rule.name)
            
            if condition_met:
                if existing_alert is None:
                    # New alert
                    await self._trigger_alert(rule, value)
                else:
                    # Update existing alert
                    existing_alert.current_value = value
                    existing_alert.triggered_at = datetime.now()
                    
            elif existing_alert is not None:
                # Alert condition no longer met, resolve alert
                await self._resolve_alert(existing_alert)
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if a condition is met"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "ne":
            return value != threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    async def _check_service_health(self) -> bool:
        """Check overall service health"""
        try:
            # Import here to avoid circular dependency
            from components.monitoring.health_monitor import HealthMonitor
            
            # This is a simplified check - in practice, you'd get the monitor instance
            return True  # Placeholder
            
        except Exception:
            return False
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger a new alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=f"Alert: {rule.description}",
            message=f"{rule.description}. Current value: {current_value:.2f}, threshold: {rule.threshold}",
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            triggered_at=datetime.now(),
            tags=rule.tags
        )
        
        # Store alert
        # Use the rule's name as the key; fix undefined variable 'rule_name'
        self.active_alerts[rule.name] = alert
        await self._store_alert(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.title}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Update storage
        await self._update_alert(alert)
        
        # Remove from active alerts
        if alert.rule_name in self.active_alerts:
            del self.active_alerts[alert.rule_name]
        
        # Send resolution notification
        await self._send_notifications(alert, is_resolution=True)
        
        logger.info(f"Alert resolved: {alert.title}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts 
                (id, rule_name, severity, status, title, message, metric_name, current_value, threshold, triggered_at, acknowledged_at, resolved_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.rule_name, alert.severity.value, alert.status.value,
                alert.title, alert.message, alert.metric_name, alert.current_value,
                alert.threshold, alert.triggered_at.timestamp(),
                alert.acknowledged_at.timestamp() if alert.acknowledged_at else None,
                alert.resolved_at.timestamp() if alert.resolved_at else None,
                json.dumps(alert.tags)
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert {alert.id}: {e}")
    
    async def _update_alert(self, alert: Alert):
        """Update alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alerts SET 
                    status = ?, acknowledged_at = ?, resolved_at = ?
                WHERE id = ?
            ''', (
                alert.status.value,
                alert.acknowledged_at.timestamp() if alert.acknowledged_at else None,
                alert.resolved_at.timestamp() if alert.resolved_at else None,
                alert.id
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update alert {alert.id}: {e}")
    
    async def _send_notifications(self, alert: Alert, is_resolution: bool = False):
        """Send notifications for an alert"""
        for channel_name, channel in self.channels.items():
            if not channel.enabled:
                continue
            
            try:
                if channel.type == "console":
                    await self._send_console_notification(alert, channel, is_resolution)
                elif channel.type == "log":
                    await self._send_log_notification(alert, channel, is_resolution)
                elif channel.type == "email":
                    await self._send_email_notification(alert, channel, is_resolution)
                elif channel.type == "webhook":
                    await self._send_webhook_notification(alert, channel, is_resolution)
                    
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    async def _send_console_notification(self, alert: Alert, channel: NotificationChannel, is_resolution: bool):
        """Send console notification"""
        status = "RESOLVED" if is_resolution else "TRIGGERED"
        severity = alert.severity.value.upper()
        
        print(f"[{severity}] {status}: {alert.title}")
        print(f"  {alert.message}")
        print(f"  Metric: {alert.metric_name} = {alert.current_value:.2f}")
        print(f"  Time: {alert.triggered_at}")
        print("-" * 50)
    
    async def _send_log_notification(self, alert: Alert, channel: NotificationChannel, is_resolution: bool):
        """Send log notification"""
        level = "INFO" if is_resolution or alert.severity == AlertSeverity.INFO else "WARNING"
        if alert.severity == AlertSeverity.ERROR:
            level = "ERROR"
        elif alert.severity == AlertSeverity.CRITICAL:
            level = "CRITICAL"
        
        status = "resolved" if is_resolution else "triggered"
        logger.log(level, f"Alert {status}: {alert.title} - {alert.message}")
    
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel, is_resolution: bool):
        """Send email notification"""
        config = channel.config
        
        subject = f"[LocalCat] {alert.severity.value.upper()}: {alert.title}"
        if is_resolution:
            subject = f"[LocalCat] RESOLVED: {alert.title}"
        
        body = f"""
Alert {status}: {alert.title}

{alert.message}

Details:
- Metric: {alert.metric_name}
- Current Value: {alert.current_value:.2f}
- Threshold: {alert.threshold}
- Severity: {alert.severity.value}
- Triggered: {alert.triggered_at}

Tags: {alert.tags}

--
LocalCat Alerting System
        """
        
        msg = MIMEMultipart()
        msg['From'] = config['from_address']
        msg['To'] = ', '.join(config['to_addresses'])
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls', True):
                server.starttls()
            
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel, is_resolution: bool):
        """Send webhook notification"""
        config = channel.config
        
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "status": "resolved" if is_resolution else "active",
            "title": alert.title,
            "message": alert.message,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "triggered_at": alert.triggered_at.isoformat(),
            "tags": alert.tags
        }
        
        try:
            response = requests.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, rule_name, severity, status, title, message, metric_name, 
                       current_value, threshold, triggered_at, acknowledged_at, resolved_at, tags
                FROM alerts 
                ORDER BY triggered_at DESC 
                LIMIT ?
            ''', (limit,))
            
            alerts = []
            for row in cursor.fetchall():
                alert = Alert(
                    id=row[0],
                    rule_name=row[1],
                    severity=AlertSeverity(row[2]),
                    status=AlertStatus(row[3]),
                    title=row[4],
                    message=row[5],
                    metric_name=row[6],
                    current_value=row[7],
                    threshold=row[8],
                    triggered_at=row[9],
                    acknowledged_at=datetime.fromtimestamp(row[10]) if row[10] else None,
                    resolved_at=datetime.fromtimestamp(row[11]) if row[11] else None,
                    tags=json.loads(row[12]) if row[12] else {}
                )
                alerts.append(alert)
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert"""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                await self._update_alert(alert)
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
