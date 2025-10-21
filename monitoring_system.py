#!/usr/bin/env python3
"""
Comprehensive monitoring system for the Plant Diagnostic System.
Includes logging, metrics collection, error tracking, and health monitoring.
"""

import logging
import time
import json
import psutil
import torch
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_reserved: Optional[float] = None
    gpu_utilization: Optional[float] = None
    active_connections: int = 0
    inference_count: int = 0
    error_count: int = 0


@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    timestamp: float
    inference_time: float
    model_type: str
    input_size: tuple
    confidence: float
    success: bool
    error_message: Optional[str] = None


class MetricsCollector:
    """Collects and stores system and inference metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.system_metrics: List[SystemMetrics] = []
        self.inference_metrics: List[InferenceMetrics] = []
        self.lock = threading.Lock()
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics."""
        with self.lock:
            self.system_metrics.append(metrics)
            if len(self.system_metrics) > self.max_metrics:
                self.system_metrics.pop(0)
    
    def add_inference_metrics(self, metrics: InferenceMetrics):
        """Add inference metrics."""
        with self.lock:
            self.inference_metrics.append(metrics)
            if len(self.inference_metrics) > self.max_metrics:
                self.inference_metrics.pop(0)
    
    def get_recent_metrics(self, minutes: int = 60) -> Dict[str, List]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            recent_system = [
                m for m in self.system_metrics if m.timestamp >= cutoff_time
            ]
            recent_inference = [
                m for m in self.inference_metrics if m.timestamp >= cutoff_time
            ]
        
        return {
            'system': recent_system,
            'inference': recent_inference
        }
    
    def get_summary_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for the last N minutes."""
        recent_metrics = self.get_recent_metrics(minutes)
        
        if not recent_metrics['system']:
            return {}
        
        system_metrics = recent_metrics['system']
        inference_metrics = recent_metrics['inference']
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in system_metrics) / len(system_metrics)
        avg_memory = sum(m.memory_percent for m in system_metrics) / len(system_metrics)
        avg_disk = sum(m.disk_percent for m in system_metrics) / len(system_metrics)
        
        # Calculate inference stats
        if inference_metrics:
            avg_inference_time = sum(m.inference_time for m in inference_metrics) / len(inference_metrics)
            success_rate = sum(1 for m in inference_metrics if m.success) / len(inference_metrics)
            total_inferences = len(inference_metrics)
        else:
            avg_inference_time = 0
            success_rate = 0
            total_inferences = 0
        
        return {
            'time_range_minutes': minutes,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_disk_percent': avg_disk,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'inference_success_rate': success_rate,
            'total_inferences': total_inferences,
            'gpu_available': torch.cuda.is_available()
        }


class HealthMonitor:
    """Monitors system health and alerts on issues."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0,
            'inference_success_rate': 0.95,
            'avg_inference_time_ms': 5000.0
        }
        self.alert_history = []
        self.alert_cooldown = 300  # 5 minutes between alerts
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health and return status."""
        recent_metrics = self.metrics_collector.get_recent_metrics(5)  # Last 5 minutes
        
        if not recent_metrics['system']:
            return {'status': 'unknown', 'issues': ['No recent metrics available']}
        
        latest_metrics = recent_metrics['system'][-1]
        issues = []
        
        # Check CPU usage
        if latest_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            issues.append(f"High CPU usage: {latest_metrics.cpu_percent:.1f}%")
        
        # Check memory usage
        if latest_metrics.memory_percent > self.alert_thresholds['memory_percent']:
            issues.append(f"High memory usage: {latest_metrics.memory_percent:.1f}%")
        
        # Check disk usage
        if latest_metrics.disk_percent > self.alert_thresholds['disk_percent']:
            issues.append(f"High disk usage: {latest_metrics.disk_percent:.1f}%")
        
        # Check GPU memory if available
        if latest_metrics.gpu_memory_allocated is not None:
            gpu_percent = (latest_metrics.gpu_memory_allocated / 
                          (latest_metrics.gpu_memory_reserved or 1)) * 100
            if gpu_percent > self.alert_thresholds['gpu_memory_percent']:
                issues.append(f"High GPU memory usage: {gpu_percent:.1f}%")
        
        # Check inference performance
        if recent_metrics['inference']:
            success_rate = sum(1 for m in recent_metrics['inference'] if m.success) / len(recent_metrics['inference'])
            if success_rate < self.alert_thresholds['inference_success_rate']:
                issues.append(f"Low inference success rate: {success_rate:.2%}")
            
            avg_time = sum(m.inference_time for m in recent_metrics['inference']) / len(recent_metrics['inference'])
            if avg_time * 1000 > self.alert_thresholds['avg_inference_time_ms']:
                issues.append(f"Slow inference: {avg_time*1000:.1f}ms average")
        
        status = 'healthy' if not issues else 'unhealthy'
        
        return {
            'status': status,
            'issues': issues,
            'timestamp': time.time(),
            'metrics': latest_metrics
        }
    
    def should_send_alert(self, issue: str) -> bool:
        """Check if alert should be sent (cooldown logic)."""
        now = time.time()
        cutoff = now - self.alert_cooldown
        
        # Remove old alerts
        self.alert_history = [alert for alert in self.alert_history if alert['timestamp'] > cutoff]
        
        # Check if this issue was recently alerted
        for alert in self.alert_history:
            if alert['issue'] == issue:
                return False
        
        return True
    
    def add_alert(self, issue: str, severity: str = 'warning'):
        """Add alert to history."""
        self.alert_history.append({
            'issue': issue,
            'severity': severity,
            'timestamp': time.time()
        })


class LoggingSystem:
    """Enhanced logging system with structured logging and log rotation."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_loggers()
    
    def setup_loggers(self):
        """Setup application loggers."""
        # Main application logger
        self.app_logger = self._create_logger(
            'plant_diagnostic',
            self.log_dir / 'app.log',
            level=logging.INFO
        )
        
        # Error logger
        self.error_logger = self._create_logger(
            'plant_diagnostic_errors',
            self.log_dir / 'errors.log',
            level=logging.ERROR
        )
        
        # Performance logger
        self.perf_logger = self._create_logger(
            'plant_diagnostic_performance',
            self.log_dir / 'performance.log',
            level=logging.INFO
        )
        
        # Inference logger
        self.inference_logger = self._create_logger(
            'plant_diagnostic_inference',
            self.log_dir / 'inference.log',
            level=logging.INFO
        )
    
    def _create_logger(self, name: str, log_file: Path, level: int) -> logging.Logger:
        """Create a logger with file rotation."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_inference(self, model_type: str, inference_time: float, 
                     success: bool, confidence: float, error: str = None):
        """Log inference details."""
        log_data = {
            'model_type': model_type,
            'inference_time': inference_time,
            'success': success,
            'confidence': confidence,
            'error': error,
            'timestamp': time.time()
        }
        
        if success:
            self.inference_logger.info(f"Inference successful: {json.dumps(log_data)}")
        else:
            self.inference_logger.error(f"Inference failed: {json.dumps(log_data)}")
    
    def log_performance(self, metrics: SystemMetrics):
        """Log performance metrics."""
        self.perf_logger.info(f"Performance metrics: {json.dumps(asdict(metrics))}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        self.error_logger.error(f"Error in {context}: {str(error)}", exc_info=True)


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, email_config: Optional[Dict] = None):
        self.email_config = email_config
        self.alert_queue = queue.Queue()
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
    
    def send_alert(self, message: str, severity: str = 'warning'):
        """Send alert notification."""
        alert = {
            'message': message,
            'severity': severity,
            'timestamp': time.time()
        }
        self.alert_queue.put(alert)
    
    def _process_alerts(self):
        """Process alert queue."""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_notification(alert)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def _send_notification(self, alert: Dict):
        """Send notification via configured channels."""
        if self.email_config:
            self._send_email(alert)
        
        # Log alert
        logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")
    
    def _send_email(self, alert: Dict):
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"Plant Diagnostic System Alert - {alert['severity'].upper()}"
            
            body = f"""
            Alert: {alert['message']}
            Severity: {alert['severity']}
            Time: {datetime.fromtimestamp(alert['timestamp'])}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class MonitoringSystem:
    """Main monitoring system that coordinates all monitoring components."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.logging_system = LoggingSystem()
        self.alert_manager = AlertManager(self.config.get('email'))
        
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self, interval: int = 60):
        """Start monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_collector.add_system_metrics(metrics)
                self.logging_system.log_performance(metrics)
                
                # Check health
                health_status = self.health_monitor.check_health()
                if health_status['status'] == 'unhealthy':
                    for issue in health_status['issues']:
                        if self.health_monitor.should_send_alert(issue):
                            self.alert_manager.send_alert(issue, 'warning')
                            self.health_monitor.add_alert(issue)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get GPU metrics if available
        gpu_memory_allocated = None
        gpu_memory_reserved = None
        gpu_utilization = None
        
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            gpu_utilization = torch.cuda.utilization()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            gpu_memory_allocated=gpu_memory_allocated,
            gpu_memory_reserved=gpu_memory_reserved,
            gpu_utilization=gpu_utilization
        )
    
    def log_inference(self, model_type: str, inference_time: float, 
                     success: bool, confidence: float, error: str = None):
        """Log inference metrics."""
        # Create inference metrics
        metrics = InferenceMetrics(
            timestamp=time.time(),
            inference_time=inference_time,
            model_type=model_type,
            input_size=(0, 0),  # Would be set by caller
            confidence=confidence,
            success=success,
            error_message=error
        )
        
        # Add to collector
        self.metrics_collector.add_inference_metrics(metrics)
        
        # Log to file
        self.logging_system.log_inference(model_type, inference_time, success, confidence, error)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        health_status = self.health_monitor.check_health()
        summary_stats = self.metrics_collector.get_summary_stats(60)  # Last hour
        
        return {
            'health': health_status,
            'performance': summary_stats,
            'monitoring_active': self.monitoring_active,
            'timestamp': time.time()
        }


# Global monitoring instance
monitoring_system = None

def initialize_monitoring(config: Optional[Dict] = None):
    """Initialize global monitoring system."""
    global monitoring_system
    monitoring_system = MonitoringSystem(config)
    return monitoring_system

def get_monitoring_system() -> Optional[MonitoringSystem]:
    """Get global monitoring system instance."""
    return monitoring_system


if __name__ == "__main__":
    # Example usage
    config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-password',
            'from': 'your-email@gmail.com',
            'to': 'admin@yourcompany.com'
        }
    }
    
    # Initialize monitoring
    monitoring = initialize_monitoring(config)
    
    # Start monitoring
    monitoring.start_monitoring(interval=30)  # Check every 30 seconds
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitoring.stop_monitoring()
        print("Monitoring stopped")

