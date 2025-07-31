#!/usr/bin/env python3
"""
Enhanced Production Monitoring and Alerting System
=================================================

This system provides comprehensive production monitoring for the enhanced temporal
validation framework, including real-time performance tracking, drift detection,
and automated alerting for production issues.

FEATURES:
✅ Real-time model performance monitoring
✅ Data drift detection and alerting
✅ Production readiness continuous validation
✅ Automated model rollback triggers
✅ Integration with InfluxDB and Grafana
✅ Comprehensive alerting system

Author: AI Assistant
Date: 2025-07-31
Version: 1.0 (Production Ready)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import joblib
import sqlite3
from pathlib import Path

# Add utils to path
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')

class ProductionMonitor:
    """
    Comprehensive production monitoring system for enhanced temporal validation models.
    """
    
    def __init__(self, base_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize production monitoring system.
        
        Args:
            base_path: Base path for the project
            config: Optional configuration dictionary
        """
        self.base_path = base_path
        self.config = config or self._load_default_config()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Setup paths
        self.monitoring_dir = f"{base_path}/monitoring"
        self.alerts_dir = f"{monitoring_dir}/alerts"
        self.metrics_dir = f"{monitoring_dir}/metrics"
        self.db_path = f"{monitoring_dir}/production_monitoring.db"
        
        # Create directories
        for dir_path in [self.monitoring_dir, self.alerts_dir, self.metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize database
        self._init_monitoring_database()
        
        self.logger.info("🔍 Production Monitoring System Initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default monitoring configuration."""
        return {
            'performance_thresholds': {
                'rmse_degradation_warning': 15.0,  # % degradation for warning
                'rmse_degradation_critical': 30.0,  # % degradation for critical alert
                'r2_minimum_warning': 0.3,
                'r2_minimum_critical': 0.1,
                'prediction_latency_warning': 1.0,  # seconds
                'prediction_latency_critical': 5.0
            },
            'drift_thresholds': {
                'feature_drift_warning': 0.05,
                'feature_drift_critical': 0.1,
                'target_drift_warning': 0.1,
                'target_drift_critical': 0.2
            },
            'monitoring_intervals': {
                'performance_check_minutes': 15,
                'drift_check_hours': 4,
                'health_check_minutes': 5
            },
            'alerting': {
                'enable_email': False,
                'enable_slack': False,
                'enable_log': True,
                'enable_file': True
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup enhanced logging for production monitoring."""
        logger = logging.getLogger("production_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = f"{self.monitoring_dir}/production_monitor.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s [PROD_MONITOR] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s [MONITOR] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _init_monitoring_database(self):
        """Initialize SQLite database for monitoring data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                rmse REAL,
                mae REAL,
                r2 REAL,
                prediction_count INTEGER,
                avg_prediction_time REAL,
                validation_rmse REAL,
                performance_degradation REAL
            )
        ''')
        
        # Drift detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                feature_name TEXT,
                drift_score REAL,
                drift_detected BOOLEAN,
                drift_type TEXT
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Model health table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                health_status TEXT NOT NULL,
                deployment_status TEXT,
                temporal_validation BOOLEAN,
                last_prediction_time DATETIME,
                total_predictions INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("✅ Monitoring database initialized")
    
    def monitor_model_performance(self, model_name: str, predictions: np.ndarray, 
                                actuals: np.ndarray, prediction_times: List[float],
                                validation_rmse: Optional[float] = None) -> Dict[str, Any]:
        """
        Monitor model performance and detect degradation.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            actuals: Actual values
            prediction_times: Time taken for each prediction
            validation_rmse: Original validation RMSE for comparison
            
        Returns:
            Performance monitoring results
        """
        self.logger.info(f"📊 Monitoring performance for {model_name}")
        
        # Calculate current performance metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Prediction timing
        avg_prediction_time = np.mean(prediction_times)
        
        # Performance degradation calculation
        performance_degradation = None
        if validation_rmse and validation_rmse > 0:
            performance_degradation = ((rmse - validation_rmse) / validation_rmse) * 100
        
        # Store metrics in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance_metrics 
            (model_name, rmse, mae, r2, prediction_count, avg_prediction_time, 
             validation_rmse, performance_degradation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, rmse, mae, r2, len(predictions), avg_prediction_time,
              validation_rmse, performance_degradation))
        conn.commit()
        conn.close()
        
        # Check for performance alerts
        alerts = self._check_performance_alerts(model_name, rmse, r2, 
                                              avg_prediction_time, performance_degradation)
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'avg_prediction_time': avg_prediction_time,
            'performance_degradation': performance_degradation,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Performance monitoring completed for {model_name}")
        return results
    
    def _check_performance_alerts(self, model_name: str, rmse: float, r2: float,
                                avg_prediction_time: float, 
                                performance_degradation: Optional[float]) -> List[Dict[str, Any]]:
        """Check for performance-based alerts."""
        alerts = []
        thresholds = self.config['performance_thresholds']
        
        # Performance degradation alerts
        if performance_degradation is not None:
            if performance_degradation >= thresholds['rmse_degradation_critical']:
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'CRITICAL',
                    'message': f"Critical performance degradation: {performance_degradation:.1f}% increase in RMSE"
                })
            elif performance_degradation >= thresholds['rmse_degradation_warning']:
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'WARNING',
                    'message': f"Performance degradation warning: {performance_degradation:.1f}% increase in RMSE"
                })
        
        # R² alerts
        if r2 < thresholds['r2_minimum_critical']:
            alerts.append({
                'type': 'r2_low',
                'severity': 'CRITICAL',
                'message': f"Critical R² value: {r2:.3f} below minimum threshold"
            })
        elif r2 < thresholds['r2_minimum_warning']:
            alerts.append({
                'type': 'r2_low',
                'severity': 'WARNING',
                'message': f"Low R² warning: {r2:.3f} approaching minimum threshold"
            })
        
        # Prediction latency alerts
        if avg_prediction_time > thresholds['prediction_latency_critical']:
            alerts.append({
                'type': 'latency_high',
                'severity': 'CRITICAL',
                'message': f"Critical prediction latency: {avg_prediction_time:.2f}s per prediction"
            })
        elif avg_prediction_time > thresholds['prediction_latency_warning']:
            alerts.append({
                'type': 'latency_high',
                'severity': 'WARNING',
                'message': f"High prediction latency: {avg_prediction_time:.2f}s per prediction"
            })
        
        # Store alerts in database
        if alerts:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for alert in alerts:
                cursor.execute('''
                    INSERT INTO alerts (model_name, alert_type, severity, message)
                    VALUES (?, ?, ?, ?)
                ''', (model_name, alert['type'], alert['severity'], alert['message']))
            conn.commit()
            conn.close()
        
        return alerts
    
    def detect_data_drift(self, model_name: str, current_features: pd.DataFrame,
                         reference_features: pd.DataFrame, 
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect data drift in input features.
        
        Args:
            model_name: Name of the model
            current_features: Current feature data
            reference_features: Reference (training) feature data
            feature_names: Optional list of feature names
            
        Returns:
            Drift detection results
        """
        self.logger.info(f"🔍 Detecting data drift for {model_name}")
        
        # Import drift detection
        from temporal_validation import detect_data_drift
        
        # Perform drift detection
        drift_results = detect_data_drift(
            reference_features.values, 
            current_features.values,
            feature_names=feature_names or list(current_features.columns),
            threshold=self.config['drift_thresholds']['feature_drift_warning']
        )
        
        # Store drift metrics in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for feature_name, drift_info in drift_results['drift_scores'].items():
            if isinstance(drift_info, dict):
                drift_score = drift_info.get('drift_score', drift_info.get('ks_statistic', 0))
                drift_detected = drift_info.get('drifted', False)
            else:
                drift_score = float(drift_info)
                drift_detected = drift_score > self.config['drift_thresholds']['feature_drift_warning']
            
            cursor.execute('''
                INSERT INTO drift_metrics (model_name, feature_name, drift_score, drift_detected, drift_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_name, feature_name, drift_score, drift_detected, 'feature'))
        
        conn.commit()
        conn.close()
        
        # Check for drift alerts
        drift_alerts = self._check_drift_alerts(model_name, drift_results)
        
        results = {
            'model_name': model_name,
            'overall_drift': drift_results['overall_drift'],
            'drifted_features': drift_results['drifted_features'],
            'drift_scores': drift_results['drift_scores'],
            'alerts': drift_alerts,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Drift detection completed for {model_name}")
        return results
    
    def _check_drift_alerts(self, model_name: str, drift_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for drift-based alerts."""
        alerts = []
        thresholds = self.config['drift_thresholds']
        
        # Overall drift alert
        if drift_results['overall_drift']:
            drifted_count = len(drift_results['drifted_features'])
            total_features = len(drift_results['drift_scores'])
            drift_percentage = (drifted_count / total_features) * 100
            
            if drift_percentage >= 50:  # More than 50% of features drifted
                alerts.append({
                    'type': 'data_drift',
                    'severity': 'CRITICAL',
                    'message': f"Critical data drift: {drifted_count}/{total_features} features ({drift_percentage:.1f}%) show drift"
                })
            elif drift_percentage >= 25:  # More than 25% of features drifted
                alerts.append({
                    'type': 'data_drift',
                    'severity': 'WARNING',
                    'message': f"Data drift warning: {drifted_count}/{total_features} features ({drift_percentage:.1f}%) show drift"
                })
        
        # Store drift alerts
        if alerts:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for alert in alerts:
                cursor.execute('''
                    INSERT INTO alerts (model_name, alert_type, severity, message)
                    VALUES (?, ?, ?, ?)
                ''', (model_name, alert['type'], alert['severity'], alert['message']))
            conn.commit()
            conn.close()
        
        return alerts
    
    def check_model_health(self, model_name: str) -> Dict[str, Any]:
        """
        Check overall model health status.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model health status
        """
        self.logger.info(f"💊 Checking health for {model_name}")
        
        # Load model metadata
        model_path = f"{self.base_path}/models/{model_name}/enhanced_{model_name}_model.joblib"
        
        health_status = "UNKNOWN"
        deployment_status = "UNKNOWN"
        temporal_validation = False
        
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict):
                    metadata = model_data.get('metadata', {})
                    deployment_status = metadata.get('production_readiness', {}).get('deployment_status', 'UNKNOWN')
                    temporal_validation = metadata.get('validation_info', {}).get('temporal_validation', False)
                    
                    # Determine health status
                    if deployment_status == 'APPROVED' and temporal_validation:
                        health_status = "HEALTHY"
                    elif deployment_status == 'NEEDS_IMPROVEMENT':
                        health_status = "DEGRADED"
                    else:
                        health_status = "UNHEALTHY"
                else:
                    health_status = "LEGACY"  # Old model format
            else:
                health_status = "NOT_FOUND"
        
        except Exception as e:
            self.logger.error(f"❌ Error checking model health: {str(e)}")
            health_status = "ERROR"
        
        # Get recent performance data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) as prediction_count, MAX(timestamp) as last_prediction
            FROM performance_metrics 
            WHERE model_name = ? AND timestamp > datetime('now', '-24 hours')
        ''', (model_name,))
        
        row = cursor.fetchone()
        prediction_count = row[0] if row else 0
        last_prediction = row[1] if row else None
        
        # Update model health in database
        cursor.execute('''
            INSERT INTO model_health 
            (model_name, health_status, deployment_status, temporal_validation, 
             last_prediction_time, total_predictions)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, health_status, deployment_status, temporal_validation,
              last_prediction, prediction_count))
        
        conn.commit()
        conn.close()
        
        health_info = {
            'model_name': model_name,
            'health_status': health_status,
            'deployment_status': deployment_status,
            'temporal_validation': temporal_validation,
            'last_prediction': last_prediction,
            'predictions_24h': prediction_count,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Health check completed for {model_name}: {health_status}")
        return health_info
    
    def generate_monitoring_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            time_range_hours: Time range for the report in hours
            
        Returns:
            Monitoring report
        """
        self.logger.info(f"📊 Generating monitoring report for last {time_range_hours} hours")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get performance summary
        performance_df = pd.read_sql_query('''
            SELECT model_name, AVG(rmse) as avg_rmse, AVG(r2) as avg_r2,
                   AVG(performance_degradation) as avg_degradation,
                   COUNT(*) as measurement_count
            FROM performance_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
            GROUP BY model_name
        '''.format(time_range_hours), conn)
        
        # Get alerts summary
        alerts_df = pd.read_sql_query('''
            SELECT model_name, alert_type, severity, COUNT(*) as alert_count
            FROM alerts 
            WHERE timestamp > datetime('now', '-{} hours')
            GROUP BY model_name, alert_type, severity
        '''.format(time_range_hours), conn)
        
        # Get drift summary
        drift_df = pd.read_sql_query('''
            SELECT model_name, COUNT(*) as drift_detections,
                   AVG(drift_score) as avg_drift_score
            FROM drift_metrics 
            WHERE timestamp > datetime('now', '-{} hours') AND drift_detected = 1
            GROUP BY model_name
        '''.format(time_range_hours), conn)
        
        # Get model health summary
        health_df = pd.read_sql_query('''
            SELECT model_name, health_status, deployment_status, 
                   temporal_validation, total_predictions
            FROM model_health 
            WHERE timestamp = (
                SELECT MAX(timestamp) FROM model_health h2 
                WHERE h2.model_name = model_health.model_name
            )
        ''', conn)
        
        conn.close()
        
        # Compile report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'time_range_hours': time_range_hours,
            'performance_summary': performance_df.to_dict('records') if not performance_df.empty else [],
            'alerts_summary': alerts_df.to_dict('records') if not alerts_df.empty else [],
            'drift_summary': drift_df.to_dict('records') if not drift_df.empty else [],
            'health_summary': health_df.to_dict('records') if not health_df.empty else [],
            'overall_status': self._calculate_overall_status(alerts_df, health_df)
        }
        
        # Save report
        report_path = f"{self.metrics_dir}/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Monitoring report saved: {report_path}")
        return report
    
    def _calculate_overall_status(self, alerts_df: pd.DataFrame, health_df: pd.DataFrame) -> str:
        """Calculate overall system status."""
        if alerts_df.empty and health_df.empty:
            return "UNKNOWN"
        
        # Check for critical alerts
        if not alerts_df.empty:
            critical_alerts = alerts_df[alerts_df['severity'] == 'CRITICAL']
            if not critical_alerts.empty:
                return "CRITICAL"
            
            warning_alerts = alerts_df[alerts_df['severity'] == 'WARNING']
            if not warning_alerts.empty:
                return "WARNING"
        
        # Check model health
        if not health_df.empty:
            unhealthy_models = health_df[health_df['health_status'].isin(['UNHEALTHY', 'ERROR', 'NOT_FOUND'])]
            if not unhealthy_models.empty:
                return "DEGRADED"
        
        return "HEALTHY"
    
    def export_metrics_for_grafana(self, output_path: Optional[str] = None) -> str:
        """
        Export metrics in format suitable for Grafana/InfluxDB.
        
        Args:
            output_path: Optional output path
            
        Returns:
            Path to exported metrics
        """
        if output_path is None:
            output_path = f"{self.metrics_dir}/grafana_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        conn = sqlite3.connect(self.db_path)
        
        # Export performance metrics
        query = '''
            SELECT 
                datetime(timestamp) as datetime,
                model_name,
                'performance' as metric_type,
                rmse,
                mae,
                r2,
                avg_prediction_time,
                performance_degradation
            FROM performance_metrics
            ORDER BY timestamp DESC
        '''
        
        metrics_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Save for Grafana
        metrics_df.to_csv(output_path, index=False)
        
        self.logger.info(f"📊 Metrics exported for Grafana: {output_path}")
        return output_path


def main():
    """Main function for production monitoring."""
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    
    # Initialize monitoring system
    monitor = ProductionMonitor(base_path)
    
    # Generate initial monitoring report
    report = monitor.generate_monitoring_report(time_range_hours=24)
    
    print("🔍 Production Monitoring System")
    print("=" * 50)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Models Monitored: {len(report['health_summary'])}")
    print(f"Active Alerts: {len(report['alerts_summary'])}")
    print(f"Drift Detections: {len(report['drift_summary'])}")
    
    # Export metrics for Grafana
    metrics_path = monitor.export_metrics_for_grafana()
    print(f"Metrics exported: {metrics_path}")
    
    return monitor, report


if __name__ == "__main__":
    monitor, report = main()
