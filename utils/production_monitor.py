#!/usr/bin/env python3
"""
Critical Fix: Production Model Monitoring and Alerting System
This implements comprehensive monitoring to detect production failures before they occur.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

class ProductionMonitor:
    """
    Comprehensive production monitoring for ML models.
    Detects data drift, performance degradation, and system anomalies.
    """
    
    def __init__(self, baseline_window_days=30, alert_threshold=0.15):
        self.baseline_window_days = baseline_window_days
        self.alert_threshold = alert_threshold
        self.baseline_stats = {}
        self.drift_history = []
        self.performance_history = []
        
    def establish_baseline(self, X_baseline: np.ndarray, y_baseline: np.ndarray, 
                          predictions_baseline: np.ndarray, feature_names: List[str] = None):
        """
        Establish baseline statistics for monitoring.
        
        Args:
            X_baseline: Baseline feature matrix
            y_baseline: Baseline target values
            predictions_baseline: Baseline model predictions
            feature_names: Names of features
        """
        self.baseline_stats = {
            'timestamp': datetime.now(),
            'sample_size': len(X_baseline),
            'feature_stats': self._calculate_feature_statistics(X_baseline, feature_names),
            'target_stats': self._calculate_target_statistics(y_baseline),
            'performance_stats': self._calculate_performance_statistics(y_baseline, predictions_baseline),
            'feature_names': feature_names or [f'feature_{i}' for i in range(X_baseline.shape[1])]
        }
        
        logging.info(f"✅ Baseline established with {len(X_baseline)} samples")
        
    def _calculate_feature_statistics(self, X: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Calculate statistical properties of features."""
        feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        stats = {}
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            stats[feature_name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'q25': float(np.percentile(feature_data, 25)),
                'q50': float(np.percentile(feature_data, 50)),
                'q75': float(np.percentile(feature_data, 75)),
                'skewness': float(self._calculate_skewness(feature_data)),
                'kurtosis': float(self._calculate_kurtosis(feature_data))
            }
        return stats
        
    def _calculate_target_statistics(self, y: np.ndarray) -> Dict:
        """Calculate statistical properties of target variable."""
        return {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'q25': float(np.percentile(y, 25)),
            'q50': float(np.percentile(y, 50)),
            'q75': float(np.percentile(y, 75)),
            'distribution': self._analyze_distribution(y)
        }
        
    def _calculate_performance_statistics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate baseline performance statistics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        residuals = y_true - y_pred
        
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mape': float(np.mean(np.abs(residuals / np.maximum(y_true, 1e-8))) * 100),
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_skewness': float(self._calculate_skewness(residuals)),
            'r2': float(1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))
        }
        
    def monitor_production_data(self, X_current: np.ndarray, y_current: np.ndarray = None,
                               predictions_current: np.ndarray = None, 
                               timestamp: datetime = None) -> Dict:
        """
        Monitor current production data against baseline.
        
        Args:
            X_current: Current feature matrix
            y_current: Current target values (if available)
            predictions_current: Current model predictions
            timestamp: Timestamp of current data
            
        Returns:
            dict: Monitoring results with alerts
        """
        if not self.baseline_stats:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
            
        timestamp = timestamp or datetime.now()
        
        monitoring_results = {
            'timestamp': timestamp,
            'sample_size': len(X_current),
            'alerts': [],
            'drift_detected': False,
            'performance_degraded': False
        }
        
        # 1. Feature Drift Detection
        feature_drift = self._detect_feature_drift(X_current)
        monitoring_results['feature_drift'] = feature_drift
        
        if feature_drift['overall_drift_score'] > self.alert_threshold:
            monitoring_results['drift_detected'] = True
            monitoring_results['alerts'].append({
                'type': 'FEATURE_DRIFT',
                'severity': 'HIGH',
                'message': f"Feature drift detected (score: {feature_drift['overall_drift_score']:.3f})",
                'affected_features': feature_drift['drifted_features']
            })
            
        # 2. Target Distribution Drift (if available)
        if y_current is not None:
            target_drift = self._detect_target_drift(y_current)
            monitoring_results['target_drift'] = target_drift
            
            if target_drift['drift_score'] > self.alert_threshold:
                monitoring_results['alerts'].append({
                    'type': 'TARGET_DRIFT',
                    'severity': 'MEDIUM',
                    'message': f"Target distribution drift detected (score: {target_drift['drift_score']:.3f})"
                })
                
        # 3. Performance Monitoring (if available)
        if y_current is not None and predictions_current is not None:
            performance_change = self._monitor_performance(y_current, predictions_current)
            monitoring_results['performance_change'] = performance_change
            
            if performance_change['performance_degraded']:
                monitoring_results['performance_degraded'] = True
                monitoring_results['alerts'].append({
                    'type': 'PERFORMANCE_DEGRADATION',
                    'severity': 'CRITICAL',
                    'message': f"Model performance degraded: RMSE increased by {performance_change['rmse_change_pct']:.1f}%"
                })
                
        # 4. Data Quality Checks
        data_quality = self._check_data_quality(X_current)
        monitoring_results['data_quality'] = data_quality
        
        if data_quality['issues']:
            monitoring_results['alerts'].append({
                'type': 'DATA_QUALITY',
                'severity': 'HIGH',
                'message': f"Data quality issues detected: {', '.join(data_quality['issues'])}"
            })
            
        # 5. Volume Anomaly Detection
        volume_anomaly = self._detect_volume_anomaly(len(X_current), timestamp)
        monitoring_results['volume_anomaly'] = volume_anomaly
        
        if volume_anomaly['anomaly_detected']:
            monitoring_results['alerts'].append({
                'type': 'VOLUME_ANOMALY',
                'severity': 'MEDIUM',
                'message': f"Volume anomaly detected: {volume_anomaly['anomaly_type']}"
            })
            
        # Store monitoring history
        self._update_monitoring_history(monitoring_results)
        
        return monitoring_results
        
    def _detect_feature_drift(self, X_current: np.ndarray) -> Dict:
        """Detect drift in feature distributions."""
        from scipy import stats
        
        baseline_features = self.baseline_stats['feature_stats']
        feature_names = self.baseline_stats['feature_names']
        
        drift_results = {
            'drifted_features': [],
            'drift_scores': {},
            'overall_drift_score': 0.0
        }
        
        total_drift_score = 0.0
        
        for i, feature_name in enumerate(feature_names):
            current_feature = X_current[:, i]
            baseline_stats = baseline_features[feature_name]
            
            # Calculate multiple drift metrics
            
            # 1. Mean shift (normalized by baseline std)
            mean_shift = abs(np.mean(current_feature) - baseline_stats['mean']) / max(baseline_stats['std'], 1e-8)
            
            # 2. Standard deviation change
            std_change = abs(np.std(current_feature) - baseline_stats['std']) / max(baseline_stats['std'], 1e-8)
            
            # 3. Distribution shape change (using skewness and kurtosis)
            current_skewness = self._calculate_skewness(current_feature)
            current_kurtosis = self._calculate_kurtosis(current_feature)
            
            skewness_change = abs(current_skewness - baseline_stats['skewness'])
            kurtosis_change = abs(current_kurtosis - baseline_stats['kurtosis'])
            
            # Composite drift score
            drift_score = (mean_shift + std_change + skewness_change * 0.1 + kurtosis_change * 0.1) / 4
            
            drift_results['drift_scores'][feature_name] = {
                'score': float(drift_score),
                'mean_shift': float(mean_shift),
                'std_change': float(std_change),
                'skewness_change': float(skewness_change),
                'kurtosis_change': float(kurtosis_change)
            }
            
            if drift_score > self.alert_threshold:
                drift_results['drifted_features'].append(feature_name)
                
            total_drift_score += drift_score
            
        drift_results['overall_drift_score'] = total_drift_score / len(feature_names)
        
        return drift_results
        
    def _detect_target_drift(self, y_current: np.ndarray) -> Dict:
        """Detect drift in target distribution."""
        baseline_target = self.baseline_stats['target_stats']
        
        # Calculate current target statistics
        current_mean = np.mean(y_current)
        current_std = np.std(y_current)
        
        # Drift metrics
        mean_drift = abs(current_mean - baseline_target['mean']) / max(baseline_target['std'], 1e-8)
        std_drift = abs(current_std - baseline_target['std']) / max(baseline_target['std'], 1e-8)
        
        # Distribution shift using quantiles
        current_q25 = np.percentile(y_current, 25)
        current_q50 = np.percentile(y_current, 50)
        current_q75 = np.percentile(y_current, 75)
        
        q25_drift = abs(current_q25 - baseline_target['q25']) / max(baseline_target['std'], 1e-8)
        q50_drift = abs(current_q50 - baseline_target['q50']) / max(baseline_target['std'], 1e-8)
        q75_drift = abs(current_q75 - baseline_target['q75']) / max(baseline_target['std'], 1e-8)
        
        drift_score = (mean_drift + std_drift + q25_drift + q50_drift + q75_drift) / 5
        
        return {
            'drift_score': float(drift_score),
            'mean_drift': float(mean_drift),
            'std_drift': float(std_drift),
            'quantile_drifts': {
                'q25': float(q25_drift),
                'q50': float(q50_drift),
                'q75': float(q75_drift)
            }
        }
        
    def _monitor_performance(self, y_current: np.ndarray, predictions_current: np.ndarray) -> Dict:
        """Monitor model performance changes."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        baseline_perf = self.baseline_stats['performance_stats']
        
        # Calculate current performance
        current_rmse = np.sqrt(mean_squared_error(y_current, predictions_current))
        current_mae = mean_absolute_error(y_current, predictions_current)
        
        # Calculate performance changes
        rmse_change = current_rmse - baseline_perf['rmse']
        rmse_change_pct = (rmse_change / baseline_perf['rmse']) * 100
        
        mae_change = current_mae - baseline_perf['mae']
        mae_change_pct = (mae_change / baseline_perf['mae']) * 100
        
        # Determine if performance has degraded significantly
        performance_degraded = (rmse_change_pct > 15) or (mae_change_pct > 15)
        
        return {
            'current_rmse': float(current_rmse),
            'current_mae': float(current_mae),
            'rmse_change': float(rmse_change),
            'rmse_change_pct': float(rmse_change_pct),
            'mae_change': float(mae_change),
            'mae_change_pct': float(mae_change_pct),
            'performance_degraded': performance_degraded
        }
        
    def _check_data_quality(self, X_current: np.ndarray) -> Dict:
        """Check for data quality issues."""
        issues = []
        
        # Check for missing values
        missing_count = np.isnan(X_current).sum()
        if missing_count > 0:
            issues.append(f"Missing values detected: {missing_count}")
            
        # Check for infinite values
        inf_count = np.isinf(X_current).sum()
        if inf_count > 0:
            issues.append(f"Infinite values detected: {inf_count}")
            
        # Check for extreme outliers (beyond 5 standard deviations)
        feature_names = self.baseline_stats['feature_names']
        outlier_features = []
        
        for i, feature_name in enumerate(feature_names):
            baseline_stats = self.baseline_stats['feature_stats'][feature_name]
            current_feature = X_current[:, i]
            
            # Calculate z-scores based on baseline statistics
            z_scores = abs((current_feature - baseline_stats['mean']) / max(baseline_stats['std'], 1e-8))
            extreme_outliers = np.sum(z_scores > 5)
            
            if extreme_outliers > len(current_feature) * 0.01:  # More than 1% extreme outliers
                outlier_features.append(feature_name)
                
        if outlier_features:
            issues.append(f"Extreme outliers in features: {', '.join(outlier_features)}")
            
        return {
            'issues': issues,
            'missing_values': int(missing_count),
            'infinite_values': int(inf_count),
            'outlier_features': outlier_features
        }
        
    def _detect_volume_anomaly(self, current_volume: int, timestamp: datetime) -> Dict:
        """Detect anomalies in data volume."""
        # Simple volume anomaly detection based on time of day/week patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Expected volume patterns (this would be learned from historical data)
        expected_volume = self.baseline_stats['sample_size']
        
        # Simple heuristics for volume anomaly
        volume_ratio = current_volume / max(expected_volume, 1)
        
        anomaly_detected = False
        anomaly_type = "normal"
        
        if volume_ratio < 0.3:
            anomaly_detected = True
            anomaly_type = "critically_low_volume"
        elif volume_ratio < 0.5:
            anomaly_detected = True
            anomaly_type = "low_volume"
        elif volume_ratio > 3.0:
            anomaly_detected = True
            anomaly_type = "high_volume"
            
        return {
            'anomaly_detected': anomaly_detected,
            'anomaly_type': anomaly_type,
            'volume_ratio': float(volume_ratio),
            'current_volume': current_volume,
            'expected_volume': expected_volume
        }
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
        
    def _analyze_distribution(self, data: np.ndarray) -> Dict:
        """Analyze distribution characteristics of data."""
        return {
            'skewness': float(self._calculate_skewness(data)),
            'kurtosis': float(self._calculate_kurtosis(data)),
            'is_normal': abs(self._calculate_skewness(data)) < 0.5 and abs(self._calculate_kurtosis(data)) < 0.5
        }
        
    def _update_monitoring_history(self, monitoring_results: Dict):
        """Update monitoring history for trend analysis."""
        # Keep only recent history (last 1000 records)
        if len(self.drift_history) > 1000:
            self.drift_history = self.drift_history[-1000:]
            
        self.drift_history.append({
            'timestamp': monitoring_results['timestamp'],
            'drift_score': monitoring_results.get('feature_drift', {}).get('overall_drift_score', 0),
            'alerts_count': len(monitoring_results['alerts'])
        })
        
    def generate_monitoring_report(self, monitoring_results: Dict) -> str:
        """Generate a comprehensive monitoring report."""
        
        report = f"""
🔍 PRODUCTION MONITORING REPORT
==============================
Timestamp: {monitoring_results['timestamp']}
Sample Size: {monitoring_results['sample_size']:,}

🚨 ALERTS ({len(monitoring_results['alerts'])}):
"""
        
        if monitoring_results['alerts']:
            for alert in monitoring_results['alerts']:
                report += f"  [{alert['severity']}] {alert['type']}: {alert['message']}\n"
        else:
            report += "  ✅ No alerts detected\n"
            
        report += f"""
📊 DRIFT ANALYSIS:
- Feature Drift Score: {monitoring_results.get('feature_drift', {}).get('overall_drift_score', 0):.3f}
- Drift Detected: {monitoring_results['drift_detected']}
- Drifted Features: {len(monitoring_results.get('feature_drift', {}).get('drifted_features', []))}

📈 PERFORMANCE:
- Performance Degraded: {monitoring_results['performance_degraded']}
"""
        
        if 'performance_change' in monitoring_results:
            perf = monitoring_results['performance_change']
            report += f"- RMSE Change: {perf['rmse_change_pct']:+.1f}%\n"
            report += f"- MAE Change: {perf['mae_change_pct']:+.1f}%\n"
            
        return report
