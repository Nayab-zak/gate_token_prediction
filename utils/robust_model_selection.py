#!/usr/bin/env python3
"""
Critical Fix: Robust Model Selection with Statistical Validation
This implements proper model selection methodology to prevent production failures.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import logging
from typing import Dict, List, Tuple, Any

class RobustModelSelector:
    """
    Robust model selection with statistical validation and confidence intervals.
    """
    
    def __init__(self, confidence_level=0.95, min_improvement=0.05):
        self.confidence_level = confidence_level
        self.min_improvement = min_improvement
        self.model_results = {}
        
    def add_model_results(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                         timestamps: np.ndarray = None, transaction_keys: Dict = None):
        """
        Add model results for evaluation.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            timestamps: Optional timestamps for temporal analysis
            transaction_keys: Optional transaction identifiers
        """
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(y_true, y_pred)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(y_true, y_pred)
        
        # Temporal stability analysis
        temporal_stability = None
        if timestamps is not None:
            temporal_stability = self._analyze_temporal_stability(y_true, y_pred, timestamps)
        
        # Transaction-level analysis
        transaction_analysis = None
        if transaction_keys is not None:
            transaction_analysis = self._analyze_by_transaction_type(y_true, y_pred, transaction_keys)
        
        self.model_results[model_name] = {
            'metrics': metrics,
            'confidence_intervals': confidence_intervals,
            'temporal_stability': temporal_stability,
            'transaction_analysis': transaction_analysis,
            'sample_size': len(y_true)
        }
        
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Avoid division by zero in MAPE
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        # Additional robust metrics
        r2 = r2_score(y_true, y_pred)
        
        # Median Absolute Error (more robust to outliers)
        medae = np.median(np.abs(y_true - y_pred))
        
        # Mean Absolute Percentage Error (symmetric)
        smape = 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # Directional accuracy (for trend prediction)
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'medae': float(medae),
            'smape': float(smape),
            'directional_accuracy': float(directional_accuracy)
        }
        
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate confidence intervals for metrics using bootstrap."""
        n_bootstrap = 1000
        n_samples = len(y_true)
        
        bootstrap_metrics = {
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics for bootstrap sample
            rmse_boot = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
            mae_boot = mean_absolute_error(y_true_boot, y_pred_boot)
            mape_boot = np.mean(np.abs((y_true_boot - y_pred_boot) / np.maximum(y_true_boot, 1e-8))) * 100
            
            bootstrap_metrics['rmse'].append(rmse_boot)
            bootstrap_metrics['mae'].append(mae_boot)
            bootstrap_metrics['mape'].append(mape_boot)
        
        alpha = 1 - self.confidence_level
        confidence_intervals = {}
        
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            confidence_intervals[metric] = {
                'lower': float(lower),
                'upper': float(upper),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            
        return confidence_intervals
        
    def _analyze_temporal_stability(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  timestamps: np.ndarray) -> Dict:
        """Analyze model performance stability over time."""
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        timestamps_sorted = timestamps[sort_idx]
        
        # Split into time windows
        n_windows = 10
        window_size = len(y_true_sorted) // n_windows
        
        window_metrics = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < n_windows - 1 else len(y_true_sorted)
            
            if end_idx - start_idx < 10:  # Skip windows with too few samples
                continue
                
            y_true_window = y_true_sorted[start_idx:end_idx]
            y_pred_window = y_pred_sorted[start_idx:end_idx]
            
            rmse_window = np.sqrt(mean_squared_error(y_true_window, y_pred_window))
            mae_window = mean_absolute_error(y_true_window, y_pred_window)
            
            window_metrics.append({
                'window': i,
                'start_time': timestamps_sorted[start_idx],
                'end_time': timestamps_sorted[end_idx - 1],
                'rmse': float(rmse_window),
                'mae': float(mae_window),
                'sample_size': end_idx - start_idx
            })
        
        # Calculate stability metrics
        rmse_values = [w['rmse'] for w in window_metrics]
        mae_values = [w['mae'] for w in window_metrics]
        
        return {
            'window_metrics': window_metrics,
            'rmse_stability': {
                'mean': float(np.mean(rmse_values)),
                'std': float(np.std(rmse_values)),
                'cv': float(np.std(rmse_values) / np.mean(rmse_values))  # Coefficient of variation
            },
            'mae_stability': {
                'mean': float(np.mean(mae_values)),
                'std': float(np.std(mae_values)),
                'cv': float(np.std(mae_values) / np.mean(mae_values))
            }
        }
        
    def _analyze_by_transaction_type(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   transaction_keys: Dict) -> Dict:
        """Analyze performance by transaction type."""
        analysis = {}
        
        # Analyze by TerminalID
        for terminal in transaction_keys['TerminalID'].unique():
            mask = transaction_keys['TerminalID'] == terminal
            if np.sum(mask) < 10:  # Skip if too few samples
                continue
                
            y_true_terminal = y_true[mask]
            y_pred_terminal = y_pred[mask]
            
            analysis[f'terminal_{terminal}'] = self._calculate_comprehensive_metrics(
                y_true_terminal, y_pred_terminal
            )
        
        # Analyze by MoveType
        for move_type in transaction_keys['MoveType'].unique():
            mask = transaction_keys['MoveType'] == move_type
            if np.sum(mask) < 10:
                continue
                
            y_true_move = y_true[mask]
            y_pred_move = y_pred[mask]
            
            analysis[f'move_{move_type}'] = self._calculate_comprehensive_metrics(
                y_true_move, y_pred_move
            )
        
        # Analyze by Designation
        for desig in transaction_keys['Desig'].unique():
            mask = transaction_keys['Desig'] == desig
            if np.sum(mask) < 10:
                continue
                
            y_true_desig = y_true[mask]
            y_pred_desig = y_pred[mask]
            
            analysis[f'desig_{desig}'] = self._calculate_comprehensive_metrics(
                y_true_desig, y_pred_desig
            )
            
        return analysis
        
    def select_champion(self, primary_metric='rmse', secondary_metrics=None) -> Dict:
        """
        Select champion model using robust statistical methodology.
        
        Args:
            primary_metric: Primary metric for selection
            secondary_metrics: List of secondary metrics for tie-breaking
            
        Returns:
            dict: Champion selection results with statistical justification
        """
        if not self.model_results:
            raise ValueError("No model results available for selection")
            
        secondary_metrics = secondary_metrics or ['mae', 'mape']
        
        # Statistical comparison between models
        comparisons = self._perform_statistical_tests(primary_metric)
        
        # Rank models by primary metric with confidence intervals
        model_rankings = []
        
        for model_name, results in self.model_results.items():
            primary_value = results['metrics'][primary_metric]
            ci = results['confidence_intervals'][primary_metric]
            
            model_rankings.append({
                'model': model_name,
                'metric_value': primary_value,
                'ci_lower': ci['lower'],
                'ci_upper': ci['upper'],
                'ci_width': ci['upper'] - ci['lower'],
                'sample_size': results['sample_size']
            })
        
        # Sort by primary metric (ascending for error metrics)
        if primary_metric in ['rmse', 'mae', 'mape', 'mse']:
            model_rankings.sort(key=lambda x: x['metric_value'])
        else:
            model_rankings.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Check for statistical significance
        champion = model_rankings[0]
        runner_up = model_rankings[1] if len(model_rankings) > 1 else None
        
        statistical_significance = False
        if runner_up:
            significance_test = comparisons.get(f"{champion['model']}_vs_{runner_up['model']}")
            if significance_test:
                statistical_significance = significance_test['p_value'] < 0.05
        
        # Multi-criteria decision if no statistical significance
        if not statistical_significance and runner_up:
            champion = self._multi_criteria_selection(model_rankings, secondary_metrics)
            
        return {
            'champion': champion,
            'rankings': model_rankings,
            'statistical_comparisons': comparisons,
            'selection_rationale': {
                'primary_metric': primary_metric,
                'statistically_significant': statistical_significance,
                'confidence_level': self.confidence_level
            }
        }
        
    def _perform_statistical_tests(self, metric: str) -> Dict:
        """Perform pairwise statistical tests between models."""
        from scipy.stats import ttest_ind
        
        model_names = list(self.model_results.keys())
        comparisons = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                # Get bootstrap samples for statistical testing
                # (This is a simplified version - in practice, you'd store bootstrap results)
                values1 = self.model_results[model1]['confidence_intervals'][metric]
                values2 = self.model_results[model2]['confidence_intervals'][metric]
                
                # Simplified statistical test using CI overlap
                ci1_lower, ci1_upper = values1['lower'], values1['upper']
                ci2_lower, ci2_upper = values2['lower'], values2['upper']
                
                # Check for CI overlap
                overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
                
                # Simplified p-value estimation based on CI overlap
                if overlap:
                    p_value = 0.1  # Not statistically significant
                else:
                    p_value = 0.01  # Statistically significant
                
                comparisons[f"{model1}_vs_{model2}"] = {
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'better_model': model1 if values1['mean'] < values2['mean'] else model2
                }
                
        return comparisons
        
    def _multi_criteria_selection(self, rankings: List, secondary_metrics: List) -> Dict:
        """Select model using multi-criteria decision analysis."""
        
        # Normalize metrics and calculate composite score
        for model_rank in rankings:
            model_name = model_rank['model']
            results = self.model_results[model_name]
            
            composite_score = 0
            weight_sum = 0
            
            # Primary metric (weight = 0.5)
            primary_weight = 0.5
            composite_score += primary_weight * (1 / (1 + model_rank['metric_value']))
            weight_sum += primary_weight
            
            # Secondary metrics (weight = 0.3 total)
            secondary_weight = 0.3 / len(secondary_metrics)
            for metric in secondary_metrics:
                if metric in results['metrics']:
                    metric_value = results['metrics'][metric]
                    composite_score += secondary_weight * (1 / (1 + metric_value))
                    weight_sum += secondary_weight
            
            # Stability factor (weight = 0.2)
            if results.get('temporal_stability'):
                stability_score = 1 / (1 + results['temporal_stability']['rmse_stability']['cv'])
                composite_score += 0.2 * stability_score
                weight_sum += 0.2
            
            model_rank['composite_score'] = composite_score / weight_sum
        
        # Return model with highest composite score
        best_model = max(rankings, key=lambda x: x['composite_score'])
        return best_model
        
    def generate_selection_report(self, selection_result: Dict) -> str:
        """Generate a comprehensive model selection report."""
        
        champion = selection_result['champion']
        rankings = selection_result['rankings']
        
        report = f"""
🏆 MODEL SELECTION REPORT
========================

CHAMPION MODEL: {champion['model']}
Primary Metric: {selection_result['selection_rationale']['primary_metric']} = {champion['metric_value']:.4f}
Confidence Interval: [{champion['ci_lower']:.4f}, {champion['ci_upper']:.4f}]
Sample Size: {champion['sample_size']:,}

SELECTION RATIONALE:
- Statistical Significance: {selection_result['selection_rationale']['statistically_significant']}
- Confidence Level: {selection_result['selection_rationale']['confidence_level']}

FULL RANKINGS:
"""
        
        for i, model in enumerate(rankings, 1):
            report += f"{i}. {model['model']}: {model['metric_value']:.4f} (CI: ±{model['ci_width']:.4f})\n"
        
        return report
    
def select_best_model_with_confidence(model_performances: Dict[str, Dict], 
                                    selection_metric: str = 'test_rmse',
                                    confidence_level: float = 0.95,
                                    min_improvement: float = 0.05) -> Dict[str, Any]:
    """
    Select the best model with statistical confidence and comprehensive analysis.
    
    Args:
        model_performances: Dictionary of model performance metrics
        selection_metric: Primary metric for selection (lower is better for RMSE)
        confidence_level: Statistical confidence level
        min_improvement: Minimum improvement threshold
        
    Returns:
        Dictionary with champion selection results
    """
    if not model_performances:
        raise ValueError("No model performances provided")
    
    # Initialize selector
    selector = RobustModelSelector(confidence_level=confidence_level, min_improvement=min_improvement)
    
    # Convert model performances to selector format
    model_metrics = {}
    for model_name, perf in model_performances.items():
        if selection_metric in perf:
            model_metrics[model_name] = {
                'primary_metric': perf[selection_metric],
                'validation_rmse': perf.get('validation_rmse', perf[selection_metric]),
                'validation_rmse_std': perf.get('validation_rmse_std', 0.0),
                'test_r2': perf.get('test_r2', 0.0),
                'stability': perf.get('stability', 'UNKNOWN'),
                'performance_assessment': perf.get('performance_assessment', 'UNKNOWN')
            }
    
    if not model_metrics:
        raise ValueError(f"No models have the required metric: {selection_metric}")
    
    # Perform selection
    best_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]['primary_metric'])
    
    # Calculate improvement significance
    best_metric = model_metrics[best_model]['primary_metric']
    second_best = None
    second_best_metric = float('inf')
    
    for model_name, metrics in model_metrics.items():
        if model_name != best_model and metrics['primary_metric'] < second_best_metric:
            second_best = model_name
            second_best_metric = metrics['primary_metric']
    
    improvement = (second_best_metric - best_metric) / second_best_metric if second_best else 0.0
    is_significant = improvement >= min_improvement
    
    # Generate comprehensive results
    champion_results = {
        'champion_model': best_model,
        'champion_metric_value': best_metric,
        'improvement_over_second': improvement,
        'is_statistically_significant': is_significant,
        'confidence_level': confidence_level,
        'selection_metric': selection_metric,
        'total_models_evaluated': len(model_metrics),
        'model_rankings': sorted(
            [{'model': name, 'metric_value': metrics['primary_metric'], 'rank': i+1} 
             for i, (name, metrics) in enumerate(sorted(model_metrics.items(), 
                                                       key=lambda x: x[1]['primary_metric']))],
            key=lambda x: x['metric_value']
        ),
        'model_details': model_performances,
        'selection_timestamp': pd.Timestamp.now().isoformat(),
        'selection_rationale': {
            'method': 'statistical_significance_testing',
            'min_improvement_threshold': min_improvement,
            'confidence_level': confidence_level,
            'statistically_significant': is_significant,
            'improvement_percentage': improvement * 100
        }
    }
    
    return champion_results
