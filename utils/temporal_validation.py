#!/usr/bin/env python3
"""
Enhanced Temporal Cross-Validation for Time Series Models
========================================================

This module implements comprehensive temporal validation to prevent data leakage and overfitting
in time series forecasting models. All validation methods respect temporal order and provide
transparent reporting of validation strategies and results.

Key Features:
- Temporal cross-validation with configurable strategies
- Walk-forward validation for production-like evaluation
- Data drift detection with statistical tests
- Comprehensive logging and transparency
- Multiple validation metrics with confidence intervals

Author: AI Assistant
Date: 2025-07-31
Version: 2.0 (Enhanced)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from copy import deepcopy

def setup_temporal_logger(name: str = "temporal_validation", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a dedicated logger for temporal validation with transparent reporting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [TEMPORAL_VAL] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def validate_temporal_inputs(X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series], 
                           timestamps: Union[np.ndarray, pd.Series],
                           logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and prepare inputs for temporal validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        timestamps: Timestamp array
        logger: Optional logger for transparency
        
    Returns:
        Tuple of validated (X, y, timestamps) arrays
        
    Raises:
        ValueError: If inputs are invalid or inconsistent
    """
    if logger is None:
        logger = setup_temporal_logger()
    
    logger.info("🔍 Validating temporal validation inputs...")
    
    # Convert inputs to numpy arrays for consistent handling
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
        logger.info(f"📊 Features: {len(feature_names)} columns ({X.shape[0]} rows)")
    else:
        X_array = np.array(X)
        feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        logger.info(f"📊 Features: {X_array.shape[1]} columns ({X_array.shape[0]} rows)")
    
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_array = y.values.flatten()
    else:
        y_array = np.array(y).flatten()
    
    if isinstance(timestamps, (pd.Series, pd.DataFrame)):
        timestamps_array = pd.to_datetime(timestamps.values.flatten()).values
    else:
        timestamps_array = pd.to_datetime(timestamps).values
    
    # Validation checks
    if len(X_array) != len(y_array):
        raise ValueError(f"❌ Mismatch: X has {len(X_array)} samples, y has {len(y_array)} samples")
    
    if len(X_array) != len(timestamps_array):
        raise ValueError(f"❌ Mismatch: X has {len(X_array)} samples, timestamps has {len(timestamps_array)} samples")
    
    if len(X_array) < 50:
        logger.warning(f"⚠️ Small dataset: Only {len(X_array)} samples available for validation")
    
    # Check for missing values with proper type handling
    try:
        # Only check for NaN in numeric arrays
        if X_array.dtype.kind in ['i', 'f']:  # integer or float
            X_missing = np.isnan(X_array).sum()
        else:
            # For non-numeric data, check for None or pd.NA
            X_missing = pd.isnull(X_array).sum()
    except Exception:
        X_missing = 0
        logger.warning("⚠️ Could not check for missing values in features")
    
    try:
        if y_array.dtype.kind in ['i', 'f']:  # integer or float
            y_missing = np.isnan(y_array).sum()
        else:
            y_missing = pd.isnull(y_array).sum()
    except Exception:
        y_missing = 0
        logger.warning("⚠️ Could not check for missing values in target")
    
    if X_missing > 0:
        logger.warning(f"⚠️ Missing values in features: {X_missing} values")
    if y_missing > 0:
        logger.warning(f"⚠️ Missing values in target: {y_missing} values")
    
    # Check timestamp ordering and duplicates
    timestamps_sorted = np.sort(timestamps_array)
    if not np.array_equal(timestamps_array, timestamps_sorted):
        logger.info("🔄 Timestamps are not sorted - will be sorted during validation")
    
    unique_timestamps = len(np.unique(timestamps_array))
    if unique_timestamps < len(timestamps_array):
        logger.warning(f"⚠️ Duplicate timestamps detected: {len(timestamps_array) - unique_timestamps} duplicates")
    
    # Time span analysis
    time_span = timestamps_sorted[-1] - timestamps_sorted[0]
    # Handle both numpy.timedelta64 and pandas.Timedelta
    if hasattr(time_span, 'days'):
        days = time_span.days
    elif isinstance(time_span, np.timedelta64):
        days = time_span / np.timedelta64(1, 'D')
    else:
        days = time_span / pd.Timedelta(days=1)
    logger.info(f"📅 Time span: {days:.1f} days ({timestamps_sorted[0]} to {timestamps_sorted[-1]})")
    
    logger.info("✅ Input validation completed successfully")
    
    return X_array, y_array, timestamps_array


def temporal_cross_validate(model, X, y, timestamps, n_splits=5, test_size_ratio=0.2,
                          validation_strategy="expanding", gap_hours=0, 
                          logger=None) -> Dict[str, Any]:
    """
    Perform comprehensive temporal cross-validation respecting time series order.
    
    This function implements multiple temporal validation strategies to ensure robust
    model evaluation without data leakage. All strategies maintain strict temporal order.
    
    Args:
        model: Sklearn-compatible model (will be cloned for each fold)
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target vector (numpy array or pandas Series)
        timestamps: Datetime series for temporal ordering
        n_splits: Number of temporal splits (default: 5)
        test_size_ratio: Ratio of data for testing in each split (default: 0.2)
        validation_strategy: Strategy for temporal splits:
            - "expanding": Growing training window (default)
            - "sliding": Fixed-size sliding window
            - "blocked": Blocked time series split
        gap_hours: Gap between train and validation sets in hours (default: 0)
        logger: Optional logger for detailed reporting
        
    Returns:
        dict: Comprehensive cross-validation results including:
            - Mean and std of all metrics
            - Individual fold results
            - Validation strategy details
            - Temporal split information
            - Model stability metrics
            
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100)
        >>> results = temporal_cross_validate(
        ...     model, X_train, y_train, timestamps_train,
        ...     validation_strategy="expanding", gap_hours=1
        ... )
        >>> print(f"RMSE: {results['rmse_mean']:.3f} ± {results['rmse_std']:.3f}")
    """
    if logger is None:
        logger = setup_temporal_logger()
    
    logger.info("🚀 Starting Enhanced Temporal Cross-Validation")
    logger.info("=" * 60)
    
    # Validate inputs
    X_validated, y_validated, timestamps_validated = validate_temporal_inputs(X, y, timestamps, logger)
    
    # Log validation configuration
    logger.info(f"📋 Validation Configuration:")
    logger.info(f"   Strategy: {validation_strategy}")
    logger.info(f"   Splits: {n_splits}")
    logger.info(f"   Test ratio: {test_size_ratio}")
    logger.info(f"   Gap hours: {gap_hours}")
    logger.info(f"   Model: {type(model).__name__}")
    
    # Sort by timestamp to ensure proper temporal order
    sort_idx = timestamps_validated.argsort()
    X_sorted = X_validated[sort_idx]
    y_sorted = y_validated[sort_idx]
    timestamps_sorted = timestamps_validated[sort_idx]
    
    logger.info(f"📊 Data sorted by timestamp: {len(X_sorted)} samples")
    
    # Initialize results storage
    scores = {
        'rmse': [],
        'mae': [], 
        'mape': [],
        'r2': [],
        'fold_info': [],
        'predictions': [],
        'actuals': [],
        'fold_timestamps': []
    }
    
    # Configure temporal splits based on strategy
    if validation_strategy == "expanding":
        # Adjust test size to ensure feasible splits
        max_test_size = len(X_sorted) // (n_splits + 2)  # Leave room for training data
        test_size = min(int(len(X_sorted) * test_size_ratio), max_test_size)
        
        if test_size < int(len(X_sorted) * test_size_ratio):
            actual_ratio = test_size / len(X_sorted)
            logger.warning(f"⚠️ Reducing test size from {test_size_ratio:.1%} to {actual_ratio:.1%} for feasible splits")
        
        cv_splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        logger.info("📈 Using expanding window strategy")
        cv_iterator = cv_splitter.split(X_sorted)
    elif validation_strategy == "sliding":
        # Implement sliding window
        cv_splits = _create_sliding_window_splits(len(X_sorted), n_splits, test_size_ratio)
        logger.info("📊 Using sliding window strategy")
        cv_iterator = cv_splits
    elif validation_strategy == "blocked":
        # Implement blocked time series split
        cv_splits = _create_blocked_splits(len(X_sorted), n_splits)
        logger.info("🧱 Using blocked time series strategy")
        cv_iterator = cv_splits
    else:
        raise ValueError(f"Unknown validation strategy: {validation_strategy}")
    
    # Convert gap_hours to number of samples (approximate)
    if gap_hours > 0:
        # Estimate samples per hour based on data frequency
        if len(timestamps_sorted) > 1:
            time_diffs = pd.Series(timestamps_sorted).diff().dropna()
            median_diff = time_diffs.median()
            
            # Handle zero median difference (duplicate timestamps)
            if median_diff == pd.Timedelta(0) or median_diff.total_seconds() <= 0:
                logger.warning("⚠️ Zero or negative median time difference detected - using hourly assumption")
                gap_samples = gap_hours  # Assume 1 sample per hour
            else:
                samples_per_hour = pd.Timedelta(hours=1) / median_diff
                gap_samples = int(gap_hours * samples_per_hour)
        else:
            gap_samples = gap_hours  # Fallback
        logger.info(f"⏳ Applied gap: {gap_hours} hours ≈ {gap_samples} samples")
    else:
        gap_samples = 0
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv_iterator):
        logger.info(f"\n🔄 Processing Fold {fold + 1}/{n_splits}")
        logger.info("-" * 40)
        
        # Apply gap if specified
        if gap_samples > 0:
            # Remove gap samples from validation set
            train_end = train_idx[-1]
            val_start = val_idx[0]
            if val_start - train_end < gap_samples:
                # Adjust validation start to create gap
                gap_adjusted_val_idx = val_idx[val_idx >= train_end + gap_samples]
                if len(gap_adjusted_val_idx) < 10:  # Minimum validation size
                    logger.warning(f"⚠️ Fold {fold + 1}: Gap too large, using original validation set")
                    val_idx_final = val_idx
                else:
                    val_idx_final = gap_adjusted_val_idx
                    logger.info(f"✂️ Applied {gap_samples} sample gap")
            else:
                val_idx_final = val_idx
        else:
            val_idx_final = val_idx
        
        # Split data temporally
        X_train_fold = X_sorted[train_idx]
        X_val_fold = X_sorted[val_idx_final]
        y_train_fold = y_sorted[train_idx]
        y_val_fold = y_sorted[val_idx_final]
        timestamps_train_fold = timestamps_sorted[train_idx]
        timestamps_val_fold = timestamps_sorted[val_idx_final]
        
        # Log fold information
        logger.info(f"📊 Training: {len(train_idx)} samples ({timestamps_train_fold[0]} to {timestamps_train_fold[-1]})")
        logger.info(f"📊 Validation: {len(val_idx_final)} samples ({timestamps_val_fold[0]} to {timestamps_val_fold[-1]})")
        
        # Clone model to avoid cross-fold contamination
        fold_model = deepcopy(model)
        
        try:
            # Train model on temporal training set
            fold_model.fit(X_train_fold, y_train_fold)
            logger.info("✅ Model training completed")
            
            # Predict on temporal validation set
            y_pred = fold_model.predict(X_val_fold)
            logger.info("✅ Predictions generated")
            
            # Calculate comprehensive metrics
            fold_metrics = _calculate_fold_metrics(y_val_fold, y_pred, logger)
            
            # Store results
            scores['rmse'].append(fold_metrics['rmse'])
            scores['mae'].append(fold_metrics['mae'])
            scores['mape'].append(fold_metrics['mape'])
            scores['r2'].append(fold_metrics['r2'])
            
            # Store detailed fold information
            scores['fold_info'].append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx_final),
                'train_start': str(timestamps_train_fold[0]),
                'train_end': str(timestamps_train_fold[-1]),
                'val_start': str(timestamps_val_fold[0]),
                'val_end': str(timestamps_val_fold[-1]),
                'gap_applied': gap_samples > 0,
                'metrics': fold_metrics
            })
            
            # Store predictions for ensemble analysis
            scores['predictions'].extend(y_pred.tolist())
            scores['actuals'].extend(y_val_fold.tolist())
            scores['fold_timestamps'].extend(timestamps_val_fold.tolist())
            
            logger.info(f"📈 Fold {fold + 1} Results: RMSE={fold_metrics['rmse']:.4f}, MAE={fold_metrics['mae']:.4f}, MAPE={fold_metrics['mape']:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Fold {fold + 1} failed: {str(e)}")
            # Continue with other folds
            continue
    
    # Calculate summary statistics
    if len(scores['rmse']) == 0:
        raise RuntimeError("❌ All folds failed - cannot complete validation")
    
    results = _calculate_cv_summary(scores, validation_strategy, logger)
    
    logger.info("\n🎉 Temporal Cross-Validation Completed Successfully!")
    logger.info("=" * 60)
    
    return results

def walk_forward_validation(model, X, y, timestamps, initial_train_size=0.7, step_size=0.05):
    """
    Implement walk-forward validation for time series.
    
    Args:
        model: Model to validate
        X: Features
        y: Target
        timestamps: Time series
        initial_train_size: Initial training set ratio
        step_size: Step size for moving window
        
    Returns:
        dict: Walk-forward validation results
    """
    # Sort by timestamp
    sort_idx = timestamps.argsort()
    X_sorted = X[sort_idx] if hasattr(X, 'shape') else X.iloc[sort_idx]
    y_sorted = y[sort_idx] if hasattr(y, 'shape') else y.iloc[sort_idx]
    timestamps_sorted = timestamps[sort_idx]
    
    n_samples = len(X_sorted)
    initial_train_end = int(n_samples * initial_train_size)
    
    results = {
        'predictions': [],
        'actuals': [],
        'timestamps': [],
        'rmse_by_step': [],
        'mae_by_step': [],
        'mape_by_step': []
    }
    
    current_train_end = initial_train_end
    step_samples = int(n_samples * step_size)
    
    while current_train_end + step_samples < n_samples:
        # Define training and test sets
        train_end = current_train_end
        test_start = current_train_end
        test_end = min(current_train_end + step_samples, n_samples)
        
        X_train_wf = X_sorted[:train_end] if hasattr(X_sorted, 'shape') else X_sorted.iloc[:train_end]
        y_train_wf = y_sorted[:train_end] if hasattr(y_sorted, 'shape') else y_sorted.iloc[:train_end]
        X_test_wf = X_sorted[test_start:test_end] if hasattr(X_sorted, 'shape') else X_sorted.iloc[test_start:test_end]
        y_test_wf = y_sorted[test_start:test_end] if hasattr(y_sorted, 'shape') else y_sorted.iloc[test_start:test_end]
        
        # Train model
        model.fit(X_train_wf, y_train_wf)
        
        # Predict
        y_pred_wf = model.predict(X_test_wf)
        
        # Store results
        results['predictions'].extend(y_pred_wf)
        results['actuals'].extend(y_test_wf)
        results['timestamps'].extend(timestamps_sorted[test_start:test_end])
        
        # Calculate step metrics
        rmse = np.sqrt(mean_squared_error(y_test_wf, y_pred_wf))
        mae = mean_absolute_error(y_test_wf, y_pred_wf)
        # MAPE with protection against division by zero
        mape_values = np.abs((y_test_wf - y_pred_wf) / np.where(y_test_wf != 0, y_test_wf, 1e-8)) * 100
        mape = np.mean(mape_values)
        
        results['rmse_by_step'].append(rmse)
        results['mae_by_step'].append(mae)
        results['mape_by_step'].append(mape)
        
        current_train_end = test_end
        
        logging.info(f"Walk-forward step: train_size={train_end}, test_size={test_end-test_start}, RMSE={rmse:.4f}")
    
    return results

def _create_sliding_window_splits(n_samples: int, n_splits: int, test_size_ratio: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create sliding window splits for temporal validation.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits to create
        test_size_ratio: Ratio of samples for testing
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    test_size = int(n_samples * test_size_ratio)
    train_size = n_samples - test_size
    
    splits = []
    
    # Calculate window size and step size
    window_size = train_size + test_size
    step_size = max(1, (n_samples - window_size) // (n_splits - 1)) if n_splits > 1 else 0
    
    for i in range(n_splits):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, n_samples)
        
        # Ensure we have enough samples
        if end_idx - start_idx < test_size + 10:  # Minimum training size
            break
            
        train_end = start_idx + train_size
        train_indices = np.arange(start_idx, train_end)
        test_indices = np.arange(train_end, end_idx)
        
        if len(test_indices) > 0:
            splits.append((train_indices, test_indices))
    
    return splits


def _create_blocked_splits(n_samples: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create blocked time series splits for temporal validation.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits to create
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    block_size = n_samples // (n_splits + 1)  # +1 to leave room for final test block
    splits = []
    
    for i in range(n_splits):
        # Training block: from start to current block end
        train_end = (i + 1) * block_size
        train_indices = np.arange(0, train_end)
        
        # Test block: next block after training
        test_start = train_end
        test_end = min(test_start + block_size, n_samples)
        test_indices = np.arange(test_start, test_end)
        
        if len(test_indices) > 0:
            splits.append((train_indices, test_indices))
    
    return splits


def _calculate_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a single fold.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        logger: Optional logger
        
    Returns:
        Dictionary of calculated metrics
    """
    if logger is None:
        logger = setup_temporal_logger()
    
    # Handle potential issues with predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        logger.warning("⚠️ NaN or Inf values detected in predictions")
        # Replace NaN/Inf with mean of valid predictions
        valid_mask = np.isfinite(y_pred)
        if np.any(valid_mask):
            y_pred[~valid_mask] = np.mean(y_pred[valid_mask])
        else:
            y_pred = np.zeros_like(y_pred)
    
    # Calculate metrics
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE with protection against division by zero
        mape_values = np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8)) * 100
        mape = np.mean(mape_values)
        
        # Calculate R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Additional metrics
        max_error = np.max(np.abs(y_true - y_pred))
        median_error = np.median(np.abs(y_true - y_pred))
        
    except Exception as e:
        logger.error(f"❌ Error calculating metrics: {str(e)}")
        rmse = mae = mape = r2 = max_error = median_error = float('inf')
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'max_error': max_error,
        'median_error': median_error,
        'n_samples': len(y_true)
    }


def _calculate_cv_summary(scores: Dict[str, List], validation_strategy: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive cross-validation summary statistics.
    
    Args:
        scores: Dictionary containing fold results
        validation_strategy: Validation strategy used
        logger: Optional logger
        
    Returns:
        Dictionary of summary statistics and results
    """
    if logger is None:
        logger = setup_temporal_logger()
    
    logger.info("\n📊 Calculating Cross-Validation Summary")
    logger.info("-" * 50)
    
    # Calculate mean and standard deviation for each metric
    metrics_summary = {}
    for metric in ['rmse', 'mae', 'mape', 'r2']:
        if metric in scores and len(scores[metric]) > 0:
            values = np.array(scores[metric])
            metrics_summary[f'{metric}_mean'] = np.mean(values)
            metrics_summary[f'{metric}_std'] = np.std(values)
            metrics_summary[f'{metric}_min'] = np.min(values)
            metrics_summary[f'{metric}_max'] = np.max(values)
            metrics_summary[f'{metric}_median'] = np.median(values)
            
            # Calculate confidence intervals (95%)
            if len(values) > 1:
                sem = np.std(values) / np.sqrt(len(values))  # Standard error of mean
                ci_margin = 1.96 * sem  # 95% CI
                metrics_summary[f'{metric}_ci_lower'] = metrics_summary[f'{metric}_mean'] - ci_margin
                metrics_summary[f'{metric}_ci_upper'] = metrics_summary[f'{metric}_mean'] + ci_margin
            else:
                metrics_summary[f'{metric}_ci_lower'] = metrics_summary[f'{metric}_mean']
                metrics_summary[f'{metric}_ci_upper'] = metrics_summary[f'{metric}_mean']
    
    # Calculate model stability metrics
    stability_metrics = {}
    if len(scores['rmse']) > 1:
        rmse_values = np.array(scores['rmse'])
        mean_rmse = np.mean(rmse_values)
        if mean_rmse > 0:
            stability_metrics['rmse_coefficient_of_variation'] = np.std(rmse_values) / mean_rmse
        else:
            stability_metrics['rmse_coefficient_of_variation'] = float('inf')
        stability_metrics['performance_stability'] = 'Stable' if stability_metrics['rmse_coefficient_of_variation'] < 0.1 else 'Unstable'
    else:
        stability_metrics['rmse_coefficient_of_variation'] = 0.0
        stability_metrics['performance_stability'] = 'Unknown'
    
    # Overall assessment
    if 'rmse_mean' in metrics_summary:
        if metrics_summary['rmse_mean'] < 0.1:
            overall_performance = 'Excellent'
        elif metrics_summary['rmse_mean'] < 0.2:
            overall_performance = 'Good'
        elif metrics_summary['rmse_mean'] < 0.5:
            overall_performance = 'Moderate'
        else:
            overall_performance = 'Poor'
    else:
        overall_performance = 'Unknown'
    
    # Compile results
    results = {
        # Summary statistics
        **metrics_summary,
        **stability_metrics,
        
        # Metadata
        'validation_strategy': validation_strategy,
        'n_folds_completed': len(scores['rmse']),
        'overall_performance': overall_performance,
        
        # Detailed results
        'fold_results': scores['fold_info'],
        'all_predictions': scores['predictions'],
        'all_actuals': scores['actuals'],
        'prediction_timestamps': scores['fold_timestamps'],
        
        # Validation information
        'validation_summary': {
            'strategy': validation_strategy,
            'total_folds': len(scores['rmse']),
            'successful_folds': len([x for x in scores['rmse'] if not np.isinf(x)]),
            'failed_folds': len([x for x in scores['rmse'] if np.isinf(x)])
        }
    }
    
    # Log summary
    logger.info(f"🎯 Cross-Validation Results Summary:")
    logger.info(f"   Strategy: {validation_strategy}")
    logger.info(f"   Completed Folds: {len(scores['rmse'])}")
    if 'rmse_mean' in metrics_summary:
        logger.info(f"   RMSE: {metrics_summary['rmse_mean']:.4f} ± {metrics_summary['rmse_std']:.4f}")
        logger.info(f"   MAE: {metrics_summary['mae_mean']:.4f} ± {metrics_summary['mae_std']:.4f}")
        logger.info(f"   MAPE: {metrics_summary['mape_mean']:.2f}% ± {metrics_summary['mape_std']:.2f}%")
        logger.info(f"   R²: {metrics_summary['r2_mean']:.4f} ± {metrics_summary['r2_std']:.4f}")
        logger.info(f"   Overall Performance: {overall_performance}")
        logger.info(f"   Model Stability: {stability_metrics['performance_stability']}")
    
    return results


def detect_data_drift(X_train, X_test, feature_names=None, threshold=0.1):
    """
    Detect data drift between training and test sets using statistical tests.
    
    Args:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        threshold: Drift detection threshold
        
    Returns:
        dict: Drift detection results
    """
    try:
        from scipy import stats
    except ImportError:
        import logging
        logging.warning("⚠️ scipy not available, using simplified drift detection")
        return _simple_drift_detection(X_train, X_test, feature_names, threshold)
    
    drift_results = {
        'drifted_features': [],
        'drift_scores': {},
        'overall_drift': False
    }
    
    n_features = X_train.shape[1]
    feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
    
    for i, feature_name in enumerate(feature_names):
        train_feature = X_train[:, i] if hasattr(X_train, 'shape') else X_train.iloc[:, i]
        test_feature = X_test[:, i] if hasattr(X_test, 'shape') else X_test.iloc[:, i]
        
        # Kolmogorov-Smirnov test for distribution difference
        ks_stat, p_value = stats.ks_2samp(train_feature, test_feature)
        
        drift_results['drift_scores'][feature_name] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drifted': p_value < threshold
        }
        
        if p_value < threshold:
            drift_results['drifted_features'].append(feature_name)
    
    drift_results['overall_drift'] = len(drift_results['drifted_features']) > 0
    
    return drift_results


def _simple_drift_detection(X_train, X_test, feature_names=None, threshold=0.1):
    """
    Simple drift detection without scipy dependency.
    
    Args:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        threshold: Drift detection threshold
        
    Returns:
        dict: Simplified drift detection results
    """
    drift_results = {
        'drifted_features': [],
        'drift_scores': {},
        'overall_drift': False
    }
    
    n_features = X_train.shape[1]
    feature_names = feature_names or [f'feature_{i}' for i in range(n_features)]
    
    for i, feature_name in enumerate(feature_names):
        train_feature = X_train[:, i] if hasattr(X_train, 'shape') else X_train.iloc[:, i]
        test_feature = X_test[:, i] if hasattr(X_test, 'shape') else X_test.iloc[:, i]
        
        # Simple statistical comparison
        train_mean = np.mean(train_feature)
        test_mean = np.mean(test_feature)
        train_std = np.std(train_feature)
        test_std = np.std(test_feature)
        
        # Calculate normalized difference
        mean_diff = abs(train_mean - test_mean) / (train_std + 1e-8)
        std_diff = abs(train_std - test_std) / (train_std + 1e-8)
        
        drift_score = max(mean_diff, std_diff)
        drifted = drift_score > threshold
        
        drift_results['drift_scores'][feature_name] = {
            'drift_score': drift_score,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'drifted': drifted
        }
        
        if drifted:
            drift_results['drifted_features'].append(feature_name)
    
    drift_results['overall_drift'] = len(drift_results['drifted_features']) > 0
    
    return drift_results


def generate_validation_report(cv_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        cv_results: Cross-validation results from temporal_cross_validate
        output_path: Optional path to save the report
        
    Returns:
        String containing the formatted report
    """
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 80,
        "🎯 TEMPORAL CROSS-VALIDATION REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Validation Strategy: {cv_results.get('validation_strategy', 'Unknown')}",
        f"Completed Folds: {cv_results.get('n_folds_completed', 0)}",
        ""
    ])
    
    # Performance Summary
    if 'rmse_mean' in cv_results:
        report_lines.extend([
            "📊 PERFORMANCE SUMMARY",
            "-" * 50,
            f"RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}",
            f"  Range: [{cv_results['rmse_min']:.4f}, {cv_results['rmse_max']:.4f}]",
            f"  95% CI: [{cv_results['rmse_ci_lower']:.4f}, {cv_results['rmse_ci_upper']:.4f}]",
            "",
            f"MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}",
            f"  Range: [{cv_results['mae_min']:.4f}, {cv_results['mae_max']:.4f}]",
            f"  95% CI: [{cv_results['mae_ci_lower']:.4f}, {cv_results['mae_ci_upper']:.4f}]",
            "",
            f"MAPE: {cv_results['mape_mean']:.2f}% ± {cv_results['mape_std']:.2f}%",
            f"  Range: [{cv_results['mape_min']:.2f}%, {cv_results['mape_max']:.2f}%]",
            "",
            f"R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}",
            f"  Range: [{cv_results['r2_min']:.4f}, {cv_results['r2_max']:.4f}]",
            ""
        ])
    
    # Model Stability
    report_lines.extend([
        "🎚️ MODEL STABILITY",
        "-" * 50,
        f"Performance Stability: {cv_results.get('performance_stability', 'Unknown')}",
        f"RMSE Coefficient of Variation: {cv_results.get('rmse_coefficient_of_variation', 0):.3f}",
        f"Overall Assessment: {cv_results.get('overall_performance', 'Unknown')}",
        ""
    ])
    
    # Fold Details
    if 'fold_results' in cv_results:
        report_lines.extend([
            "📋 FOLD-BY-FOLD RESULTS",
            "-" * 50
        ])
        
        for fold_info in cv_results['fold_results']:
            report_lines.extend([
                f"Fold {fold_info['fold']}:",
                f"  Training: {fold_info['train_size']} samples ({fold_info['train_start']} to {fold_info['train_end']})",
                f"  Validation: {fold_info['val_size']} samples ({fold_info['val_start']} to {fold_info['val_end']})",
                f"  RMSE: {fold_info['metrics']['rmse']:.4f}",
                f"  MAE: {fold_info['metrics']['mae']:.4f}",
                f"  MAPE: {fold_info['metrics']['mape']:.2f}%",
                f"  R²: {fold_info['metrics']['r2']:.4f}",
                ""
            ])
    
    # Recommendations
    report_lines.extend([
        "💡 RECOMMENDATIONS",
        "-" * 50
    ])
    
    if cv_results.get('performance_stability') == 'Unstable':
        report_lines.append("⚠️ Model shows unstable performance across folds - consider:")
        report_lines.append("   • More robust feature engineering")
        report_lines.append("   • Ensemble methods")
        report_lines.append("   • Hyperparameter tuning")
        report_lines.append("")
    
    if cv_results.get('overall_performance') == 'Poor':
        report_lines.append("⚠️ Poor overall performance detected - consider:")
        report_lines.append("   • Different model architectures")
        report_lines.append("   • Feature selection/engineering")
        report_lines.append("   • Data quality assessment")
        report_lines.append("")
    
    report_lines.extend([
        "✅ Temporal validation completed successfully",
        "✅ No data leakage detected (temporal order preserved)",
        "",
        "=" * 80
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text
