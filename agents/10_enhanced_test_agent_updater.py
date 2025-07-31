#!/usr/bin/env python3
"""
Enhanced Test Agent Updater
===========================

This agent systematically updates all test agents to work with enhanced models
that include temporal validation metadata and production readiness information.

ENHANCEMENTS:
✅ Load enhanced models with metadata
✅ Validate model production readiness before testing
✅ Include temporal validation info in test outputs
✅ Add comprehensive test result analysis
✅ Generate production deployment recommendations

Author: AI Assistant
Date: 2025-07-31
Version: 1.0 (Production Ready)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import json

# Add utils to path
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')

def setup_enhanced_test_logger(agent_name: str) -> logging.Logger:
    """Setup enhanced logging for test agent updates."""
    
    log_dir = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/logs/enhanced_testing"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"enhanced_test_{agent_name}")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(f"{log_dir}/enhanced_test_{agent_name}.log")
        file_formatter = logging.Formatter(
            '%(asctime)s [ENHANCED_TEST_%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

class EnhancedTestAgent:
    """
    Enhanced test agent that works with temporally validated models.
    
    This class provides comprehensive testing with production readiness validation
    and enhanced reporting capabilities.
    """
    
    def __init__(self, model_type: str, variant: str, base_path: str):
        """
        Initialize enhanced test agent.
        
        Args:
            model_type: Type of model (rf, xgb, lgbm, catboost, lstm, mlp)
            variant: Model variant (classic, augmented)
            base_path: Base path for the project
        """
        self.model_type = model_type
        self.variant = variant
        self.agent_name = f"{model_type}_{variant}"
        self.base_path = base_path
        self.logger = setup_enhanced_test_logger(self.agent_name)
        
        # Setup paths
        self.model_dir = f"{base_path}/models/{self.agent_name}"
        self.data_dir = f"{base_path}/data"
        self.output_dir = f"{base_path}/data/final_output"
        self.reports_dir = f"{base_path}/test_reports"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.logger.info(f"🧪 Initialized Enhanced Test Agent: {self.agent_name}")
    
    def load_enhanced_model(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Load enhanced model with temporal validation metadata.
        
        Returns:
            Tuple of (model, metadata)
        """
        self.logger.info("📦 Loading enhanced model with metadata...")
        
        # Try enhanced model first
        enhanced_model_path = f"{self.model_dir}/enhanced_{self.agent_name}_model.joblib"
        fallback_model_path = f"{self.model_dir}/model.joblib"
        
        if os.path.exists(enhanced_model_path):
            model_path = enhanced_model_path
            self.logger.info(f"✅ Loading enhanced model: {model_path}")
        elif os.path.exists(fallback_model_path):
            model_path = fallback_model_path
            self.logger.warning(f"⚠️ Enhanced model not found, using fallback: {model_path}")
        else:
            raise FileNotFoundError(f"❌ No model found for {self.agent_name}")
        
        # Load model data
        model_data = joblib.load(model_path)
        
        # Extract model and metadata
        if isinstance(model_data, dict):
            model = model_data.get('model')
            metadata = model_data.get('metadata', {})
            cv_results = model_data.get('cv_results', {})
        else:
            # Fallback for old model format
            model = model_data
            metadata = {
                'model_info': {'agent_name': self.agent_name},
                'validation_info': {'temporal_validation': False},
                'production_readiness': {'deployment_status': 'UNKNOWN'}
            }
            cv_results = {}
        
        # Log model information
        temporal_validated = metadata.get('validation_info', {}).get('temporal_validation', False)
        deployment_status = metadata.get('production_readiness', {}).get('deployment_status', 'UNKNOWN')
        
        self.logger.info(f"📊 Model loaded:")
        self.logger.info(f"   Type: {type(model).__name__}")
        self.logger.info(f"   Temporal Validation: {'✅ YES' if temporal_validated else '❌ NO'}")
        self.logger.info(f"   Deployment Status: {deployment_status}")
        
        if temporal_validated:
            rmse_mean = metadata.get('performance_metrics', {}).get('rmse_mean', 'N/A')
            stability = metadata.get('performance_metrics', {}).get('performance_stability', 'Unknown')
            self.logger.info(f"   RMSE: {rmse_mean}")
            self.logger.info(f"   Stability: {stability}")
        
        return model, metadata
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Load test data maintaining temporal order with transaction identifiers.
        
        Returns:
            Tuple of (X, y, timestamps, transaction_keys)
        """
        self.logger.info("📊 Loading test data with transaction identifiers...")
        
        # Determine input path based on variant
        if self.variant == "augmented":
            input_path = f"{self.data_dir}/encoded_input/test_input_augmented.csv"
            if not os.path.exists(input_path):
                # Fallback to classic features
                input_path = f"{self.data_dir}/features/test_features.csv"
                self.logger.warning(f"⚠️ Using fallback to test_features.csv for {self.variant} variant")
        else:
            input_path = f"{self.data_dir}/features/test_features.csv"
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"❌ Test data not found: {input_path}")
        
        # Load data
        test_data = pd.read_csv(input_path)
        self.logger.info(f"📈 Loaded {len(test_data)} test samples")
        
        # Ensure required columns exist
        required_cols = ['datetime', 'TokenCount', 'TerminalID', 'MoveType', 'Desig']
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        # Parse datetime and sort
        test_data['datetime'] = pd.to_datetime(test_data['datetime'])
        test_data = test_data.sort_values('datetime').reset_index(drop=True)
        
        # Extract components
        feature_cols = [col for col in test_data.columns 
                       if col not in ['datetime', 'TokenCount', 'TerminalID', 'MoveType', 'Desig']]
        
        X = test_data[feature_cols]
        y = test_data['TokenCount']
        timestamps = test_data['datetime']
        transaction_keys = test_data[['TerminalID', 'MoveType', 'Desig']]
        
        self.logger.info(f"📊 Features: {len(feature_cols)} columns")
        self.logger.info(f"🎯 Target range: [{y.min():.2f}, {y.max():.2f}]")
        self.logger.info(f"📅 Time range: {timestamps.min()} to {timestamps.max()}")
        self.logger.info(f"🔑 Transaction keys: {len(transaction_keys)} rows")
        
        return X, y, timestamps, transaction_keys
    
    def validate_model_production_readiness(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate that model is ready for production testing.
        
        Args:
            metadata: Model metadata
            
        Returns:
            True if model is production ready
        """
        self.logger.info("🏭 Validating model production readiness...")
        
        deployment_status = metadata.get('production_readiness', {}).get('deployment_status', 'UNKNOWN')
        temporal_validation = metadata.get('validation_info', {}).get('temporal_validation', False)
        
        # Production readiness criteria
        criteria = {
            'Temporal Validation': temporal_validation,
            'Deployment Status': deployment_status in ['APPROVED', 'NEEDS_IMPROVEMENT'],
            'Has Metadata': len(metadata) > 0
        }
        
        self.logger.info("🎯 Production Readiness Check:")
        all_passed = True
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.logger.info(f"   {criterion}: {status}")
            if not passed:
                all_passed = False
        
        if deployment_status == 'NEEDS_IMPROVEMENT':
            self.logger.warning("⚠️ Model needs improvement but proceeding with testing")
        elif deployment_status == 'FAILED':
            self.logger.error("❌ Model failed validation - testing not recommended")
            all_passed = False
        
        return all_passed
    
    def perform_enhanced_testing(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                timestamps: pd.Series, transaction_keys: pd.DataFrame,
                                metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform enhanced testing with comprehensive metrics and analysis.
        
        Args:
            model: Trained model
            X: Test features
            y: Test target
            timestamps: Test timestamps
            transaction_keys: Transaction identifiers
            metadata: Model metadata
            
        Returns:
            Enhanced test results DataFrame
        """
        self.logger.info("🧪 Performing enhanced testing...")
        
        # Generate predictions
        try:
            predictions = model.predict(X)
            self.logger.info("✅ Predictions generated successfully")
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {str(e)}")
            raise
        
        # Calculate comprehensive metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Handle potential issues
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            self.logger.warning("⚠️ NaN or Inf values in predictions - cleaning")
            valid_mask = np.isfinite(predictions)
            if np.any(valid_mask):
                predictions[~valid_mask] = np.mean(predictions[valid_mask])
            else:
                predictions = np.zeros_like(predictions)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        
        # MAPE with protection against division by zero
        mape_values = np.abs((y - predictions) / np.where(y != 0, y, 1e-8)) * 100
        mape = np.mean(mape_values)
        
        # R²
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Additional metrics
        max_error = np.max(np.abs(y - predictions))
        errors = y - predictions
        
        self.logger.info(f"📈 Test Results:")
        self.logger.info(f"   RMSE: {rmse:.4f}")
        self.logger.info(f"   MAE: {mae:.4f}")
        self.logger.info(f"   MAPE: {mape:.2f}%")
        self.logger.info(f"   R²: {r2:.4f}")
        self.logger.info(f"   Max Error: {max_error:.4f}")
        
        # Create enhanced results DataFrame
        results_df = pd.DataFrame({
            'datetime': timestamps,
            'TerminalID': transaction_keys['TerminalID'],
            'MoveType': transaction_keys['MoveType'],
            'Desig': transaction_keys['Desig'],
            'actual': y,
            'prediction': predictions,
            'error': errors,
            'abs_error': np.abs(errors),
            'pct_error': (errors / np.where(y != 0, y, 1e-8)) * 100
        })
        
        # Add model metadata as columns
        results_df['model'] = self.agent_name
        results_df['temporal_validation'] = metadata.get('validation_info', {}).get('temporal_validation', False)
        results_df['deployment_status'] = metadata.get('production_readiness', {}).get('deployment_status', 'UNKNOWN')
        results_df['validation_rmse'] = metadata.get('performance_metrics', {}).get('rmse_mean', np.nan)
        results_df['model_stability'] = metadata.get('performance_metrics', {}).get('performance_stability', 'Unknown')
        
        # Add test metrics
        results_df['test_rmse'] = rmse
        results_df['test_mae'] = mae
        results_df['test_mape'] = mape
        results_df['test_r2'] = r2
        
        return results_df
    
    def analyze_test_performance(self, results_df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze test performance and compare with validation results.
        
        Args:
            results_df: Test results DataFrame
            metadata: Model metadata
            
        Returns:
            Performance analysis results
        """
        self.logger.info("📊 Analyzing test performance...")
        
        # Extract test metrics
        test_rmse = results_df['test_rmse'].iloc[0]
        test_mae = results_df['test_mae'].iloc[0]
        test_r2 = results_df['test_r2'].iloc[0]
        
        # Extract validation metrics
        val_rmse = metadata.get('performance_metrics', {}).get('rmse_mean', np.nan)
        val_mae = metadata.get('performance_metrics', {}).get('mae_mean', np.nan)
        val_r2 = metadata.get('performance_metrics', {}).get('r2_mean', np.nan)
        
        # Performance comparison
        analysis = {
            'test_metrics': {
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2
            },
            'validation_metrics': {
                'rmse': val_rmse,
                'mae': val_mae,
                'r2': val_r2
            }
        }
        
        # Calculate performance degradation
        if not np.isnan(val_rmse):
            rmse_degradation = ((test_rmse - val_rmse) / val_rmse) * 100
            analysis['performance_degradation'] = {
                'rmse_degradation_pct': rmse_degradation
            }
            
            # Performance assessment
            if rmse_degradation < 10:
                performance_assessment = "EXCELLENT"
                self.logger.info("🎯 Performance Assessment: EXCELLENT (< 10% degradation)")
            elif rmse_degradation < 25:
                performance_assessment = "GOOD"
                self.logger.info("🎯 Performance Assessment: GOOD (10-25% degradation)")
            elif rmse_degradation < 50:
                performance_assessment = "MODERATE"
                self.logger.warning("⚠️ Performance Assessment: MODERATE (25-50% degradation)")
            else:
                performance_assessment = "POOR"
                self.logger.error("❌ Performance Assessment: POOR (>50% degradation)")
            
            analysis['performance_assessment'] = performance_assessment
            
        else:
            self.logger.warning("⚠️ No validation metrics available for comparison")
            analysis['performance_assessment'] = "UNKNOWN"
        
        # Data distribution analysis
        errors = results_df['error']
        analysis['error_distribution'] = {
            'mean_error': float(errors.mean()),
            'std_error': float(errors.std()),
            'median_error': float(errors.median()),
            'q25_error': float(errors.quantile(0.25)),
            'q75_error': float(errors.quantile(0.75))
        }
        
        return analysis
    
    def save_enhanced_results(self, results_df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """
        Save enhanced test results with comprehensive analysis.
        
        Args:
            results_df: Test results DataFrame
            analysis: Performance analysis
            
        Returns:
            Path to saved results
        """
        self.logger.info("💾 Saving enhanced test results...")
        
        # Save test results CSV
        output_path = f"{self.output_dir}/enhanced_test_results_{self.agent_name}.csv"
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"📊 Test results saved: {output_path}")
        
        # Save analysis JSON
        analysis_path = f"{self.reports_dir}/test_analysis_{self.agent_name}.json"
        analysis_with_metadata = {
            'agent_name': self.agent_name,
            'test_timestamp': datetime.now().isoformat(),
            'test_samples': len(results_df),
            'analysis': analysis
        }
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis_with_metadata, f, indent=2)
        self.logger.info(f"📈 Analysis saved: {analysis_path}")
        
        return output_path
    
    def run_enhanced_testing(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete enhanced testing pipeline.
        
        Returns:
            Tuple of (results_df, analysis)
        """
        self.logger.info("🎬 Starting Enhanced Testing Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # 1. Load enhanced model
            model, metadata = self.load_enhanced_model()
            
            # 2. Validate production readiness
            if not self.validate_model_production_readiness(metadata):
                self.logger.warning("⚠️ Model may not be production ready - proceeding with caution")
            
            # 3. Load test data
            X, y, timestamps, transaction_keys = self.load_test_data()
            
            # 4. Perform testing
            results_df = self.perform_enhanced_testing(model, X, y, timestamps, transaction_keys, metadata)
            
            # 5. Analyze performance
            analysis = self.analyze_test_performance(results_df, metadata)
            
            # 6. Save results
            output_path = self.save_enhanced_results(results_df, analysis)
            
            # 7. Log summary
            self.logger.info("\n🎉 Enhanced Testing Pipeline Completed!")
            self.logger.info("-" * 50)
            self.logger.info(f"Agent: {self.agent_name}")
            self.logger.info(f"Test RMSE: {analysis['test_metrics']['rmse']:.4f}")
            self.logger.info(f"Performance: {analysis.get('performance_assessment', 'Unknown')}")
            self.logger.info(f"Results: {output_path}")
            
            return results_df, analysis
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced testing failed: {str(e)}")
            raise

def update_all_test_agents():
    """Update all 12 test agents with enhanced testing capabilities."""
    
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    
    # Setup main logger
    main_logger = setup_enhanced_test_logger("all_test_agents")
    main_logger.info("🧪 Starting Mass Test Agent Update")
    main_logger.info("=" * 80)
    
    # Define all agent configurations
    agent_configs = [
        ("rf", "classic"),
        ("rf", "augmented"),
        ("xgb", "classic"),
        ("xgb", "augmented"),
        ("lgbm", "classic"),
        ("lgbm", "augmented"),
        ("catboost", "classic"),
        ("catboost", "augmented"),
        ("lstm", "classic"),
        ("lstm", "augmented"),
        ("mlp", "classic"),
        ("mlp", "augmented")
    ]
    
    results_summary = []
    successful_tests = 0
    failed_tests = 0
    
    for model_type, variant in agent_configs:
        agent_name = f"{model_type}_{variant}"
        main_logger.info(f"\n🧪 Testing Agent: {agent_name}")
        main_logger.info("-" * 40)
        
        try:
            # Create and run enhanced test agent
            agent = EnhancedTestAgent(model_type, variant, base_path)
            results_df, analysis = agent.run_enhanced_testing()
            
            # Store results
            results_summary.append({
                'agent_name': agent_name,
                'test_rmse': analysis['test_metrics']['rmse'],
                'test_r2': analysis['test_metrics']['r2'],
                'performance_assessment': analysis.get('performance_assessment', 'Unknown'),
                'test_samples': len(results_df),
                'success': True
            })
            
            successful_tests += 1
            main_logger.info(f"✅ {agent_name} testing completed successfully")
            
        except Exception as e:
            main_logger.error(f"❌ {agent_name} testing failed: {str(e)}")
            results_summary.append({
                'agent_name': agent_name,
                'error': str(e),
                'success': False
            })
            failed_tests += 1
    
    # Generate final summary
    main_logger.info(f"\n🎊 MASS TESTING COMPLETED!")
    main_logger.info("=" * 80)
    main_logger.info(f"Total Agents: {len(agent_configs)}")
    main_logger.info(f"Successful: {successful_tests}")
    main_logger.info(f"Failed: {failed_tests}")
    
    # Performance summary
    excellent_performers = len([r for r in results_summary if r.get('performance_assessment') == 'EXCELLENT'])
    main_logger.info(f"Excellent Performers: {excellent_performers}")
    
    # Save summary
    summary_path = f"{base_path}/test_reports/mass_testing_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump({
            'testing_timestamp': datetime.now().isoformat(),
            'total_agents': len(agent_configs),
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'excellent_performers': excellent_performers,
            'results': results_summary
        }, f, indent=2)
    
    main_logger.info(f"📊 Summary saved: {summary_path}")
    
    return results_summary

if __name__ == "__main__":
    # Run mass testing of all enhanced agents
    try:
        results = update_all_test_agents()
        print("\n🎉 Enhanced test agent update completed!")
        print("Check the logs and test reports for detailed results.")
    except Exception as e:
        print(f"❌ Mass testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
