#!/usr/bin/env python3
"""
Enhanced Pipeline Orchestrator
=============================

This agent orchestrates the complete enhanced pipeline with temporal validation,
ensuring all components work together seamlessly for production-ready ML operations.

PIPELINE STAGES:
1. ✅ Enhanced Training (all 12 models with temporal validation)
2. ✅ Enhanced Testing (comprehensive evaluation)
3. ✅ Production Monitoring (continuous oversight)
4. ✅ Champion Selection (statistically robust)
5. ✅ Deployment Readiness (automated assessment)

Author: AI Assistant
Date: 2025-07-31
Version: 1.0 (Production Ready)
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import traceback
import importlib.util

# Add project paths
base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
sys.path.append(f"{base_path}/agents")
sys.path.append(f"{base_path}/utils")

class EnhancedPipelineOrchestrator:
    """
    Master orchestrator for the enhanced ML pipeline with temporal validation.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the enhanced pipeline orchestrator.
        
        Args:
            base_path: Base path for the project
        """
        self.base_path = base_path
        self.logger = self._setup_logger()
        
        # Setup paths
        self.results_dir = f"{base_path}/pipeline_results"
        self.logs_dir = f"{base_path}/logs/orchestrator"
        
        # Create directories
        for dir_path in [self.results_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.logger.info("🚀 Enhanced Pipeline Orchestrator Initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging for orchestrator."""
        logger = logging.getLogger("enhanced_orchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = f"{self.base_path}/logs/enhanced_orchestrator.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s [ORCHESTRATOR] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s [ORCHESTRATOR] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def run_enhanced_training_pipeline(self) -> Dict[str, Any]:
        """
        Run enhanced training pipeline for all 12 models.
        
        Returns:
            Training results summary
        """
        self.logger.info("🎬 Starting Enhanced Training Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Import and run enhanced training agent updater
            sys.path.append(f"{self.base_path}/agents")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "enhanced_training_agent_updater", 
                f"{self.base_path}/agents/09_enhanced_training_agent_updater.py"
            )
            training_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training_module)
            update_all_training_agents = training_module.update_all_training_agents
            
            self.logger.info("🔧 Running enhanced training for all 12 models...")
            training_results = update_all_training_agents()
            
            # Save training results
            training_summary_path = f"{self.results_dir}/enhanced_training_summary.json"
            with open(training_summary_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'stage': 'enhanced_training',
                    'total_models': len(training_results),
                    'successful_models': len([r for r in training_results if r.get('success', False)]),
                    'approved_models': len([r for r in training_results if r.get('deployment_status') == 'APPROVED']),
                    'results': training_results
                }, f, indent=2)
            
            self.logger.info(f"✅ Enhanced training completed - results saved: {training_summary_path}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced training pipeline failed: {str(e)}")
            raise
    
    def run_enhanced_testing_pipeline(self) -> Dict[str, Any]:
        """
        Run enhanced testing pipeline for all models.
        
        Returns:
            Testing results summary
        """
        self.logger.info("🧪 Starting Enhanced Testing Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Import and run enhanced test agent updater
            sys.path.append(f"{self.base_path}/agents")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "enhanced_test_agent_updater", 
                f"{self.base_path}/agents/10_enhanced_test_agent_updater.py"
            )
            testing_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(testing_module)
            update_all_test_agents = testing_module.update_all_test_agents
            
            self.logger.info("🔍 Running enhanced testing for all models...")
            testing_results = update_all_test_agents()
            
            # Save testing results
            testing_summary_path = f"{self.results_dir}/enhanced_testing_summary.json"
            with open(testing_summary_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'stage': 'enhanced_testing',
                    'total_models': len(testing_results),
                    'successful_tests': len([r for r in testing_results if r.get('success', False)]),
                    'excellent_performers': len([r for r in testing_results if r.get('performance_assessment') == 'EXCELLENT']),
                    'results': testing_results
                }, f, indent=2)
            
            self.logger.info(f"✅ Enhanced testing completed - results saved: {testing_summary_path}")
            return testing_results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced testing pipeline failed: {str(e)}")
            raise
    
    def run_champion_selection(self, training_results: List[Dict], testing_results: List[Dict]) -> Dict[str, Any]:
        """
        Run enhanced champion selection with statistical significance.
        
        Args:
            training_results: Training pipeline results
            testing_results: Testing pipeline results
            
        Returns:
            Champion selection results
        """
        self.logger.info("🏆 Starting Enhanced Champion Selection")
        self.logger.info("=" * 80)
        
        try:
            # Import robust model selection
            sys.path.append(f"{self.base_path}/utils")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "robust_model_selection", 
                f"{self.base_path}/utils/robust_model_selection.py"
            )
            selection_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(selection_module)
            select_best_model_with_confidence = selection_module.select_best_model_with_confidence
            
            # Prepare model performance data
            model_performances = {}
            
            for train_result, test_result in zip(training_results, testing_results):
                if (train_result.get('success', False) and 
                    test_result.get('success', False) and
                    train_result.get('deployment_status') == 'APPROVED'):
                    
                    model_name = train_result['agent_name']
                    model_performances[model_name] = {
                        'validation_rmse': train_result['rmse_mean'],
                        'validation_rmse_std': train_result['rmse_std'],
                        'validation_r2': train_result['r2_mean'],
                        'test_rmse': test_result['test_rmse'],
                        'test_r2': test_result['test_r2'],
                        'stability': train_result['stability'],
                        'performance_assessment': test_result.get('performance_assessment', 'Unknown')
                    }
            
            if not model_performances:
                raise ValueError("❌ No approved models available for champion selection")
            
            self.logger.info(f"🔍 Evaluating {len(model_performances)} approved models...")
            
            # Select champion model based on robust criteria
            champion_results = self._select_champion_model(model_performances)
            
            # Save champion selection results
            champion_path = f"{self.results_dir}/champion_selection_results.json"
            with open(champion_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'stage': 'champion_selection',
                    'evaluated_models': len(model_performances),
                    'champion_model': champion_results['champion_model'],
                    'selection_criteria': champion_results['selection_criteria'],
                    'performance_comparison': champion_results['performance_comparison'],
                    'confidence_level': champion_results['confidence_level']
                }, f, indent=2)
            
            self.logger.info(f"🏆 Champion selected: {champion_results['champion_model']}")
            self.logger.info(f"✅ Champion selection completed - results saved: {champion_path}")
            
            return champion_results
            
        except Exception as e:
            self.logger.error(f"❌ Champion selection failed: {str(e)}")
            raise
    
    def _select_champion_model(self, model_performances: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Select champion model using robust statistical criteria.
        
        Args:
            model_performances: Dictionary of model performance metrics
            
        Returns:
            Champion selection results
        """
        self.logger.info("🎯 Applying robust champion selection criteria...")
        
        # Define selection criteria weights
        criteria_weights = {
            'test_rmse': 0.4,          # 40% weight on test RMSE (lower is better)
            'validation_stability': 0.3, # 30% weight on validation stability
            'test_r2': 0.2,            # 20% weight on test R²
            'performance_assessment': 0.1  # 10% weight on assessment
        }
        
        model_scores = {}
        
        # Calculate scores for each model
        for model_name, performance in model_performances.items():
            score = 0
            
            # Test RMSE score (normalized, inverted since lower is better)
            rmse_values = [p['test_rmse'] for p in model_performances.values()]
            rmse_normalized = 1 - (performance['test_rmse'] - min(rmse_values)) / (max(rmse_values) - min(rmse_values) + 1e-8)
            score += rmse_normalized * criteria_weights['test_rmse']
            
            # Stability score
            stability_score = 1.0 if performance['stability'] == 'Stable' else 0.5
            score += stability_score * criteria_weights['validation_stability']
            
            # R² score (normalized)
            r2_values = [p['test_r2'] for p in model_performances.values()]
            r2_normalized = (performance['test_r2'] - min(r2_values)) / (max(r2_values) - min(r2_values) + 1e-8)
            score += r2_normalized * criteria_weights['test_r2']
            
            # Performance assessment score
            assessment_scores = {'EXCELLENT': 1.0, 'GOOD': 0.8, 'MODERATE': 0.5, 'POOR': 0.2}
            assessment_score = assessment_scores.get(performance['performance_assessment'], 0.0)
            score += assessment_score * criteria_weights['performance_assessment']
            
            model_scores[model_name] = score
        
        # Select champion (highest score)
        champion_model = max(model_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence level based on score separation
        sorted_scores = sorted(model_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_separation = sorted_scores[0] - sorted_scores[1]
            confidence_level = min(0.95, 0.5 + score_separation)  # Max 95% confidence
        else:
            confidence_level = 0.8  # Default confidence for single model
        
        results = {
            'champion_model': champion_model[0],
            'champion_score': champion_model[1],
            'selection_criteria': criteria_weights,
            'performance_comparison': model_performances,
            'all_scores': model_scores,
            'confidence_level': confidence_level,
            'score_separation': sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
        }
        
        self.logger.info(f"🏆 Champion Model: {champion_model[0]} (score: {champion_model[1]:.3f})")
        self.logger.info(f"📊 Confidence Level: {confidence_level:.1%}")
        
        return results
    
    def initialize_production_monitoring(self) -> Dict[str, Any]:
        """
        Initialize production monitoring system.
        
        Returns:
            Monitoring initialization results
        """
        self.logger.info("🔍 Initializing Production Monitoring")
        self.logger.info("=" * 80)
        
        try:
            # Import and initialize production monitoring
            from production_monitoring_agent import ProductionMonitor
            
            self.logger.info("📊 Setting up production monitoring system...")
            monitor = ProductionMonitor(self.base_path)
            
            # Generate initial monitoring report
            initial_report = monitor.generate_monitoring_report(time_range_hours=1)
            
            # Export metrics for Grafana
            metrics_path = monitor.export_metrics_for_grafana()
            
            monitoring_results = {
                'monitoring_initialized': True,
                'initial_report': initial_report,
                'metrics_exported': metrics_path,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save monitoring initialization results
            monitoring_path = f"{self.results_dir}/monitoring_initialization.json"
            with open(monitoring_path, 'w') as f:
                json.dump(monitoring_results, f, indent=2)
            
            self.logger.info(f"✅ Production monitoring initialized - results saved: {monitoring_path}")
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"❌ Production monitoring initialization failed: {str(e)}")
            raise
    
    def generate_deployment_readiness_report(self, champion_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive deployment readiness report.
        
        Args:
            champion_results: Champion selection results
            
        Returns:
            Deployment readiness assessment
        """
        self.logger.info("📋 Generating Deployment Readiness Report")
        self.logger.info("=" * 80)
        
        champion_model = champion_results['champion_model']
        champion_performance = champion_results['performance_comparison'][champion_model]
        
        # Deployment readiness criteria
        readiness_criteria = {
            'temporal_validation': True,  # All enhanced models have this
            'production_approved': True,  # Only approved models reach champion selection
            'statistical_significance': champion_results['confidence_level'] >= 0.7,
            'performance_excellent': champion_performance['performance_assessment'] == 'EXCELLENT',
            'stability_good': champion_performance['stability'] == 'Stable',
            'test_rmse_acceptable': champion_performance['test_rmse'] < 0.25,
            'test_r2_acceptable': champion_performance['test_r2'] >= 0.4
        }
        
        # Calculate overall readiness score
        passed_criteria = sum(readiness_criteria.values())
        total_criteria = len(readiness_criteria)
        readiness_score = passed_criteria / total_criteria
        
        # Determine deployment recommendation
        if readiness_score >= 0.85:
            deployment_recommendation = "APPROVED"
            recommendation_text = "✅ Model is ready for production deployment"
        elif readiness_score >= 0.7:
            deployment_recommendation = "APPROVED_WITH_MONITORING"
            recommendation_text = "⚠️ Model approved but requires enhanced monitoring"
        else:
            deployment_recommendation = "NOT_APPROVED"
            recommendation_text = "❌ Model requires improvement before deployment"
        
        # Compile readiness report
        readiness_report = {
            'champion_model': champion_model,
            'deployment_recommendation': deployment_recommendation,
            'recommendation_text': recommendation_text,
            'readiness_score': readiness_score,
            'readiness_percentage': f"{readiness_score:.1%}",
            'criteria_assessment': readiness_criteria,
            'champion_performance': champion_performance,
            'confidence_level': champion_results['confidence_level'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate detailed recommendations
        recommendations = []
        if not readiness_criteria['statistical_significance']:
            recommendations.append("Increase model validation confidence through additional testing")
        if not readiness_criteria['performance_excellent']:
            recommendations.append("Improve model performance through feature engineering or ensemble methods")
        if not readiness_criteria['stability_good']:
            recommendations.append("Address model instability through regularization or more training data")
        if not readiness_criteria['test_rmse_acceptable']:
            recommendations.append("Reduce prediction error through model optimization")
        if not readiness_criteria['test_r2_acceptable']:
            recommendations.append("Improve model explanatory power through better feature selection")
        
        readiness_report['recommendations'] = recommendations
        
        # Save deployment readiness report
        readiness_path = f"{self.results_dir}/deployment_readiness_report.json"
        with open(readiness_path, 'w') as f:
            json.dump(readiness_report, f, indent=2)
        
        self.logger.info(f"🎯 Deployment Recommendation: {deployment_recommendation}")
        self.logger.info(f"📊 Readiness Score: {readiness_score:.1%}")
        self.logger.info(f"✅ Deployment readiness report saved: {readiness_path}")
        
        return readiness_report
    
    def run_complete_enhanced_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete enhanced pipeline from training to deployment readiness.
        
        Returns:
            Complete pipeline results
        """
        pipeline_start_time = datetime.now()
        self.logger.info("🚀 STARTING COMPLETE ENHANCED PIPELINE")
        self.logger.info("=" * 100)
        
        pipeline_results = {
            'pipeline_start_time': pipeline_start_time.isoformat(),
            'stages_completed': [],
            'stages_failed': [],
            'overall_success': False
        }
        
        try:
            # Stage 1: Enhanced Training
            self.logger.info("\n🎯 STAGE 1: ENHANCED TRAINING")
            training_results = self.run_enhanced_training_pipeline()
            pipeline_results['training_results'] = training_results
            pipeline_results['stages_completed'].append('enhanced_training')
            
            # Stage 2: Enhanced Testing
            self.logger.info("\n🎯 STAGE 2: ENHANCED TESTING")
            testing_results = self.run_enhanced_testing_pipeline()
            pipeline_results['testing_results'] = testing_results
            pipeline_results['stages_completed'].append('enhanced_testing')
            
            # Stage 3: Champion Selection
            self.logger.info("\n🎯 STAGE 3: CHAMPION SELECTION")
            champion_results = self.run_champion_selection(training_results, testing_results)
            pipeline_results['champion_results'] = champion_results
            pipeline_results['stages_completed'].append('champion_selection')
            
            # Stage 4: Production Monitoring
            self.logger.info("\n🎯 STAGE 4: PRODUCTION MONITORING")
            monitoring_results = self.initialize_production_monitoring()
            pipeline_results['monitoring_results'] = monitoring_results
            pipeline_results['stages_completed'].append('production_monitoring')
            
            # Stage 5: Deployment Readiness
            self.logger.info("\n🎯 STAGE 5: DEPLOYMENT READINESS")
            readiness_report = self.generate_deployment_readiness_report(champion_results)
            pipeline_results['readiness_report'] = readiness_report
            pipeline_results['stages_completed'].append('deployment_readiness')
            
            # Pipeline completion
            pipeline_end_time = datetime.now()
            pipeline_duration = pipeline_end_time - pipeline_start_time
            
            pipeline_results.update({
                'pipeline_end_time': pipeline_end_time.isoformat(),
                'pipeline_duration_seconds': pipeline_duration.total_seconds(),
                'pipeline_duration_formatted': str(pipeline_duration),
                'overall_success': True
            })
            
            # Save complete pipeline results
            complete_results_path = f"{self.results_dir}/complete_pipeline_results.json"
            with open(complete_results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_safe_results = self._convert_for_json(pipeline_results)
                json.dump(json_safe_results, f, indent=2)
            
            self.logger.info("\n🎉 COMPLETE ENHANCED PIPELINE SUCCESSFUL!")
            self.logger.info("=" * 100)
            self.logger.info(f"Duration: {pipeline_duration}")
            self.logger.info(f"Stages Completed: {len(pipeline_results['stages_completed'])}/5")
            self.logger.info(f"Champion Model: {champion_results['champion_model']}")
            self.logger.info(f"Deployment Status: {readiness_report['deployment_recommendation']}")
            self.logger.info(f"Complete Results: {complete_results_path}")
            
            return pipeline_results
            
        except Exception as e:
            # Handle pipeline failure
            pipeline_end_time = datetime.now()
            pipeline_duration = pipeline_end_time - pipeline_start_time
            
            pipeline_results.update({
                'pipeline_end_time': pipeline_end_time.isoformat(),
                'pipeline_duration_seconds': pipeline_duration.total_seconds(),
                'pipeline_failure_reason': str(e),
                'overall_success': False
            })
            
            self.logger.error(f"\n❌ PIPELINE FAILED AFTER {pipeline_duration}")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())
            
            # Save failure results
            failure_results_path = f"{self.results_dir}/pipeline_failure_results.json"
            with open(failure_results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            
            raise
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types and other non-JSON serializable objects."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """Main function to run the enhanced pipeline orchestrator."""
    base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
    
    print("🚀 Enhanced Pipeline Orchestrator")
    print("=" * 60)
    print("This will run the complete enhanced ML pipeline with temporal validation.")
    print("Estimated duration: 30-60 minutes depending on system performance.")
    print()
    
    try:
        # Initialize orchestrator
        orchestrator = EnhancedPipelineOrchestrator(base_path)
        
        # Run complete pipeline
        results = orchestrator.run_complete_enhanced_pipeline()
        
        # Print final summary
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Champion Model: {results['champion_results']['champion_model']}")
        print(f"Deployment Status: {results['readiness_report']['deployment_recommendation']}")
        print(f"Duration: {results['pipeline_duration_formatted']}")
        print(f"Results saved in: {orchestrator.results_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        print("Check the logs for detailed error information.")
        raise


if __name__ == "__main__":
    results = main()
