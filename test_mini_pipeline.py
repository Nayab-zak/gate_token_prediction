#!/usr/bin/env python3
"""
Test enhanced pipeline with just 2 models (1 classic, 1 augmented).
"""

import sys
import os
import logging
from datetime import datetime

# Add paths
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/utils')
sys.path.append('/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents')

def setup_test_logger():
    """Setup logger for testing."""
    logger = logging.getLogger("test_mini_pipeline")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s [TEST] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def test_mini_pipeline():
    """Test pipeline with just 2 models."""
    logger = setup_test_logger()
    logger.info("🧪 Testing Mini Enhanced Pipeline")
    logger.info("=" * 60)
    
    try:
        import importlib.util
        
        # Load training agent module
        spec = importlib.util.spec_from_file_location(
            "enhanced_training_agent_updater", 
            "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly/agents/09_enhanced_training_agent_updater.py"
        )
        training_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training_module)
        
        base_path = "/home/wk-12195/Fatima/predictive_modeling/gate_token_prediction_hourly"
        
        # Test configurations - just 2 models
        test_configs = [
            ("rf", "classic"),
            ("rf", "augmented")
        ]
        
        training_results = []
        successful_trainings = 0
        
        for model_type, variant in test_configs:
            agent_name = f"{model_type}_{variant}"
            logger.info(f"\n🤖 Training {agent_name}...")
            logger.info("-" * 40)
            
            try:
                # Create and run training agent
                agent = training_module.EnhancedTrainingAgent(model_type, variant, base_path)
                
                # Run training
                model, cv_results, deployment_status = agent.train_enhanced_model()
                
                training_results.append({
                    'agent_name': agent_name,
                    'rmse_mean': cv_results.get('rmse_mean', 0),
                    'r2_mean': cv_results.get('r2_mean', 0),
                    'deployment_status': deployment_status,
                    'success': True
                })
                
                successful_trainings += 1
                logger.info(f"✅ {agent_name} training succeeded")
                logger.info(f"   RMSE: {cv_results.get('rmse_mean', 0):.4f}")
                logger.info(f"   R²: {cv_results.get('r2_mean', 0):.4f}")
                logger.info(f"   Status: {deployment_status}")
                
            except Exception as e:
                logger.error(f"❌ {agent_name} training failed: {str(e)}")
                training_results.append({
                    'agent_name': agent_name,
                    'error': str(e),
                    'success': False
                })
        
        # Summary
        logger.info(f"\n🎊 MINI PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Total Models: {len(test_configs)}")
        logger.info(f"Successful: {successful_trainings}")
        logger.info(f"Failed: {len(test_configs) - successful_trainings}")
        
        if successful_trainings > 0:
            approved_models = len([r for r in training_results if r.get('deployment_status') == 'APPROVED'])
            logger.info(f"Approved Models: {approved_models}")
            
            return True, training_results
        else:
            return False, training_results
        
    except Exception as e:
        logger.error(f"❌ Mini pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, []

if __name__ == "__main__":
    start_time = datetime.now()
    success, results = test_mini_pipeline()
    end_time = datetime.now()
    
    print(f"\n⏰ Test Duration: {end_time - start_time}")
    
    if success:
        print("🎉 MINI PIPELINE SUCCESS - Enhanced training works!")
        print("Ready to run full pipeline.")
    else:
        print("❌ MINI PIPELINE FAILED - Need more fixes.")
