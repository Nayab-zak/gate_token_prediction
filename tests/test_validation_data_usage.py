"""
Test script to verify if training agents are properly using validation data.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import MODEL_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("validation_data_test")

def check_model_validation_usage():
    """
    Check if model hyperparameter files contain validation metrics,
    which indicates that validation data was properly used.
    """
    model_types = [
        'rf_classic', 'rf_augmented',
        'xgb_classic', 'xgb_augmented',
        'lgbm_classic', 'lgbm_augmented',
        'mlp_classic', 'mlp_augmented',
        'lstm_classic', 'lstm_augmented',
        'catboost_classic', 'catboost_augmented'
    ]
    
    validation_usage = {}
    for model_type in model_types:
        model_dir = os.path.join(MODEL_DIR, model_type)
        hp_path = os.path.join(model_dir, 'hyperparam.json')
        
        if not os.path.exists(hp_path):
            logger.warning(f"⚠️ No hyperparameter file found for {model_type}")
            validation_usage[model_type] = {
                'exists': False,
                'uses_validation': False,
                'metrics': None
            }
            continue
            
        try:
            with open(hp_path, 'r') as f:
                params = json.load(f)
                
            has_validation_metrics = 'validation_metrics' in params
            validation_usage[model_type] = {
                'exists': True,
                'uses_validation': has_validation_metrics,
                'metrics': params.get('validation_metrics')
            }
        except Exception as e:
            logger.error(f"❌ Error reading {hp_path}: {e}")
            validation_usage[model_type] = {
                'exists': True,
                'error': str(e),
                'uses_validation': False
            }
    
    return validation_usage


def report_validation_usage(usage_data):
    """
    Generate a report on validation data usage across models.
    """
    logger.info("\n" + "="*60)
    logger.info(" VALIDATION DATA USAGE REPORT ".center(60, "="))
    logger.info("="*60)
    
    total_models = len(usage_data)
    models_with_validation = sum(1 for model in usage_data.values() if model['uses_validation'])
    
    logger.info(f"Total models analyzed: {total_models}")
    logger.info(f"Models using validation data: {models_with_validation}")
    logger.info(f"Models not using validation data: {total_models - models_with_validation}")
    logger.info("-"*60)
    
    # Report on individual models
    for model_name, info in usage_data.items():
        if not info['exists']:
            logger.info(f"🔘 {model_name:20} | Not trained yet")
            continue
            
        if info.get('error'):
            logger.info(f"❌ {model_name:20} | Error: {info['error']}")
            continue
            
        if info['uses_validation']:
            metrics = info['metrics']
            metric_str = ""
            if metrics:
                if 'val_rmse' in metrics:
                    metric_str = f"RMSE: {metrics['val_rmse']:.4f}"
                elif 'val_loss' in metrics:
                    metric_str = f"Loss: {metrics['val_loss']:.4f}"
                    
            logger.info(f"✅ {model_name:20} | Uses validation data | {metric_str}")
        else:
            logger.info(f"⚠️ {model_name:20} | Does NOT use validation data")
    
    logger.info("="*60)
    
    # Summary
    success_rate = models_with_validation / total_models * 100 if total_models > 0 else 0
    logger.info(f"Validation data usage: {success_rate:.1f}%")
    
    if success_rate < 100 and success_rate > 0:
        logger.warning("⚠️ Some models are not using validation data properly.")
        logger.warning("   Run training for these models again with the updated agents.")
    elif success_rate == 0:
        logger.error("❌ None of the models are using validation data!")
        logger.error("   Please ensure that training agents have been properly updated.")
    else:
        logger.info("✅ All models are properly using validation data.")
    
    logger.info("="*60)


if __name__ == '__main__':
    logger.info("Starting validation data usage test...")
    validation_usage = check_model_validation_usage()
    report_validation_usage(validation_usage)
