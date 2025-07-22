#!/usr/bin/env python3
"""
LightGBM Training Agent - Works with both dense and sparse data
"""

from agents.base_training_agent import BaseTrainingAgent
import lightgbm as lgb
from typing import Dict, Any

class TrainLightGBMAgent(BaseTrainingAgent):
    def __init__(self, data_type: str = "dense", config_path: str = "config.yaml", log_dir: str = "logs"):
        self.data_type = data_type
        model_name = f"lightgbm_{data_type}" 
        super().__init__(model_name, config_path, log_dir)
    
    def get_search_space(self) -> Dict[str, Any]:
        """Return hyperparameter search space for LightGBM"""
        return self.config['tuning']['search_spaces']['lightgbm']
    
    def create_model(self, **params) -> lgb.LGBMRegressor:
        """Create LightGBM model with given parameters"""
        return lgb.LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params
        )
    
    def get_model_type(self) -> str:
        """Return model type"""
        return self.data_type

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM Agent')
    parser.add_argument('--data-type', choices=['dense', 'sparse'], default='dense',
                       help='Data type to use (dense embeddings or sparse wide)')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--hyper-tune', action='store_true', 
                       help='Force hyperparameter retuning even if existing params found')
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only (no training)')
    
    args = parser.parse_args()
    
    agent = TrainLightGBMAgent(args.data_type, args.config)
    
    if args.test_only:
        success = agent.run(mode='test')
    else:
        success = agent.run(mode='train', force_retune=args.hyper_tune)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
