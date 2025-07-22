#!/usr/bin/env python3
"""
MLP Regressor Training Agent - Dense embeddings
"""

from agents.base_training_agent import BaseTrainingAgent
from sklearn.neural_network import MLPRegressor
from typing import Dict, Any

class TrainMLPAgent(BaseTrainingAgent):
    def __init__(self, config_path: str = "config.yaml", log_dir: str = "logs"):
        super().__init__("mlp", config_path, log_dir)
    
    def get_search_space(self) -> Dict[str, Any]:
        """Return hyperparameter search space for MLP"""
        return self.config['tuning']['search_spaces']['mlp']
    
    def create_model(self, **params) -> MLPRegressor:
        """Create MLP model with given parameters"""
        return MLPRegressor(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            **params
        )
    
    def get_model_type(self) -> str:
        """Return model type"""
        return 'dense'

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MLP Agent')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--hyper-tune', action='store_true', 
                       help='Force hyperparameter retuning even if existing params found')
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only (no training)')
    
    args = parser.parse_args()
    
    agent = TrainMLPAgent(args.config)
    
    if args.test_only:
        success = agent.run(mode='test')
    else:
        success = agent.run(mode='train', force_retune=args.hyper_tune)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
