#!/usr/bin/env python3
"""
Elastic Net Training Agent - Dense embeddings baseline
"""

from agents.base_training_agent import BaseTrainingAgent
from sklearn.linear_model import ElasticNet
from typing import Dict, Any

class TrainElasticNetAgent(BaseTrainingAgent):
    def __init__(self, config_path: str = "config.yaml", log_dir: str = "logs"):
        super().__init__("elasticnet", config_path, log_dir)
    
    def get_search_space(self) -> Dict[str, Any]:
        """Return hyperparameter search space for Elastic Net"""
        return self.config['tuning']['search_spaces']['elasticnet']
    
    def create_model(self, **params) -> ElasticNet:
        """Create Elastic Net model with given parameters"""
        return ElasticNet(
            random_state=42,
            **params
        )
    
    def get_model_type(self) -> str:
        """Return model type"""
        return 'dense'

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Elastic Net Agent')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--hyper-tune', action='store_true', 
                       help='Force hyperparameter retuning even if existing params found')
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only (no training)')
    
    args = parser.parse_args()
    
    agent = TrainElasticNetAgent(args.config)
    
    if args.test_only:
        success = agent.run(mode='test')
    else:
        success = agent.run(mode='train', force_retune=args.hyper_tune)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
