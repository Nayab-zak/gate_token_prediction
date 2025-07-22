#!/usr/bin/env python3
"""
Random Forest Training Agent - Dense embeddings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_training_agent import BaseTrainingAgent
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any

class TrainRandomForestAgent(BaseTrainingAgent):
    def __init__(self, config_path: str = "config.yaml", log_dir: str = "logs"):
        super().__init__("random_forest", config_path, log_dir)
    
    def get_search_space(self) -> Dict[str, Any]:
        """Return hyperparameter search space for Random Forest"""
        return self.config['tuning']['search_spaces']['random_forest']
    
    def create_model(self, **params) -> RandomForestRegressor:
        """Create Random Forest model with given parameters"""
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
    
    def get_model_type(self) -> str:
        """Return model type"""
        return 'dense'

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Random Forest Agent')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--hyper-tune', action='store_true', 
                       help='Force hyperparameter retuning even if existing params found')
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only (no training)')
    
    args = parser.parse_args()
    
    agent = TrainRandomForestAgent(args.config)
    
    if args.test_only:
        success = agent.run(mode='test')
    else:
        success = agent.run(mode='train', force_retune=args.hyper_tune)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
