#!/usr/bin/env python3
"""
Hybrid MLP Regressor Training Agent - Always uses combined (wide+encoded) features
"""

from agents.base_training_agent import BaseTrainingAgent
from sklearn.neural_network import MLPRegressor
from typing import Dict, Any

class TrainHybridMLPAgent(BaseTrainingAgent):
    def __init__(self, config_path: str = "config.yaml", log_dir: str = "logs"):
        # Always use combined features and unique model name
        model_name = "hybrid_mlp"
        super().__init__(model_name, config_path, log_dir)
        self.feature_set = 'combined'
    
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

    def load_data(self):
        """Load combined (wide+encoded) features for all splits"""
        from pathlib import Path
        import pandas as pd
        pre_dir = Path(self.config['data']['preprocessed_dir'])
        train_path = pre_dir / "combined_train.csv"
        val_path = pre_dir / "combined_val.csv"
        test_path = pre_dir / "combined_test.csv"
        self.logger.info(f"Loading combined features for Hybrid MLP model")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        self.logger.info(f"Loaded - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        return train_df, val_df, test_df

    def run(self, mode='train', force_retune=False) -> bool:
        """Override run to log feature set used in metadata."""
        try:
            self.logger.info(f"Starting {mode} process for {self.model_name} (feature_set=combined)")
            result = super().run(mode=mode, force_retune=force_retune)
            return result
        except Exception as e:
            self.logger.error(f"Run failed for {self.model_name} (feature_set=combined): {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hybrid MLP Agent')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--hyper-tune', action='store_true', 
                       help='Force hyperparameter retuning even if existing params found')
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only (no training)')
    
    args = parser.parse_args()
    
    agent = TrainHybridMLPAgent(args.config)
    
    if args.test_only:
        success = agent.run(mode='test')
    else:
        success = agent.run(mode='train', force_retune=args.hyper_tune)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
