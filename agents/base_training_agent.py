#!/usr/bin/env python3
"""
Base Training Agent - Common functionality for all model training agents
"""

import pandas as pd
import numpy as np
import logging
import yaml
import json
import joblib
import optuna
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class BaseTrainingAgent(ABC):
    def __init__(self, model_name: str, config_path: str = "config.yaml", log_dir: str = "logs"):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / f"train_{model_name}_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f'Train{model_name.title()}Agent')
        
        # Set paths
        self.models_dir = Path(self.config['models']['dir'])
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        self.predictions_dir = Path(self.config['data']['predictions_dir']) / self.model_name
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
    
    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        """Return hyperparameter search space for this model"""
        pass
    
    @abstractmethod
    def create_model(self, **params) -> Any:
        """Create model instance with given parameters"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type: 'dense' or 'sparse'"""
        pass
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load appropriate data type based on model"""
        try:
            model_type = self.get_model_type()
            
            if model_type == 'dense':
                # Load encoded embeddings
                train_path = Path(self.config['data']['encoded_input_dir']) / "Z_train.csv"
                val_path = Path(self.config['data']['encoded_input_dir']) / "Z_val.csv"
                test_path = Path(self.config['data']['encoded_input_dir']) / "Z_test.csv"
            else:  # sparse
                # Load scaled wide format
                train_path = Path(self.config['data']['preprocessed_dir']) / "X_train_scaled.csv"
                val_path = Path(self.config['data']['preprocessed_dir']) / "X_val_scaled.csv" 
                test_path = Path(self.config['data']['preprocessed_dir']) / "X_test_scaled.csv"
            
            self.logger.info(f"Loading {model_type} data for {self.model_name}")
            
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            
            self.logger.info(f"Loaded - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features_targets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                test_df: pd.DataFrame) -> Tuple:
        """Prepare X and y for training"""
        try:
            # For now, we'll create a simple target by summing all counts
            # This can be extended to predict specific series
            
            # Separate timestamps and features
            X_train = train_df.drop('timestamp', axis=1) if 'timestamp' in train_df.columns else train_df
            X_val = val_df.drop('timestamp', axis=1) if 'timestamp' in val_df.columns else val_df
            X_test = test_df.drop('timestamp', axis=1) if 'timestamp' in test_df.columns else test_df
            
            # For dense data, create target from original wide data
            if self.get_model_type() == 'dense':
                # Load original wide data for targets
                wide_train = pd.read_csv(Path(self.config['data']['preprocessed_dir']) / "wide_train.csv")
                wide_val = pd.read_csv(Path(self.config['data']['preprocessed_dir']) / "wide_val.csv")
                wide_test = pd.read_csv(Path(self.config['data']['preprocessed_dir']) / "wide_test.csv")
                
                # Sum all series columns (excluding timestamp) as target
                series_cols = [col for col in wide_train.columns if col != 'timestamp']
                y_train = wide_train[series_cols].sum(axis=1)
                y_val = wide_val[series_cols].sum(axis=1)
                y_test = wide_test[series_cols].sum(axis=1)
            else:
                # For sparse data, sum all feature columns as target
                y_train = X_train.sum(axis=1)
                y_val = X_val.sum(axis=1)
                y_test = X_test.sum(axis=1)
            
            self.logger.info(f"Prepared features - X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing features and targets: {str(e)}")
            raise
    
    def get_model_paths(self, mode='train') -> Dict[str, Path]:
        """Get paths for model artifacts"""
        if mode == 'test':
            # For test mode, find existing model files
            existing_models = list(self.models_dir.glob(f"{self.model_name}_best_model_*.pkl"))
            if not existing_models:
                # No existing models found, use current timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                # Use the timestamp from the most recent model
                latest_model = max(existing_models, key=lambda x: x.stat().st_mtime)
                # Extract timestamp from filename (format: model_best_model_YYYYMMDD_HHMMSS.pkl)
                filename_parts = latest_model.stem.split('_')
                if len(filename_parts) >= 2:
                    timestamp = '_'.join(filename_parts[-2:])  # Get last two parts: YYYYMMDD_HHMMSS
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # For training mode, use current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'best_params': self.models_dir / f"{self.model_name}_best_params_{timestamp}.json",
            'best_model': self.models_dir / f"{self.model_name}_best_model_{timestamp}.pkl",
            'tuning_study': self.models_dir / f"{self.model_name}_study_{timestamp}.pkl",
            'predictions_train': self.predictions_dir / f"{self.model_name}_train_preds_{timestamp}.csv",
            'predictions_test': self.predictions_dir / f"{self.model_name}_test_preds_{timestamp}.csv",
            'metadata': self.predictions_dir / f"{self.model_name}_metadata_{timestamp}.yaml"
        }
    
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, X_val: pd.DataFrame,
                 y_train: pd.Series, y_val: pd.Series) -> float:
        """Optuna objective function"""
        try:
            # Get hyperparameters from search space
            search_space = self.get_search_space()
            params = {}
            
            for param_name, param_config in search_space.items():
                if isinstance(param_config, list) and len(param_config) == 2:
                    if isinstance(param_config[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # Create and train model
            model = self.create_model(**params)
            model.fit(X_train, y_train)
            
            # Predict and calculate validation score
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            
            return mae
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {str(e)}")
            return float('inf')
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                            y_train: pd.Series, y_val: pd.Series) -> Tuple[Dict, Any]:
        """Tune hyperparameters using Optuna"""
        try:
            self.logger.info(f"Starting hyperparameter tuning for {self.model_name}")
            
            # Create study
            study = optuna.create_study(direction='minimize')
            
            # Optimize
            n_trials = self.config['tuning']['n_trials']
            timeout = self.config['tuning']['timeout_minutes'] * 60
            
            study.optimize(
                lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            
            self.logger.info(f"Best trial: {study.best_trial.number}")
            self.logger.info(f"Best value: {study.best_trial.value:.6f}")
            self.logger.info(f"Best params: {study.best_trial.params}")
            
            return study.best_trial.params, study
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
    
    def save_model_artifacts(self, model: Any, best_params: Dict, study: Any,
                           train_preds: np.ndarray, test_preds: np.ndarray,
                           y_train: pd.Series, y_test: pd.Series,
                           train_timestamps: pd.Series, test_timestamps: pd.Series,
                           paths: Dict[str, Path]):
        """Save all model artifacts"""
        try:
            # Save best parameters
            with open(paths['best_params'], 'w') as f:
                json.dump(best_params, f, indent=2)
            
            # Save best model
            joblib.dump(model, paths['best_model'])
            
            # Save tuning study
            joblib.dump(study, paths['tuning_study'])
            
            # Apply intelligent rounding to predictions
            # Since predictions are counts (non-negative integers), round to nearest integer
            # and ensure non-negative values
            train_preds_rounded = np.maximum(0, np.round(train_preds)).astype(int)
            test_preds_rounded = np.maximum(0, np.round(test_preds)).astype(int)
            
            # Save predictions with metadata
            train_pred_df = pd.DataFrame({
                'timestamp': train_timestamps,
                'true_count': y_train,
                'pred_count': train_preds_rounded
            })
            train_pred_df.to_csv(paths['predictions_train'], index=False)
            
            test_pred_df = pd.DataFrame({
                'timestamp': test_timestamps,
                'true_count': y_test,
                'pred_count': test_preds_rounded
            })
            test_pred_df.to_csv(paths['predictions_test'], index=False)
            
            # Calculate metrics using the original float predictions for accuracy
            train_metrics = self.calculate_metrics(y_train, train_preds)
            test_metrics = self.calculate_metrics(y_test, test_preds)
            
            # Save metadata
            metadata = {
                'model': self.model_name,
                'data_type': self.get_model_type(),
                'hyperparameters_path': str(paths['best_params']),
                'model_path': str(paths['best_model']),
                'training_timestamp': datetime.now().isoformat(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'best_params': best_params
            }
            
            with open(paths['metadata'], 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            self.logger.info(f"Model artifacts saved to {self.models_dir}")
            self.logger.info(f"Test metrics - MAE: {test_metrics['mae']:.4f}, "
                           f"RMSE: {test_metrics['rmse']:.4f}, MAPE: {test_metrics['mape']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error saving model artifacts: {str(e)}")
            raise
    
    def _test_existing_model(self, X_test, y_test, test_timestamps, paths):
        """Test existing trained model"""
        try:
            # Check if model exists
            if not paths['best_model'].exists():
                self.logger.error(f"No trained model found at {paths['best_model']}")
                return False
            
            # Load existing model
            model = joblib.load(paths['best_model'])
            self.logger.info(f"Loaded existing model from {paths['best_model']}")
            
            # Make predictions and apply rounding
            test_preds = model.predict(X_test)
            test_preds_rounded = np.maximum(0, np.round(test_preds)).astype(int)
            
            # Calculate and log metrics using original float predictions
            test_metrics = self.calculate_metrics(y_test, test_preds)
            
            self.logger.info(f"Test metrics - MAE: {test_metrics['mae']:.4f}, "
                           f"RMSE: {test_metrics['rmse']:.4f}, MAPE: {test_metrics['mape']:.4f}")
            
            # Save test predictions with rounded values
            test_pred_df = pd.DataFrame({
                'timestamp': test_timestamps,
                'true_count': y_test,
                'pred_count': test_preds_rounded
            })
            
            test_pred_path = paths['predictions_test'].parent / f"{self.model_name}_test_predictions_latest.csv"
            test_pred_df.to_csv(test_pred_path, index=False)
            
            self.logger.info(f"Test predictions saved to {test_pred_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error testing existing model: {str(e)}")
            return False
    
    def run(self, mode='train', force_retune=False) -> bool:
        """Main execution method
        
        Args:
            mode: 'train' for full training, 'test' for testing existing model only
            force_retune: If True, force hyperparameter retuning even if existing params found
        """
        try:
            self.logger.info(f"Starting {mode} process for {self.model_name}")
            
            # Load data
            train_df, val_df, test_df = self.load_data()
            
            # Prepare features and targets
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features_targets(
                train_df, val_df, test_df
            )
            
            # Get timestamps
            train_timestamps = train_df['timestamp'] if 'timestamp' in train_df.columns else pd.Series(range(len(train_df)))
            test_timestamps = test_df['timestamp'] if 'timestamp' in test_df.columns else pd.Series(range(len(test_df)))
            
            # Get model paths
            paths = self.get_model_paths(mode)
            
            if mode == 'test':
                # Test mode: load existing model and evaluate
                return self._test_existing_model(X_test, y_test, test_timestamps, paths)
            
            # Training mode: Check if we should skip tuning (model already exists)
            existing_params = None
            if not force_retune and self.config['orchestration']['skip_completed_models']:
                # Look for existing best params files
                existing_files = list(self.models_dir.glob(f"{self.model_name}_best_params_*.json"))
                if existing_files:
                    latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"Found existing model parameters: {latest_file}")
                    with open(latest_file, 'r') as f:
                        existing_params = json.load(f)
            
            # Tune hyperparameters or use existing
            if existing_params and not force_retune:
                self.logger.info("Using existing hyperparameters")
                best_params = existing_params
                study = None
            else:
                best_params, study = self.tune_hyperparameters(X_train, X_val, y_train, y_val)
            
            # Train final model on train + validation data
            X_final = pd.concat([X_train, X_val], ignore_index=True)
            y_final = pd.concat([y_train, y_val], ignore_index=True)
            
            final_model = self.create_model(**best_params)
            final_model.fit(X_final, y_final)
            
            # Make predictions
            train_preds = final_model.predict(X_train)
            test_preds = final_model.predict(X_test)
            
            # Save artifacts
            self.save_model_artifacts(
                final_model, best_params, study,
                train_preds, test_preds,
                y_train, y_test,
                train_timestamps, test_timestamps,
                paths
            )
            
            self.logger.info(f"Training process completed successfully for {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"{mode.capitalize()} process failed for {self.model_name}: {str(e)}")
            return False
