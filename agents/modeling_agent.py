import pandas as pd
import numpy as np
import json
import os
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step
from typing import Optional
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lightgbm as lgb
import xgboost as xgb
import optuna

setup_logging()

class ModelingAgent:
    def __init__(self, df: pd.DataFrame, target: str, logger=None):
        self.df = df.copy()
        self.target = target
        self.logger = logger or get_logger('modeling_agent')
        self.models = {}
        self.best_params = {}
        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

    @track_step('train_baseline')
    def train_baseline(self):
        """Fit hourly mean and same-hour-last-week predictors."""
        df = self.df.copy()
        # Ensure MoveDate is datetime
        if 'MoveDate' in df.columns:
            df['MoveDate'] = pd.to_datetime(df['MoveDate'], errors='coerce')
        df['hour'] = df['MoveDate'].dt.hour
        df['dow'] = df['MoveDate'].dt.dayofweek
        # Hourly mean
        hourly_mean = df.groupby('hour')[self.target].mean()
        df['hourly_mean_pred'] = df['hour'].map(hourly_mean)
        # Same hour last week
        df = df.sort_values('MoveDate')
        df['last_week_pred'] = df[self.target].shift(24*7)
        self.models['baseline'] = {'hourly_mean': hourly_mean}
        self.logger.info('Trained baseline predictors.')
        return df[['hourly_mean_pred', 'last_week_pred']]

    @track_step('train_glms')
    def train_glms(self):
        """Fit Poisson and Negative Binomial GLMs via statsmodels."""
        df = self.df.copy()
        # Only use numeric and low-cardinality categorical features
        exclude_cols = [self.target, 'MoveDate']
        # Use only numeric columns and categoricals with <50 unique values
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns
                    if df[col].nunique() < 50 and col not in exclude_cols]
        features = [col for col in num_cols + cat_cols if col not in exclude_cols]
        if not features:
            raise ValueError('No suitable features for GLM.')
        formula = f"{self.target} ~ {' + '.join(features)}"
        poisson = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
        nb = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
        self.models['poisson'] = poisson
        self.models['neg_binom'] = nb
        self.logger.info(f'Trained Poisson and Negative Binomial GLMs with features: {features}')
        return poisson, nb

    @track_step('train_trees')
    def train_trees(self):
        """Fit LightGBM and XGBoost with TimeSeriesSplit CV."""
        # Only use numeric columns for X
        X = self.X.select_dtypes(include=['int', 'float', 'bool'])
        y = self.y
        tscv = TimeSeriesSplit(n_splits=5)
        lgbm = lgb.LGBMRegressor()
        xgbm = xgb.XGBRegressor()
        lgbm_scores = cross_val_score(lgbm, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        xgbm_scores = cross_val_score(xgbm, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        lgbm.fit(X, y)
        xgbm.fit(X, y)
        self.models['lgbm'] = lgbm
        self.models['xgbm'] = xgbm
        self.logger.info(f"Trained LightGBM (MAE: {-lgbm_scores.mean():.2f}), XGBoost (MAE: {-xgbm_scores.mean():.2f})")
        return lgbm, xgbm

    @track_step('tune_parameters')
    def tune_parameters(self, n_trials: int = 50):
        """Use Optuna to tune tree model hyperparameters, save best to models/best_params.json."""
        # Only use numeric columns for X
        X = self.X.select_dtypes(include=['int', 'float', 'bool'])
        y = self.y
        tscv = TimeSeriesSplit(n_splits=3)
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            }
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            return -scores.mean()
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        os.makedirs('models', exist_ok=True)
        with open('models/best_params.json', 'w') as f:
            json.dump(self.best_params, f)
        self.logger.info(f"Best params saved: {self.best_params}")
        return self.best_params

    @track_step('evaluate')
    def evaluate(self):
        """Compute MAE, RMSE, deviance on CV and hold-out, logging results."""
        X = self.X
        y = self.y
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                maes, rmses = [], []
                for train_idx, test_idx in tscv.split(X):
                    model.fit(X.iloc[train_idx], y.iloc[train_idx])
                    preds = model.predict(X.iloc[test_idx])
                    maes.append(mean_absolute_error(y.iloc[test_idx], preds))
                    rmses.append(mean_squared_error(y.iloc[test_idx], preds, squared=False))
                results[name] = {'MAE': np.mean(maes), 'RMSE': np.mean(rmses)}
            elif hasattr(model, 'fittedvalues'):
                preds = model.fittedvalues
                mae = mean_absolute_error(y, preds)
                rmse = mean_squared_error(y, preds, squared=False)
                results[name] = {'MAE': mae, 'RMSE': rmse}
        self.logger.info(f"Evaluation results: {results}")
        return results

    def get_feature_names(self):
        # Only use numeric columns for tree models
        return list(self.X.select_dtypes(include=['int', 'float', 'bool']).columns)
