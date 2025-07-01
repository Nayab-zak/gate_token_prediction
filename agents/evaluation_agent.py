import sys
import warnings
import os
# Set LightGBM log level as early as possible
os.environ['LGBM_LOG_LEVEL'] = '0'
os.environ['LIGHTGBM_LOG_LEVEL'] = '0'
os.environ['LGBM_VERBOSE'] = '0'
os.environ['LIGHTGBM_VERBOSE'] = '0'
# Suppress all warnings
warnings.filterwarnings('ignore')

# --- Robust suppression of LightGBM C++ warnings ---
import tempfile
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Redirects process-level stderr to /dev/null (for C++ warnings)."""
    fd = sys.stderr.fileno()
    # Save a copy of the original stderr
    old_stderr = os.dup(fd)
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, fd)
            os.close(old_stderr)

with suppress_stderr():
    import lightgbm as lgb
lgb.basic._log_warning = lambda *a, **k: None  # Suppress LightGBM C++ warnings

import pandas as pd
import numpy as np
from typing import List
from utils.logger import setup_logging, get_logger
from utils.footsteps import track_step
import matplotlib.pyplot as plt

setup_logging()

class EvaluationAgent:
    def __init__(self, model, df_holdout: pd.DataFrame, logger=None, feature_names=None):
        self.model = model
        self.df_holdout = df_holdout.copy()
        self.logger = logger or get_logger('evaluation_agent')
        # Save feature names if provided, else try to infer from model
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(model, 'feature_names_in_'):
            self.feature_names = list(model.feature_names_in_)
        else:
            self.feature_names = None

    def _ensure_numeric_features(self, X):
        # Convert object columns to category if possible, else drop or encode
        for col in X.select_dtypes(include=['object']).columns:
            # If few unique values, treat as category
            if X[col].nunique() < 100:
                X[col] = X[col].astype('category')
            else:
                # Drop high-cardinality object columns
                X = X.drop(columns=[col])
        return X

    @track_step('score_holdout')
    def score_holdout(self) -> dict:
        """
        Score model on the last 3 months of holdout data, returning metrics.
        """
        df = self.df_holdout.copy()
        if 'MoveDate' in df.columns:
            last_date = df['MoveDate'].max()
            cutoff = last_date - pd.DateOffset(months=3)
            df = df[df['MoveDate'] >= cutoff]
        y_true = df['TokenCount']
        X = df.drop(columns=['TokenCount', 'MoveDate'], errors='ignore')
        # Align columns to match model features exactly (add missing as 0, drop extras)
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0  # or np.nan, or a default value
            X = X[self.feature_names]
        X = self._ensure_numeric_features(X)
        y_pred = self.model.predict(X)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
        self.logger.info(f"Holdout MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2%}")
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    @track_step('score_slice')
    def score_slice(self, by: List[str] = ['is_holiday','is_weekend']) -> pd.DataFrame:
        """
        Return a breakdown DataFrame of metrics by specified columns.
        """
        df = self.df_holdout.copy()
        y_true = df['TokenCount']
        X = df.drop(columns=['TokenCount', 'MoveDate'], errors='ignore')
        # Align columns to match model features exactly (add missing as 0, drop extras)
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0  # or np.nan, or a default value
            X = X[self.feature_names]
        X = self._ensure_numeric_features(X)
        y_pred = self.model.predict(X)
        df['y_true'] = y_true
        df['y_pred'] = y_pred
        results = []
        for keys, group in df.groupby(by):
            mae = np.mean(np.abs(group['y_true'] - group['y_pred']))
            rmse = np.sqrt(np.mean((group['y_true'] - group['y_pred']) ** 2))
            results.append(dict(zip(by, keys if isinstance(keys, tuple) else [keys])))
            results[-1].update({'MAE': mae, 'RMSE': rmse, 'count': len(group)})
        breakdown = pd.DataFrame(results)
        self.logger.info(f"Slice breakdown by {by}:\n{breakdown}")
        return breakdown

    @track_step('rolling_test')
    def rolling_test(self, window: int = 168, step: int = 24) -> pd.DataFrame:
        """
        Simulate retrain+forecast with a rolling window, returning stability results.
        """
        df = self.df_holdout.copy().sort_values('MoveDate')
        results = []
        for start in range(0, len(df) - window, step):
            train = df.iloc[start:start+window]
            test = df.iloc[start+window:start+window+step]
            if len(test) == 0:
                break
            X_train = train.drop(columns=['TokenCount', 'MoveDate'], errors='ignore')
            y_train = train['TokenCount']
            X_test = test.drop(columns=['TokenCount', 'MoveDate'], errors='ignore')
            y_test = test['TokenCount']
            # Align columns to match model features exactly (add missing as 0, drop extras)
            if self.feature_names is not None:
                for col in self.feature_names:
                    if col not in X_train.columns:
                        X_train[col] = 0
                    if col not in X_test.columns:
                        X_test[col] = 0
                X_train = X_train[self.feature_names]
                X_test = X_test[self.feature_names]
            X_train = self._ensure_numeric_features(X_train)
            X_test = self._ensure_numeric_features(X_test)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            results.append({'start': train['MoveDate'].min(), 'end': test['MoveDate'].max(), 'MAE': mae, 'RMSE': rmse})
        res_df = pd.DataFrame(results)
        self.logger.info(f"Rolling test results:\n{res_df}")
        return res_df

    @track_step('psi')
    def psi(self, features: List[str], threshold: float = 0.2) -> pd.DataFrame:
        """
        Compute Population Stability Index (PSI) month-over-month for given features.
        """
        df = self.df_holdout.copy()
        df['month'] = df['MoveDate'].dt.to_period('M')
        psi_results = []
        for feature in features:
            prev_dist = None
            for month, group in df.groupby('month'):
                dist = group[feature].value_counts(normalize=True)
                if prev_dist is not None:
                    psi = np.sum((dist - prev_dist) * np.log((dist + 1e-8) / (prev_dist + 1e-8)))
                    psi_results.append({'feature': feature, 'month': month, 'psi': psi})
                prev_dist = dist
        psi_df = pd.DataFrame(psi_results)
        flagged = psi_df[psi_df['psi'] > threshold]
        self.logger.info(f"PSI flagged features:\n{flagged}")
        return psi_df

    @track_step('report')
    def report(self, path: str = 'reports/evaluation_report.md'):
        """
        Write a summary report with plots and tables.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metrics = self.score_holdout()
        psi_df = self.psi(features=['TokenCount','ContainerCount'])
        with open(path, 'w') as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"## Holdout Metrics\n{metrics}\n\n")
            f.write(f"## PSI\n{psi_df.to_markdown()}\n\n")
        # Example plot
        plt.figure(figsize=(8,4))
        self.df_holdout['TokenCount'].plot(label='Actual')
        # Align columns to match model features exactly (add missing as 0, drop extras)
        if self.feature_names is not None:
            X = self.df_holdout.drop(columns=['TokenCount', 'MoveDate'], errors='ignore').copy()
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        else:
            X = self.df_holdout.drop(columns=['TokenCount', 'MoveDate'], errors='ignore')
        y_pred = self.model.predict(X)
        plt.plot(self.df_holdout['MoveDate'], y_pred, label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted TokenCount')
        plt.tight_layout()
        plot_path = path.replace('.md', '.png')
        plt.savefig(plot_path)
        self.logger.info(f"Report written to {path} and plot saved to {plot_path}")
