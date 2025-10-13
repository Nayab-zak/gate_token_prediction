from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np

def conformal_from_residuals(residuals: np.ndarray, q_low: float, q_high: float) -> Tuple[float,float]:
    abs_err = np.abs(residuals)
    low_q = np.quantile(abs_err, q_high)
    high_q = np.quantile(abs_err, q_high)
    return float(low_q), float(high_q)

class QuantileLGBM:
    """Wrapper for LightGBM quantile regressor that preserves feature names"""
    def __init__(self, params: Dict[str, Any], alpha: float):
        import lightgbm as lgb
        qparams = params.copy()
        qparams.update(dict(objective="quantile", alpha=alpha, min_data_in_leaf=10))
        self.model = lgb.LGBMRegressor(**qparams)
    
    def fit(self, X, y):
        # Keep as DataFrame/Series to preserve feature names for LightGBM
        return self.model.fit(X, y)
    
    def predict(self, X):
        # Keep as DataFrame to preserve feature names for LightGBM
        return self.model.predict(X)

def make_quantile_lgbm(params: Dict[str, Any], alpha: float):
    return QuantileLGBM(params, alpha)
