from __future__ import annotations
from app.models.base import BaseModel
import lightgbm as lgb
import numpy as np

class LightGBMModel(BaseModel):
    name = "lightgbm"
    def __init__(self, **params): 
        # Add default parameters to prevent warnings
        default_params = {
            'verbosity': -1,  # Reduce LightGBM output
            'min_data_in_leaf': 20,  # Minimum samples per leaf
            'min_gain_to_split': 0.0,  # Minimum gain to make a split
            'feature_fraction': 0.9,  # Use 90% of features
        }
        default_params.update(params)
        self.model = lgb.LGBMRegressor(**default_params)
    def fit(self, X, y): 
        # Keep as DataFrame/Series to preserve feature names for LightGBM
        self.model.fit(X, y)
    def predict(self, X): 
        # Keep as DataFrame to preserve feature names for LightGBM
        return self.model.predict(X).tolist()
