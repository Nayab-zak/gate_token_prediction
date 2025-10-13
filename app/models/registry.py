from __future__ import annotations
from typing import Any
from sklearn.linear_model import ElasticNet
from app.models.base import BaseModel

class ElasticNetWrapper(BaseModel):
    name = "elasticnet"
    def __init__(self, **p): 
        self.model = ElasticNet(**p)
    def fit(self, X, y): 
        self.model.fit(X, y)
    def predict(self, X): 
        return self.model.predict(X).tolist()

def get_model(name: str, **params: Any):
    name = name.lower()
    if name == "lightgbm":
        from app.models.lightgbm_ import LightGBMModel
        return LightGBMModel(**params)
    if name == "xgboost":
        from app.models.xgboost_ import XGBoostModel
        return XGBoostModel(**params)
    if name == "catboost":
        from app.models.catboost_ import CatBoostModel
        return CatBoostModel(**params)
    if name == "elasticnet":
        return ElasticNetWrapper(**params)
    raise ValueError(f"Unknown model {name}")
