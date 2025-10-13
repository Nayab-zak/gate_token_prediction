from __future__ import annotations
from app.models.base import BaseModel
import xgboost as xgb

class XGBoostModel(BaseModel):
    name = "xgboost"
    def __init__(self, **params): self.model = xgb.XGBRegressor(**params)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X).tolist()
