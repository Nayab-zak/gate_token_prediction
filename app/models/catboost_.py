from __future__ import annotations
from app.models.base import BaseModel
from catboost import CatBoostRegressor

class CatBoostModel(BaseModel):
    name = "catboost"
    def __init__(self, **params): self.model = CatBoostRegressor(verbose=False, **params)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X).tolist()
