from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import joblib, os

class BaseModel(ABC):
    name: str = "base"
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X): ...
    def predict_interval(self, X) -> Tuple[list[float], list[float]]:
        preds = self.predict(X)
        return [p*0.9 for p in preds], [p*1.1 for p in preds]
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
