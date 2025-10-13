from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureNamePreservingTransformer:
    """Wrapper around ColumnTransformer that preserves feature names for LightGBM compatibility"""
    
    def __init__(self, numeric_cols, categorical_cols, scale_numeric=False):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scale_numeric = scale_numeric
        
        num = ("num",
               StandardScaler(with_mean=True, with_std=True) if scale_numeric else "passthrough",
               numeric_cols)
        cat = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        self.transformer = ColumnTransformer(transformers=[num, cat], remainder="drop")
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        # Generate feature names after fitting
        self._generate_feature_names()
        return self
    
    def transform(self, X):
        # Transform to numpy array
        X_transformed = self.transformer.transform(X)
        # Convert back to DataFrame with proper feature names for LightGBM
        return pd.DataFrame(X_transformed, columns=self.feature_names_, index=X.index)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def _generate_feature_names(self):
        """Generate feature names from the fitted transformer"""
        feature_names = []
        
        # Get numeric feature names
        if self.scale_numeric:
            feature_names.extend(self.numeric_cols)
        else:
            feature_names.extend(self.numeric_cols)
        
        # Get categorical feature names (after one-hot encoding)
        if self.categorical_cols:
            cat_transformer = self.transformer.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_names = cat_transformer.get_feature_names_out(self.categorical_cols)
                feature_names.extend(cat_names)
            else:
                # Fallback for older sklearn versions
                for col in self.categorical_cols:
                    for category in cat_transformer.categories_[self.categorical_cols.index(col)]:
                        feature_names.append(f"{col}_{category}")
        
        self.feature_names_ = feature_names

def build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False):
    """Build a preprocessor that preserves feature names for LightGBM compatibility"""
    return FeatureNamePreservingTransformer(numeric_cols, categorical_cols, scale_numeric)
