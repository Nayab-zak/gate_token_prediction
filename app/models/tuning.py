from __future__ import annotations
import optuna, numpy as np
from typing import Any, Callable, Dict, Tuple
from sklearn.metrics import mean_absolute_error

def time_series_objective_folds(X_df, y, build_preprocessor, num_cols, cat_cols, model_factory, param_space, splits):
    def suggest(trial: optuna.Trial):
        params = {}
        for k, v in param_space.items():
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                params[k] = trial.suggest_int(k, lo, hi)
            else:
                params[k] = trial.suggest_float(k, float(lo), float(hi))
        return params
    def _obj(trial: optuna.Trial):
        params = suggest(trial)
        maes = []
        for tr, va in splits:
            pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
            Xtr = pre.fit_transform(X_df.iloc[tr], y.iloc[tr])
            Xv  = pre.transform(X_df.iloc[va])
            model = model_factory(params)
            model.fit(Xtr, y.iloc[tr])
            pred = model.predict(Xv)
            mae = mean_absolute_error(y.iloc[va], pred)
            maes.append(mae)
        return float(np.mean(maes))
    return _obj
