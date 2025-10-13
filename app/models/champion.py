from __future__ import annotations
import os, json
from typing import Dict, Any

def choose_champion(results: list[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    primary = policy.get("primary_metric", "MAE")
    lower = policy.get("lower_is_better", True)
    def key(r):
        m = r["metrics_cv"][primary]
        return m if lower else -m
    return sorted(results, key=key)[0]

def persist_champion(artifact_dir: str, model_name: str, model_obj, preprocessor, bundle_meta: Dict[str, Any]):
    champ_dir = os.path.join(artifact_dir, "champion")
    os.makedirs(champ_dir, exist_ok=True)
    import joblib
    joblib.dump(model_obj, os.path.join(champ_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(champ_dir, "preprocessor.joblib"))
    with open(os.path.join(champ_dir, "run_meta.json"), "w") as f:
        json.dump(bundle_meta, f, indent=2)
    for k in ["params_optimized", "metrics_cv", "metrics_backtest", "columns_used", "feature_spec", "data_contract", "intervals"]:
        if k in bundle_meta:
            with open(os.path.join(champ_dir, f"{k}.json"), "w") as f:
                json.dump(bundle_meta[k], f, indent=2)
