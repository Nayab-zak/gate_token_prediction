from __future__ import annotations
import numpy as np
from typing import Dict

def mape(y, yhat): 
    y = np.asarray(y); yhat=np.asarray(yhat)
    denom = np.where(y==0, 1e-6, y)
    return float(np.mean(np.abs((y - yhat) / denom)) * 100.0)

def smape(y, yhat):
    y = np.asarray(y); yhat=np.asarray(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    denom = np.where(denom==0, 1e-6, denom)
    return float(np.mean(np.abs(y - yhat)) / np.mean(denom) * 100.0)

def percentile_err(y, yhat, p=50):
    y = np.asarray(y); yhat=np.asarray(yhat)
    e = y - yhat
    return float(np.percentile(np.abs(e), p))

def compute_metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y); yhat=np.asarray(yhat)
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    denom = np.where(y==0, 1e-6, y)
    mape = float(np.mean(np.abs((y - yhat) / denom)) * 100.0)
    smape = float(np.mean(2*np.abs(y - yhat)/(np.abs(y)+np.abs(yhat)+1e-6)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "P50": percentile_err(y,yhat,50), "P90": percentile_err(y,yhat,90)}
