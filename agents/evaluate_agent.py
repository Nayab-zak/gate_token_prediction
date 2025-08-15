from __future__ import annotations
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import settings
from utils.io import get_output_path

def _read_preds(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found: {p}")
    return pd.read_csv(p)

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    # Build a plotting timestamp from MoveDate_pred + MoveHour_pred (local-style; we won’t tz-convert)
    if "MoveDate_pred" not in df.columns or "MoveHour_pred" not in df.columns:
        raise ValueError("Preds must contain MoveDate_pred and MoveHour_pred")
    base = pd.to_datetime(df["MoveDate_pred"], errors="coerce")
    hour = pd.to_numeric(df["MoveHour_pred"], errors="coerce").fillna(0).astype(int)
    df = df.copy()
    df["ts_pred"] = base.dt.floor("D") + pd.to_timedelta(hour, unit="h")
    return df

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.abs(y_true).sum()
    wape = float(np.nan) if denom == 0 else float(np.abs(y_true - y_pred).sum() / denom)
    denom2 = (np.abs(y_true) + np.abs(y_pred))
    denom2 = np.where(denom2 == 0, 1.0, denom2)
    smape = float((2.0 * np.abs(y_pred - y_true) / denom2).mean())
    return {"RMSE": rmse, "MAE": mae, "WAPE": wape, "SMAPE": smape}

def _pick_series(df: pd.DataFrame, t=None, m=None, d=None) -> tuple[str,str,str]:
    if t and m and d:
        return t, m, d
    # pick busiest series by true volume
    if "TokenCount_true" in df.columns:
        grp = df.groupby(["TerminalID","MoveType","Desig"], dropna=False)["TokenCount_true"].sum()
    else:
        grp = df.groupby(["TerminalID","MoveType","Desig"], dropna=False)["TokenCount_pred"].sum()
    return grp.sort_values(ascending=False).index[0]

def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=settings.EVAL_PRED_PATH)
    ap.add_argument("--terminal", default=settings.EVAL_FILTER_TERMINAL or "")
    ap.add_argument("--movetype", default=settings.EVAL_FILTER_MOVETYPE or "")
    ap.add_argument("--desig", default=settings.EVAL_FILTER_DESIG or "")
    args = ap.parse_args()

    preds = _read_preds(args.preds)
    preds = _ensure_ts(preds)

    # sanity: need ground truth for proper eval
    have_truth = "TokenCount_true" in preds.columns
    if not have_truth:
        raise ValueError("TokenCount_true missing in predictions file. Re-run training to produce test truth, or point to predictions_valid.csv.")

    # overall metrics
    metrics = _metrics(preds["TokenCount_true"].values, preds["TokenCount_pred"].values)

    # scatter: Actual vs Predicted
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(preds["TokenCount_true"].values, preds["TokenCount_pred"].values, s=8, alpha=0.6)
    lim = max(float(preds["TokenCount_true"].max()), float(preds["TokenCount_pred"].max()))
    ax.plot([0, lim], [0, lim])
    ax.set_xlabel("Actual (TokenCount)")
    ax.set_ylabel("Predicted (TokenCount)")
    ax.set_title("Actual vs Predicted (Test)")

    report_dir = Path(settings.EVAL_REPORT_DIR)
    
    # Save scatter plot with file replacement
    scatter_path = get_output_path(
        base_dir=report_dir,
        filename="scatter_actual_vs_pred.png",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="scatter_actual_vs_pred*.png"
    )
    _save(fig, scatter_path)

    # pick series for time plot
    t, m, d = _pick_series(preds, args.terminal or None, args.movetype or None, args.desig or None)
    series = preds[(preds["TerminalID"] == t) & (preds["MoveType"] == m) & (preds["Desig"] == d)].copy()
    series = series.sort_values("ts_pred")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(series["ts_pred"], series["TokenCount_true"], label="Actual")
    ax2.plot(series["ts_pred"], series["TokenCount_pred"], label="Predicted")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("TokenCount")
    ax2.set_title(f"Time series — {t}/{m}/{d}")
    ax2.legend()

    # Save timeseries plot with file replacement
    timeseries_path = get_output_path(
        base_dir=report_dir,
        filename=f"timeseries_{t}_{m}_{d}.png",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern=f"timeseries_{t}_{m}_{d}*.png"
    )
    _save(fig2, timeseries_path)

    # save metrics with file replacement
    metrics_path = get_output_path(
        base_dir=report_dir,
        filename="metrics_test.json",
        replace_files=settings.REPLACE_INTERMEDIATE_FILES,
        keep_last_n=settings.KEEP_LAST_N_VERSIONS,
        cleanup_pattern="metrics_test*.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"overall": metrics, "series": {"TerminalID": t, "MoveType": m, "Desig": d}}, f, indent=2, default=str)

    print("✅ Eval done")
    print(f"Metrics  → {metrics_path}")
    print(f"Scatter  → {scatter_path}")
    print(f"Series   → {timeseries_path}")

if __name__ == "__main__":
    main()
