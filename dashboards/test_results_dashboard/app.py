import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from config import TEST_SPLIT_MONTHS, FORECAST_HORIZON, TOP_IF_SLIDER

# ─── Page Setup & Theme ───────────────────────────────────────────────────────
st.set_page_config(page_title="Test Results Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .main {background-color: #FFFFFF; color: #000000;}
      .sidebar .sidebar-content {background-color: #F0F2F6; color: #000000;}
      h1, h2, h3, h4 {color: #000000;}
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Locate dashboard CSVs ─────────────────────────────────────────────────────
# Correct BASE_DIR: three levels up to project root, then data/dashboard_data
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
BASE_DIR = os.path.join(project_root, "data", "dashboard_data")
print(f"Looking for test results in: {BASE_DIR}")
os.makedirs(BASE_DIR, exist_ok=True)
all_files = [f for f in os.listdir(BASE_DIR) if f.endswith("_test_results.csv")]

# Debug: show what files we found
st.sidebar.write("Found dashboard_data files:", all_files)

# Build a raw_df to inspect full date range
raw_dfs = []
for fname in all_files:
    path = os.path.join(BASE_DIR, fname)
    try:
        tmp = pd.read_csv(path, parse_dates=["datetime"] )
        raw_dfs.append(tmp.assign(model=fname.replace("_test_results.csv","")))
    except Exception:
        pass

if raw_dfs:
    raw_df = pd.concat(raw_dfs, ignore_index=True)
    min_dt = raw_df["datetime"].min().date()
    max_dt = raw_df["datetime"].max().date()
    st.sidebar.write(f"Data covers: {min_dt} → {max_dt}")
else:
    today = datetime.today().date()
    min_dt = today - timedelta(days=30*TEST_SPLIT_MONTHS)
    max_dt = today
    st.sidebar.write("No data found; using default date range.")

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.header("Controls")

# Date pickers defaulted to actual data range
start_date = st.sidebar.date_input("Start Date", value=min_dt, min_value=min_dt, max_value=max_dt)
end_date   = st.sidebar.date_input("End Date",   value=max_dt, min_value=min_dt, max_value=max_dt)

# Model selector
model_keys = ["All models"] + [f.replace("_test_results.csv","") for f in all_files]
selected_model = st.sidebar.selectbox("Select Model", model_keys, key="model_picker")

# Metric toggles
metrics = ["rmse", "mae", "mape"]
selected_metrics = st.sidebar.multiselect("Metrics", metrics, default=metrics)

# Forecast horizon slider
horizon = st.sidebar.slider("Forecast Horizon (hrs)", 1, FORECAST_HORIZON, FORECAST_HORIZON)

# What‑If sliders
st.sidebar.markdown("---")
st.sidebar.write("### What‑If Sliders")
what_if = {feat: st.sidebar.slider(feat, 0.0, 1.0, 0.5) for feat in TOP_IF_SLIDER}

# ─── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_results(model_key, start_dt, end_dt):
    df = pd.DataFrame()
    files = all_files if model_key == "All models" else [f"{model_key}_test_results.csv"]

    for fname in files:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            tmp = pd.read_csv(path, parse_dates=["datetime"])
            tmp["model"] = fname.replace("_test_results.csv","")
            df = pd.concat([df, tmp], ignore_index=True)

    # Date filter
    if not df.empty and "datetime" in df.columns:
        mask = (
            (df["datetime"].dt.date >= start_dt)
            & (df["datetime"].dt.date <= end_dt)
        )
        df = df.loc[mask]

    # Guarantee model column exists
    if "model" not in df.columns:
        df["model"] = pd.NA

    return df


df = load_results(selected_model, start_date, end_date)

# ─── Dashboard Title ─────────────────────────────────────────────────────────
st.title("Test Results Dashboard")

# ─── Alerts for Metric Spikes ────────────────────────────────────────────────
if not df.empty and "model" in df.columns:
    for m in selected_metrics:
        if m not in df.columns:
            continue
        thresh = df[m].mean() + 2 * df[m].std()
        if selected_model == "All models":
            val = df[m].max()
        else:
            subset = df[df["model"] == selected_model]
            val = subset[m].max() if not subset.empty else None
        if val is not None and val > thresh:
            st.warning(f"{m.upper()} = {val:.2f} exceeds threshold {thresh:.2f}")
else:
    st.info("No data available for the selected date/model range.")

st.markdown("---")

# ─── Metric Cards & Sparklines ────────────────────────────────────────────────
if not df.empty and "model" in df.columns:
    cols = st.columns(len(selected_metrics))
    for idx, m in enumerate(selected_metrics):
        with cols[idx]:
            if m not in df.columns:
                st.warning(f"Metric {m} not available")
                continue

            if selected_model == "All models":
                label = f"Max {m.upper()}"
                value = df[m].max()
                chart_data = df[m]
            else:
                subset = df[df["model"] == selected_model]
                label = m.upper()
                value = subset[m].max() if not subset.empty else float("nan")
                chart_data = subset[m]

            st.metric(label=label, value=f"{value:.2f}")
            st.line_chart(chart_data)
st.markdown("---")

# ─── Residual Distribution ────────────────────────────────────────────────────
st.subheader("Residual Distribution")
if not df.empty and "model" in df.columns:
    fig, ax = plt.subplots()
    for name, grp in df.groupby("model"):
        ax.hist(grp["error"], bins=50, alpha=0.5, label=name)
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("No residuals to display.")

st.markdown("---")

# ─── Error Heatmap ────────────────────────────────────────────────────────────
st.subheader("Error Heatmap (Hour vs Weekday")
if not df.empty:
    df["hour"]    = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    heat = df.pivot_table(
        index="weekday", columns="hour", values="error", aggfunc="mean"
    )
    if not heat.empty and not heat.isnull().all().all():
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(heat, cmap="RdBu_r", center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available to render error heatmap.")
else:
    st.warning("No data available for heatmap.")

st.markdown("---")

# ─── Feature Importance (SHAP) ────────────────────────────────────────────────
st.subheader("Feature Importance (SHAP Summary)")
st.write("Global SHAP summary plot here")

st.markdown("---")

# ─── Predictions Table ────────────────────────────────────────────────────────
st.subheader("Predictions Table")
if not df.empty:
    st.dataframe(
        df[["datetime","model","actual","prediction","error"]]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
else:
    st.warning("No data to display in the predictions table.")

st.markdown("---")
st.write("End of Test Results Dashboard")
