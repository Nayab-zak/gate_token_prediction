import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from glob import glob
import warnings
import pandas as pd
warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from agents.config import REPORTS_DIR, DASHBOARD_DATA_PATH
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000  # Fix OverflowError for large plots

st.set_page_config(page_title="Experiment Plots Dashboard", layout="wide")
st.title("Experiment Plots Dashboard")

REPORTS_DIR = REPORTS_DIR
DATA_PATH = DASHBOARD_DATA_PATH

# Find all experiment subfolders (exp1, exp2, etc.)
exp_dirs = [d for d in os.listdir(REPORTS_DIR) if os.path.isdir(os.path.join(REPORTS_DIR, d))]
if not exp_dirs:
    exp_dirs = ["."]
exp_map = {exp: os.path.join(REPORTS_DIR, exp) if exp != "." else REPORTS_DIR for exp in exp_dirs}

selected_exp = st.selectbox("Select Experiment", exp_map.keys())
exp_path = exp_map[selected_exp]

# Load experiment predictions for real-time filtering
pred_path = os.path.join(exp_path, "predictions.csv")
if os.path.exists(pred_path):
    df = pd.read_csv(pred_path, parse_dates=['MoveDate'])
else:
    st.error(f"Predictions file not found: {pred_path}")
    st.stop()

# Filter options
if 'year' in df.columns:
    years = sorted(df['year'].dropna().astype(int).unique())
    selected_years = st.multiselect("Filter by Year", years, default=years)
    df = df[df['year'].isin(selected_years)]
if 'month' in df.columns:
    months = sorted(df['month'].dropna().astype(int).unique())
    selected_months = st.multiselect("Filter by Month", months, default=months)
    df = df[df['month'].isin(selected_months)]

# Tabs for each plot type
plot_tabs = st.tabs([
    "Actual vs Predicted (Time Series)",
    "Actual vs Predicted (Scatter)",
    "Residuals Over Time",
    "Residuals Histogram",
    "MAE/RMSE by Month",
    "Boxplot by Month",
    "Feature Importance"
])

# 1. Actual vs Predicted (Time Series)
with plot_tabs[0]:
    st.subheader("Actual vs Predicted TokenCount (Time Series)")
    if 'TokenCount' in df.columns and 'y_pred' in df.columns:
        plt.figure(figsize=(12,4))
        plt.plot(df['MoveDate'], df['TokenCount'], label='Actual', alpha=0.7)
        plt.plot(df['MoveDate'], df['y_pred'], label='Predicted', alpha=0.7)
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('TokenCount')
        plt.title('Actual vs Predicted TokenCount (Filtered)')
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("TokenCount or y_pred column not found in data.")

# 2. Actual vs Predicted (Scatter)
with plot_tabs[1]:
    st.subheader("Actual vs Predicted TokenCount (Scatter)")
    if 'y_pred' in df.columns:
        plt.figure(figsize=(6,6))
        plt.scatter(df['TokenCount'], df['y_pred'], alpha=0.3)
        plt.plot([df['TokenCount'].min(), df['TokenCount'].max()], [df['TokenCount'].min(), df['TokenCount'].max()], 'r--', label='Ideal')
        plt.xlabel('Actual TokenCount')
        plt.ylabel('Predicted TokenCount')
        plt.title('Actual vs Predicted (Filtered)')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("No predictions available for scatter plot.")

# 3. Residuals Over Time
with plot_tabs[2]:
    st.subheader("Residuals Over Time")
    if 'y_pred' in df.columns:
        residuals = df['TokenCount'] - df['y_pred']
        plt.figure(figsize=(12,4))
        plt.plot(df['MoveDate'], residuals, label='Residuals', alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', label='Zero Error')
        plt.xlabel('Date')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residuals Over Time (Filtered)')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("No predictions available for residuals plot.")

# 4. Residuals Histogram
with plot_tabs[3]:
    st.subheader("Distribution of Residuals")
    if 'y_pred' in df.columns:
        residuals = df['TokenCount'] - df['y_pred']
        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=50, alpha=0.7, label='Residuals')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.xlabel('Residual (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals (Filtered)')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("No predictions available for residuals histogram.")

# 5. MAE/RMSE by Month
with plot_tabs[4]:
    st.subheader("MAE and RMSE by Month")
    if 'year' in df.columns and 'month' in df.columns and 'y_pred' in df.columns:
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        grouped = df.groupby('year_month', group_keys=False).apply(lambda g: pd.Series({
            'MAE': (g['TokenCount'] - g['y_pred']).abs().mean(),
            'RMSE': ((g['TokenCount'] - g['y_pred'])**2).mean()**0.5
        }), include_groups=False)
        plt.figure(figsize=(10,4))
        plt.plot(grouped.index, grouped['MAE'], label='MAE')
        plt.plot(grouped.index, grouped['RMSE'], label='RMSE')
        plt.xlabel('Year-Month')
        plt.ylabel('Error')
        plt.title('MAE and RMSE by Month (Filtered)')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("No predictions or date columns available for MAE/RMSE plot.")

# 6. Boxplot by Month
with plot_tabs[5]:
    st.subheader("Actual vs Predicted by Month (Boxplot)")
    if 'year' in df.columns and 'month' in df.columns and 'y_pred' in df.columns:
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        plt.figure(figsize=(14,6))
        # Remove 'label' argument from sns.boxplot calls to avoid TypeError
        sns.boxplot(x='year_month', y='TokenCount', data=df, color='skyblue', showfliers=False, width=0.5, boxprops=dict(alpha=.5))
        sns.boxplot(x='year_month', y='y_pred', data=df, color='orange', showfliers=False, width=0.3, boxprops=dict(alpha=.5))
        plt.title('Actual vs Predicted TokenCount by Month (Filtered)')
        plt.xlabel('Year-Month')
        plt.ylabel('TokenCount')
        plt.xticks(rotation=45)
        plt.legend(['Actual', 'Predicted'])
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("No predictions or date columns available for boxplot.")

# 7. Feature Importance
with plot_tabs[6]:
    st.subheader("Feature Importances")
    imp_path = os.path.join(exp_path, 'feature_importance.png')
    if os.path.exists(imp_path):
        st.image(imp_path, caption="Feature Importances (static)", use_container_width=True, output_format='PNG')
    else:
        st.info("Feature importance plot not found for this experiment.")

# Optionally: allow side-by-side comparison (static images only)
enable_compare = st.checkbox("Compare experiments side by side")
if enable_compare and len(exp_dirs) > 1:
    exp1, exp2 = st.selectbox("Experiment 1", exp_dirs, key="exp1"), st.selectbox("Experiment 2", exp_dirs, key="exp2")
    col1, col2 = st.columns(2)
    # Try to load predictions.csv for each experiment
    pred_path1 = os.path.join(REPORTS_DIR, exp1, "predictions.csv")
    pred_path2 = os.path.join(REPORTS_DIR, exp2, "predictions.csv")
    df1, df2 = None, None
    if os.path.exists(pred_path1):
        df1 = pd.read_csv(pred_path1, parse_dates=['MoveDate'])
    else:
        with col1:
            st.warning(f"predictions.csv not found for {exp1}. Please ensure per-experiment predictions are saved.")
    if os.path.exists(pred_path2):
        df2 = pd.read_csv(pred_path2, parse_dates=['MoveDate'])
    else:
        with col2:
            st.warning(f"predictions.csv not found for {exp2}. Please ensure per-experiment predictions are saved.")
    # If both loaded, apply year/month filters and plot side by side
    if df1 is not None and df2 is not None:
        # Filter by year/month if columns exist
        if 'year' in df1.columns and 'month' in df1.columns and 'selected_years' in locals() and 'selected_months' in locals():
            df1 = df1[df1['year'].isin(selected_years)]
            df1 = df1[df1['month'].isin(selected_months)]
        if 'year' in df2.columns and 'month' in df2.columns and 'selected_years' in locals() and 'selected_months' in locals():
            df2 = df2[df2['year'].isin(selected_years)]
            df2 = df2[df2['month'].isin(selected_months)]
        # Define a helper to plot all main plots for a given df/col
        def plot_all(df, col, exp_name):
            with col:
                st.markdown(f"### {exp_name}")
                # 1. Actual vs Predicted (Time Series)
                st.write("Actual vs Predicted (Time Series)")
                if 'TokenCount' in df.columns and 'y_pred' in df.columns:
                    plt.figure(figsize=(12,4))
                    plt.plot(df['MoveDate'], df['TokenCount'], label='Actual', alpha=0.7)
                    plt.plot(df['MoveDate'], df['y_pred'], label='Predicted', alpha=0.7)
                    plt.legend()
                    plt.xlabel('Date')
                    plt.ylabel('TokenCount')
                    plt.title('Actual vs Predicted TokenCount (Filtered)')
                    st.pyplot(plt.gcf())
                    plt.close()
                # 2. Actual vs Predicted (Scatter)
                st.write("Actual vs Predicted (Scatter)")
                if 'y_pred' in df.columns:
                    plt.figure(figsize=(6,6))
                    plt.scatter(df['TokenCount'], df['y_pred'], alpha=0.3)
                    plt.plot([df['TokenCount'].min(), df['TokenCount'].max()], [df['TokenCount'].min(), df['TokenCount'].max()], 'r--', label='Ideal')
                    plt.xlabel('Actual TokenCount')
                    plt.ylabel('Predicted TokenCount')
                    plt.title('Actual vs Predicted (Filtered)')
                    plt.legend()
                    st.pyplot(plt.gcf())
                    plt.close()
                # 3. Residuals Over Time
                st.write("Residuals Over Time")
                if 'y_pred' in df.columns:
                    residuals = df['TokenCount'] - df['y_pred']
                    plt.figure(figsize=(12,4))
                    plt.plot(df['MoveDate'], residuals, label='Residuals', alpha=0.7)
                    plt.axhline(0, color='red', linestyle='--', label='Zero Error')
                    plt.xlabel('Date')
                    plt.ylabel('Residual (Actual - Predicted)')
                    plt.title('Residuals Over Time (Filtered)')
                    plt.legend()
                    st.pyplot(plt.gcf())
                    plt.close()
                # 4. Residuals Histogram
                st.write("Residuals Histogram")
                if 'y_pred' in df.columns:
                    residuals = df['TokenCount'] - df['y_pred']
                    plt.figure(figsize=(6,4))
                    plt.hist(residuals, bins=50, alpha=0.7, label='Residuals')
                    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
                    plt.xlabel('Residual (Actual - Predicted)')
                    plt.ylabel('Frequency')
                    plt.title('Distribution of Residuals (Filtered)')
                    plt.legend()
                    st.pyplot(plt.gcf())
                    plt.close()
                # 5. MAE/RMSE by Month
                st.write("MAE and RMSE by Month")
                if 'year' in df.columns and 'month' in df.columns and 'y_pred' in df.columns:
                    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
                    grouped = df.groupby('year_month', group_keys=False).apply(lambda g: pd.Series({
                        'MAE': (g['TokenCount'] - g['y_pred']).abs().mean(),
                        'RMSE': ((g['TokenCount'] - g['y_pred'])**2).mean()**0.5
                    }), include_groups=False)
                    plt.figure(figsize=(10,4))
                    plt.plot(grouped.index, grouped['MAE'], label='MAE')
                    plt.plot(grouped.index, grouped['RMSE'], label='RMSE')
                    plt.xlabel('Year-Month')
                    plt.ylabel('Error')
                    plt.title('MAE and RMSE by Month (Filtered)')
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt.gcf())
                    plt.close()
                # 6. Boxplot by Month
                st.write("Actual vs Predicted by Month (Boxplot)")
                if 'year' in df.columns and 'month' in df.columns and 'y_pred' in df.columns:
                    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
                    plt.figure(figsize=(14,6))
                    # Remove 'label' argument from sns.boxplot calls to avoid TypeError
                    sns.boxplot(x='year_month', y='TokenCount', data=df, color='skyblue', showfliers=False, width=0.5, boxprops=dict(alpha=.5))
                    sns.boxplot(x='year_month', y='y_pred', data=df, color='orange', showfliers=False, width=0.3, boxprops=dict(alpha=.5))
                    plt.title('Actual vs Predicted TokenCount by Month (Filtered)')
                    plt.xlabel('Year-Month')
                    plt.ylabel('TokenCount')
                    plt.xticks(rotation=45)
                    plt.legend(['Actual', 'Predicted'])
                    st.pyplot(plt.gcf())
                    plt.close()
                # 7. Feature Importance (static image)
                st.write("Feature Importances (static)")
                imp_path = os.path.join(REPORTS_DIR, exp_name, 'feature_importance.png')
                if os.path.exists(imp_path):
                    st.image(imp_path, caption="Feature Importances (static)", use_container_width=True, output_format='PNG')
                else:
                    st.info("Feature importance plot not found for this experiment.")
        plot_all(df1, col1, exp1)
        plot_all(df2, col2, exp2)
else:
    # Optionally: allow side-by-side comparison (static images only)
    exp1, exp2 = st.selectbox("Experiment 1", exp_dirs, key="exp1"), st.selectbox("Experiment 2", exp_dirs, key="exp2")
    col1, col2 = st.columns(2)
    plot_names = sorted(set(os.path.basename(f) for f in glob(os.path.join(REPORTS_DIR, exp1, "*.png"))))
    for plot_name in plot_names:
        plot1 = os.path.join(REPORTS_DIR, exp1, plot_name)
        plot2 = os.path.join(REPORTS_DIR, exp2, plot_name)
        with col1:
            st.markdown(f"**{exp1}: {plot_name}**")
            if os.path.exists(plot1):
                st.image(Image.open(plot1), use_container_width=True)
            else:
                st.write("Not found")
        with col2:
            st.markdown(f"**{exp2}: {plot_name}**")
            if os.path.exists(plot2):
                st.image(Image.open(plot2), use_container_width=True)
            else:
                st.write("Not found")
