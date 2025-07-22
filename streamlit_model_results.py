#!/usr/bin/env python3
"""
Streamlit App - Per-Model Results Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="Model Results Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {'data': {'predictions_dir': 'data/predictions'}}

@st.cache_data
def get_available_models():
    """Get list of available models"""
    config = load_config()
    predictions_dir = Path(config['data']['predictions_dir'])
    
    if not predictions_dir.exists():
        return []
    
    models = []
    for model_dir in predictions_dir.iterdir():
        if model_dir.is_dir():
            # Check if there are prediction files
            pred_files = list(model_dir.glob("*_test_preds_*.csv"))
            if pred_files:
                models.append(model_dir.name)
    
    return sorted(models)

@st.cache_data
def load_model_data(model_name):
    """Load model predictions and metadata"""
    config = load_config()
    model_dir = Path(config['data']['predictions_dir']) / model_name
    
    # Get latest files
    test_pred_files = list(model_dir.glob("*_test_preds_*.csv"))
    train_pred_files = list(model_dir.glob("*_train_preds_*.csv"))
    metadata_files = list(model_dir.glob("*_metadata_*.yaml"))
    
    if not test_pred_files:
        return None, None, None
    
    # Use latest files
    latest_test = max(test_pred_files, key=lambda x: x.stat().st_mtime)
    latest_train = max(train_pred_files, key=lambda x: x.stat().st_mtime) if train_pred_files else None
    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime) if metadata_files else None
    
    # Load data
    test_df = pd.read_csv(latest_test)
    train_df = pd.read_csv(latest_train) if latest_train else None
    
    metadata = None
    if latest_metadata:
        try:
            with open(latest_metadata, 'r') as f:
                content = f.read()
                # Remove problematic NumPy object serializations
                lines = content.split('\n')
                filtered_lines = []
                skip_lines = False
                
                for line in lines:
                    if 'test_metrics:' in line:
                        skip_lines = True
                        continue
                    if skip_lines and (line.startswith('  ') or line.strip() == ''):
                        continue
                    if skip_lines and not line.startswith(' '):
                        skip_lines = False
                    
                    if not skip_lines:
                        filtered_lines.append(line)
                
                filtered_content = '\n'.join(filtered_lines)
                metadata = yaml.safe_load(filtered_content)
        except Exception as e:
            st.warning(f"Could not load metadata: {e}")
            metadata = {'model': model_name, 'data_type': 'unknown'}
    
    return test_df, train_df, metadata

def calculate_metrics(df):
    """Calculate metrics for predictions"""
    if df is None or 'true_count' not in df.columns or 'pred_count' not in df.columns:
        return {}
    
    y_true = df['true_count']
    y_pred = df['pred_count']
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Avoid division by zero for MAPE
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.inf
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def plot_time_series(df, title="Predictions vs Actual"):
    """Plot time series predictions"""
    if df is None or len(df) == 0:
        return None
    
    fig = go.Figure()
    
    # Convert timestamp if it's a string
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        x_axis = df['timestamp']
    else:
        x_axis = df.index
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df['true_count'],
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df['pred_count'],
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_residuals(df):
    """Plot residual distribution"""
    if df is None or len(df) == 0:
        return None
    
    residuals = df['true_count'] - df['pred_count']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        name='Residuals',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Residual Distribution",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Frequency",
        template='plotly_white'
    )
    
    return fig

def plot_scatter_predictions(df):
    """Plot predicted vs actual scatter"""
    if df is None or len(df) == 0:
        return None
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df['true_count'],
        y=df['pred_count'],
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(df['true_count'].min(), df['pred_count'].min())
    max_val = max(df['true_count'].max(), df['pred_count'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Predicted vs Actual",
        xaxis_title="Actual Count",
        yaxis_title="Predicted Count",
        template='plotly_white'
    )
    
    return fig

# Main app
def main():
    st.title("📊 Model Results Dashboard")
    st.markdown("Explore individual model performance and predictions")
    
    # Sidebar
    st.sidebar.header("Model Selection")
    
    models = get_available_models()
    if not models:
        st.error("No trained models found. Please run some training agents first.")
        st.stop()
    
    selected_model = st.sidebar.selectbox("Select Model", models)
    
    # Load model data
    with st.spinner(f"Loading {selected_model} data..."):
        test_df, train_df, metadata = load_model_data(selected_model)
    
    if test_df is None:
        st.error(f"Could not load data for model: {selected_model}")
        st.stop()
    
    # Model information
    st.header(f"Model: {selected_model.upper()}")
    
    if metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Type", metadata.get('data_type', 'Unknown'))
        with col2:
            st.metric("Training Date", metadata.get('training_timestamp', 'Unknown')[:10])
        with col3:
            st.metric("Best Model Path", "Available" if 'model_path' in metadata else "Missing")
    
    # Metrics
    st.subheader("📈 Performance Metrics")
    
    test_metrics = calculate_metrics(test_df)
    train_metrics = calculate_metrics(train_df) if train_df is not None else {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Test Set Performance**")
        if test_metrics:
            for metric, value in test_metrics.items():
                st.metric(metric, f"{value:.4f}")
    
    with col2:
        if train_metrics:
            st.markdown("**Train Set Performance**")
            for metric, value in train_metrics.items():
                st.metric(metric, f"{value:.4f}")
    
    # Date range filter
    st.subheader("🔍 Analysis Controls")
    
    if 'timestamp' in test_df.columns:
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        min_date = test_df['timestamp'].min().date()
        max_date = test_df['timestamp'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data
        mask = (test_df['timestamp'].dt.date >= start_date) & (test_df['timestamp'].dt.date <= end_date)
        filtered_df = test_df[mask]
    else:
        filtered_df = test_df
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    # Time series plot
    ts_fig = plot_time_series(filtered_df, f"{selected_model} - Time Series Predictions")
    if ts_fig:
        st.plotly_chart(ts_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals
        res_fig = plot_residuals(filtered_df)
        if res_fig:
            st.plotly_chart(res_fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        scatter_fig = plot_scatter_predictions(filtered_df)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Prediction Data")
    
    display_df = filtered_df.copy()
    if 'timestamp' in display_df.columns:
        display_df['residual'] = display_df['true_count'] - display_df['pred_count']
        display_df['abs_error'] = np.abs(display_df['residual'])
    
    st.dataframe(display_df, use_container_width=True)
    
    # User notes
    st.subheader("📝 Notes")
    notes_file = Path(f"models/{selected_model}_notes.txt")
    
    if notes_file.exists():
        with open(notes_file, 'r') as f:
            existing_notes = f.read()
    else:
        existing_notes = ""
    
    notes = st.text_area("Add your observations about this model:", existing_notes, height=100)
    
    if st.button("Save Notes"):
        with open(notes_file, 'w') as f:
            f.write(notes)
        st.success("Notes saved!")

if __name__ == "__main__":
    main()
