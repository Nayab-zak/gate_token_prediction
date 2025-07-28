#!/usr/bin/env python3
"""
Streamlit App - Model Comparison Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import json

# Custom YAML loader to handle numpy scalars
class CustomYAMLLoader(yaml.SafeLoader):
    pass

def numpy_scalar_constructor(loader, node):
    """Custom constructor for numpy scalars"""
    return float(loader.construct_scalar(node))

# Register numpy scalar constructors
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
    numpy_scalar_constructor
)
CustomYAMLLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:numpy.float64',
    numpy_scalar_constructor
)

# Page config
st.set_page_config(
    page_title="Model Comparison Dashboard",
    page_icon="🏆",
    layout="wide"
)

@st.cache_data
def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {'data': {'predictions_dir': 'data/predictions'}, 'models': {'champion_file': 'models/champion.txt'}}

@st.cache_data
def load_champion_model():
    """Load champion model name"""
    try:
        with open('models/champion.txt', 'r') as f:
            return f.read().strip()
    except:
        return None

@st.cache_data
def collect_all_model_results():
    """Collect results from all trained models"""
    config = load_config()
    predictions_dir = Path(config['data']['predictions_dir'])
    
    if not predictions_dir.exists():
        return pd.DataFrame()
    
    results = []
    champion = load_champion_model()
    
    for model_dir in predictions_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Get latest metadata file
        metadata_files = list(model_dir.glob("*_metadata_*.yaml"))
        if not metadata_files:
            continue
        
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_metadata, 'r') as f:
                metadata = yaml.load(f, Loader=CustomYAMLLoader)
            
            # Extract metrics safely
            test_metrics = metadata.get('test_metrics', {})
            train_metrics = metadata.get('train_metrics', {})
            
            result = {
                'Model': model_name,
                'Champion': model_name == champion,
                'Data Type': metadata.get('data_type', 'Unknown'),
                'Training Date': str(metadata.get('training_timestamp', ''))[:10],
                'Test MAE': float(test_metrics.get('mae', 0.0)) if test_metrics.get('mae') is not None else np.nan,
                'Test RMSE': float(test_metrics.get('rmse', 0.0)) if test_metrics.get('rmse') is not None else np.nan,
                'Test MAPE': float(test_metrics.get('mape', 0.0)) if test_metrics.get('mape') is not None else np.nan,
                'Train MAE': float(train_metrics.get('mae', 0.0)) if train_metrics.get('mae') is not None else np.nan,
                'Train RMSE': float(train_metrics.get('rmse', 0.0)) if train_metrics.get('rmse') is not None else np.nan,
                'Train MAPE': float(train_metrics.get('mape', 0.0)) if train_metrics.get('mape') is not None else np.nan,
                'Metadata Path': str(latest_metadata)
            }
            
            results.append(result)
            
        except Exception as e:
            st.warning(f"Could not load metadata for {model_name}: {str(e)}")
            # Try to extract metrics from JSON files as fallback
            try:
                json_files = list(model_dir.glob("*_best_params_*.json"))
                if json_files:
                    st.info(f"Attempting fallback loading for {model_name}")
                    result = {
                        'Model': model_name,
                        'Champion': model_name == champion,
                        'Data Type': 'Unknown',
                        'Training Date': 'Unknown',
                        'Test MAE': np.nan,
                        'Test RMSE': np.nan,
                        'Test MAPE': np.nan,
                        'Train MAE': np.nan,
                        'Train RMSE': np.nan,
                        'Train MAPE': np.nan,
                        'Metadata Path': 'JSON fallback'
                    }
                    results.append(result)
            except:
                pass
            continue
    
    return pd.DataFrame(results)

@st.cache_data
def load_model_predictions(model_name):
    """Load predictions for a specific model"""
    config = load_config()
    model_dir = Path(config['data']['predictions_dir']) / model_name
    
    test_pred_files = list(model_dir.glob("*_test_preds_*.csv"))
    if not test_pred_files:
        return None
    
    latest_test = max(test_pred_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_test)

def plot_model_comparison_bar(df, metric='Test MAE'):
    """Create bar chart comparing models"""
    if df.empty or metric not in df.columns:
        return None
    
    # Sort by metric
    df_sorted = df.sort_values(metric)
    
    # Create colors - gold for champion, blue for others
    colors = ['gold' if is_champ else 'lightblue' for is_champ in df_sorted['Champion']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['Model'],
        y=df_sorted[metric],
        text=df_sorted[metric].round(4),
        textposition='outside',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>' + 
                     f'{metric}: %{{y:.4f}}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Model Comparison - {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        template='plotly_white'
    )
    
    return fig

def plot_predictions_scatter(models_data, selected_models):
    """Create scatter plot comparing predictions"""
    if not models_data or not selected_models:
        return None
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(selected_models)]
    
    for i, model in enumerate(selected_models):
        if model in models_data:
            df = models_data[model]
            fig.add_trace(go.Scatter(
                x=df['true_count'],
                y=df['pred_count'],
                mode='markers',
                name=model,
                marker=dict(color=colors[i], opacity=0.6),
                hovertemplate=f'<b>{model}</b><br>Actual: %{{x}}<br>Predicted: %{{y}}<extra></extra>'
            ))
    
    # Perfect prediction line
    if models_data:
        all_true = pd.concat([models_data[model]['true_count'] for model in selected_models if model in models_data])
        all_pred = pd.concat([models_data[model]['pred_count'] for model in selected_models if model in models_data])
        
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
    
    fig.update_layout(
        title="Model Predictions Comparison",
        xaxis_title="Actual Count",
        yaxis_title="Predicted Count",
        template='plotly_white'
    )
    
    return fig

def save_champion_model(champion_model):
    """Save champion model selection"""
    config = load_config()
    champion_file = config['models']['champion_file']
    
    Path(champion_file).parent.mkdir(exist_ok=True, parents=True)
    
    with open(champion_file, 'w') as f:
        f.write(f"{champion_model}\n")

def load_current_champion():
    """Load current champion model"""
    config = load_config()
    champion_file = Path(config['models']['champion_file'])
    
    if champion_file.exists():
        with open(champion_file, 'r') as f:
            return f.read().strip()
    
    return None

# Main app
def main():
    st.title("🏆 Model Comparison Dashboard")
    st.markdown("Compare performance across all trained models")
    
    # Load all model results
    with st.spinner("Loading model results..."):
        results_df = collect_all_model_results()
    
    if results_df.empty:
        st.error("No trained models found. Please run some training agents first.")
        st.stop()
    
    # Overview table
    st.header("📋 Model Overview")
    
    # Format the display dataframe
    display_df = results_df.copy()
    numeric_cols = ['Test MAE', 'Test RMSE', 'Test MAPE', 'Train MAE', 'Train RMSE', 'Train MAPE']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    st.dataframe(display_df[['Model', 'Data Type', 'Training Date', 'Test MAE', 'Test RMSE', 'Test MAPE']], 
                use_container_width=True)
    
    # Champion selection
    st.header("🏆 Champion Model Selection")
    
    current_champion = load_current_champion()
    if current_champion:
        st.info(f"Current Champion: **{current_champion}**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        champion_choice = st.selectbox("Select Champion Model:", results_df['Model'].tolist())
    with col2:
        if st.button("Set as Champion", type="primary"):
            save_champion_model(champion_choice)
            st.success(f"✅ {champion_choice} set as champion!")
            st.rerun()
    
    # Metrics comparison
    st.header("📊 Performance Comparison")
    
    metric_choice = st.selectbox("Select Metric for Comparison:", 
                                ['Test MAE', 'Test RMSE', 'Test MAPE'])
    
    # Bar chart
    bar_fig = plot_model_comparison_bar(results_df, metric_choice)
    if bar_fig:
        st.plotly_chart(bar_fig, use_container_width=True)
    
    # Detailed comparison
    st.header("🔍 Detailed Model Comparison")
    
    # Model selection for detailed comparison
    selected_models = st.multiselect(
        "Select models for detailed comparison:",
        results_df['Model'].tolist(),
        default=results_df['Model'].tolist()[:3] if len(results_df) >= 3 else results_df['Model'].tolist()
    )
    
    if selected_models:
        # Load predictions for selected models
        models_data = {}
        with st.spinner("Loading prediction data..."):
            for model in selected_models:
                pred_df = load_model_predictions(model)
                if pred_df is not None:
                    models_data[model] = pred_df
        
        if models_data:
            # Scatter plot comparison
            scatter_fig = plot_predictions_scatter(models_data, selected_models)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Side-by-side metrics
            st.subheader("📈 Metrics Comparison")
            
            comparison_data = []
            for model in selected_models:
                if model in models_data:
                    df = models_data[model]
                    mae = np.mean(np.abs(df['true_count'] - df['pred_count']))
                    rmse = np.sqrt(np.mean((df['true_count'] - df['pred_count']) ** 2))
                    
                    # MAPE calculation
                    non_zero_mask = df['true_count'] != 0
                    if non_zero_mask.sum() > 0:
                        mape = np.mean(np.abs((df['true_count'][non_zero_mask] - df['pred_count'][non_zero_mask]) / df['true_count'][non_zero_mask])) * 100
                    else:
                        mape = np.inf
                    
                    comparison_data.append({
                        'Model': model,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                # Create metrics comparison chart
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('MAE', 'RMSE', 'MAPE'),
                    specs=[[{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}]]
                )
                
                for i, metric in enumerate(['MAE', 'RMSE', 'MAPE'], 1):
                    fig.add_trace(
                        go.Bar(x=comp_df['Model'], y=comp_df[metric], name=metric, showlegend=False),
                        row=1, col=i
                    )
                
                fig.update_layout(height=400, title_text="Metrics Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics table
                st.dataframe(comp_df.round(4), use_container_width=True)
    
    # Model rankings
    st.header("🥇 Model Rankings")
    
    rankings = {}
    for metric in ['Test MAE', 'Test RMSE', 'Test MAPE']:
        if metric in results_df.columns:
            sorted_models = results_df.sort_values(metric)['Model'].tolist()
            rankings[metric] = sorted_models
    
    if rankings:
        ranking_df = pd.DataFrame(rankings)
        ranking_df.index = range(1, len(ranking_df) + 1)
        ranking_df.index.name = 'Rank'
        st.dataframe(ranking_df, use_container_width=True)
    
    # Export results
    st.header("📤 Export Results")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download Results CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Generate Report"):
            st.markdown("### 📊 Model Comparison Report")
            st.markdown(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Total Models:** {len(results_df)}")
            
            if not results_df.empty:
                best_mae_model = results_df.loc[results_df['Test MAE'].idxmin(), 'Model']
                best_rmse_model = results_df.loc[results_df['Test RMSE'].idxmin(), 'Model']
                best_mape_model = results_df.loc[results_df['Test MAPE'].idxmin(), 'Model']
                
                st.markdown(f"**Best MAE:** {best_mae_model}")
                st.markdown(f"**Best RMSE:** {best_rmse_model}")
                st.markdown(f"**Best MAPE:** {best_mape_model}")

if __name__ == "__main__":
    main()
