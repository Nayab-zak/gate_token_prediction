import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from config import TEST_SPLIT_MONTHS, FORECAST_HORIZON, TOP_IF_SLIDER

# ─── Page Setup & Theme ───────────────────────────────────────────────────────
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .main {background-color: #FFFFFF; color: #000000;}
      .sidebar .sidebar-content {background-color: #F0F2F6; color: #000000;}
      h1, h2, h3, h4 {color: #000000;}
      .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
      }
      .st-emotion-cache-16txtl3 h4 {
        margin-top: 0;
      }
      hr {margin: 1.5em 0;}
      .highlight-text {
        color: #0068c9;
        font-weight: bold;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Locate dashboard CSVs ─────────────────────────────────────────────────────
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
BASE_DIR = os.path.join(project_root, "data", "dashboard_data")
CHAMPION_DIR = os.path.join(project_root, "models", "champion_model")
os.makedirs(BASE_DIR, exist_ok=True)
all_files = [f for f in os.listdir(BASE_DIR) if f.endswith("_test_results.csv")]

# Load champion model info if available
champion_info = {}
champion_path = os.path.join(CHAMPION_DIR, 'champ.json')
if os.path.exists(champion_path):
    try:
        with open(champion_path, 'r') as f:
            champion_info = json.load(f)
    except:
        pass

# Build a raw_df to inspect full date range
raw_dfs = []
for fname in all_files:
    path = os.path.join(BASE_DIR, fname)
    try:
        tmp = pd.read_csv(path, parse_dates=["datetime"])
        raw_dfs.append(tmp.assign(model=fname.replace("_test_results.csv","")))
    except Exception as e:
        st.sidebar.error(f"Error loading {fname}: {str(e)}")

if raw_dfs:
    raw_df = pd.concat(raw_dfs, ignore_index=True)
    min_dt = raw_df["datetime"].min().date()
    max_dt = raw_df["datetime"].max().date()
    date_range = f"{min_dt} → {max_dt}"
else:
    today = datetime.today().date()
    min_dt = today - timedelta(days=30*TEST_SPLIT_MONTHS)
    max_dt = today
    date_range = "Default date range (no data found)"

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.title("Model Evaluation")
st.sidebar.info(f"Data Range: {date_range}")

# Date pickers defaulted to actual data range
start_date = st.sidebar.date_input("Start Date", value=min_dt, min_value=min_dt, max_value=max_dt)
end_date = st.sidebar.date_input("End Date", value=max_dt, min_value=min_dt, max_value=max_dt)

# Model selector
model_keys = ["All models", "Compare Models", "Champion Only"] + [f.replace("_test_results.csv","") for f in all_files]
selected_model = st.sidebar.selectbox("Select Model View", model_keys, key="model_picker")

# View mode selector
view_mode = st.sidebar.radio("Dashboard Focus", ["Model Performance", "Time Series Analysis", "Error Analysis"])

# Advanced options expander
with st.sidebar.expander("Advanced Options"):
    # Forecast horizon slider
    horizon = st.slider("Forecast Horizon (hrs)", 1, FORECAST_HORIZON, FORECAST_HORIZON)
    
    # Metric selector
    metrics = ["rmse", "mae", "mape", "accuracy"]
    selected_metrics = st.multiselect("Metrics to Show", metrics, default=metrics[:3])
    
    # Sampling for large datasets
    use_sampling = st.checkbox("Sample data for faster plots", value=True)
    sample_size = st.slider("Sample size", 100, 5000, 1000) if use_sampling else None

# What‑If Analysis expander
with st.sidebar.expander("What-If Analysis"):
    st.write("Adjust feature values to simulate predictions:")
    what_if = {feat: st.slider(feat, 0.0, 1.0, 0.5) for feat in TOP_IF_SLIDER}

# ─── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_results(model_key, start_dt, end_dt, sample_size=None):
    df = pd.DataFrame()
    
    if model_key == "All models":
        files = all_files
    elif model_key == "Compare Models":
        files = all_files
    elif model_key == "Champion Only" and 'champion' in champion_info:
        files = [f"{champion_info['champion']}_test_results.csv"]
    else:
        files = [f"{model_key}_test_results.csv"]

    for fname in files:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            try:
                tmp = pd.read_csv(path, parse_dates=["datetime"])
                model_name = fname.replace("_test_results.csv","")
                tmp["model"] = model_name
                tmp["is_champion"] = (model_name == champion_info.get('champion', ''))
                df = pd.concat([df, tmp], ignore_index=True)
            except Exception as e:
                st.error(f"Error loading {fname}: {str(e)}")

    # Date filter
    if not df.empty and "datetime" in df.columns:
        mask = (
            (df["datetime"].dt.date >= start_dt)
            & (df["datetime"].dt.date <= end_dt)
        )
        df = df.loc[mask]
        
        # Add time features for analysis
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        
        # Apply sampling if needed
        if sample_size and len(df) > sample_size:
            return df.sample(sample_size, random_state=42)
            
    # Guarantee model column exists
    if "model" not in df.columns:
        df["model"] = pd.NA

    return df

# Load data based on selection
df = load_results(selected_model, start_date, end_date, sample_size if use_sampling else None)

# ─── Dashboard Header ─────────────────────────────────────────────────────────
if selected_model == "Champion Only" and champion_info:
    champion_name = champion_info.get('champion', 'No champion selected')
    st.title(f"📊 Model Evaluation Dashboard: {champion_name} (Champion Model)")
elif selected_model == "All models" or selected_model == "Compare Models":
    st.title(f"📊 Model Evaluation Dashboard: Model Comparison")
else:
    st.title(f"📊 Model Evaluation Dashboard: {selected_model}")

# ─── Model Summary Card ───────────────────────────────────────────────────────
if not df.empty:
    # Display key metrics in a card layout at the top
    st.markdown("### 📋 Model Performance Summary")
    
    # Create metrics summary table
    metrics_df = df.groupby('model')[['rmse', 'mae', 'mape', 'accuracy']].mean().reset_index()
    
    # Highlight champion model
    if 'champion' in champion_info:
        champion = champion_info['champion']
        metrics_df['is_champion'] = metrics_df['model'] == champion
    
    # Format metrics and show summary table with conditional formatting
    metrics_df = metrics_df.round(2)
    
    # Display in a clean format
    if selected_model not in ["All models", "Compare Models"]:
        # Single model view - show metrics in columns
        metric_cols = st.columns(len(selected_metrics) + 1)
        
        if selected_model == "Champion Only" and champion_info:
            model_data = metrics_df[metrics_df['model'] == champion_info['champion']]
            with metric_cols[0]:
                st.markdown("#### Champion Model")
                st.markdown(f"<span class='highlight-text'>{champion_info['champion']}</span>", unsafe_allow_html=True)
        else:
            model_data = metrics_df[metrics_df['model'] == selected_model]
            with metric_cols[0]:
                st.markdown("#### Model")
                st.markdown(f"<span class='highlight-text'>{selected_model}</span>", unsafe_allow_html=True)
            
        # Display metric values
        for i, metric in enumerate(selected_metrics):
            if metric in model_data.columns:
                with metric_cols[i+1]:
                    st.markdown(f"#### {metric.upper()}")
                    st.markdown(f"<span class='highlight-text'>{model_data[metric].values[0]}</span>", unsafe_allow_html=True)
    else:
        # Multiple models - show comparison table
        st.dataframe(
            metrics_df[['model', 'is_champion'] + selected_metrics]
            .sort_values('mape')
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "is_champion": st.column_config.CheckboxColumn("Champion", help="Is this the champion model?")
            }
        )
else:
    st.info("No data available for the selected date/model range.")

# ────────────────────────────────────────────────────────────────────────────────
# Main dashboard content - show different views based on view_mode selection
# ────────────────────────────────────────────────────────────────────────────────

if view_mode == "Model Performance":
    if df.empty:
        st.warning("No data available to visualize model performance.")
    else:
        st.markdown("## 📈 Model Performance")
        
        # Overall metrics comparison
        st.markdown("### Model Metrics Comparison")
        
        if selected_model in ["All models", "Compare Models"]:
            # Bar chart comparing metrics across models
            metrics_df = df.groupby('model')[selected_metrics].mean().reset_index()
            
            for metric in selected_metrics:
                if metric in metrics_df.columns:
                    fig = px.bar(
                        metrics_df, 
                        x='model', 
                        y=metric, 
                        title=f"{metric.upper()} by Model",
                        color='model',
                        text_auto='.2f'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            # Performance over time
            model_subset = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            
            # Time series of metrics
            fig = go.Figure()
            for metric in selected_metrics:
                if metric in model_subset.columns:
                    # Calculate rolling average for smoother visualization
                    rolling_metric = model_subset.sort_values('datetime').set_index('datetime')[metric].rolling('24H').mean()
                    fig.add_trace(go.Scatter(
                        x=rolling_metric.index,
                        y=rolling_metric,
                        mode='lines',
                        name=f"{metric.upper()} (24hr rolling)"
                    ))
            
            fig.update_layout(
                title='Metrics Over Time (24hr Rolling Average)',
                xaxis_title='Date',
                yaxis_title='Metric Value',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Predictions vs Actuals
        st.markdown("### Predictions vs Actuals")
        
        if selected_model in ["All models", "Compare Models"]:
            # Sample data for each model for clearer visualization
            model_colors = px.colors.qualitative.Plotly
            sampled_data = []
            
            for i, (name, group) in enumerate(df.groupby('model')):
                # Get a sample that's evenly distributed over time
                group_sorted = group.sort_values('datetime')
                step = max(1, len(group_sorted) // 100)  # Limit to ~100 points per model
                sample = group_sorted.iloc[::step]
                sample['color'] = model_colors[i % len(model_colors)]
                sampled_data.append(sample)
                
            sampled_df = pd.concat(sampled_data)
            
            # Create scatter plot with prediction vs actual
            fig = px.scatter(
                sampled_df,
                x='actual',
                y='prediction',
                color='model',
                opacity=0.7,
                title='Predictions vs Actuals by Model',
                labels={'prediction': 'Predicted Value', 'actual': 'Actual Value'},
                hover_data=['datetime', 'error', 'model']
            )
            
            # Add diagonal line (perfect predictions)
            min_val = min(sampled_df['actual'].min(), sampled_df['prediction'].min())
            max_val = max(sampled_df['actual'].max(), sampled_df['prediction'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Perfect Prediction'
                )
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Single model view
            model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            
            # Create 2x2 grid with different plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot of predictions vs actuals
                fig = px.scatter(
                    model_data,
                    x='actual',
                    y='prediction',
                    color='error',
                    color_continuous_scale='RdYlGn_r',
                    opacity=0.7,
                    title='Predictions vs Actuals',
                    labels={'prediction': 'Predicted Value', 'actual': 'Actual Value'},
                    hover_data=['datetime', 'error']
                )
                
                # Add diagonal line (perfect predictions)
                min_val = min(model_data['actual'].min(), model_data['prediction'].min())
                max_val = max(model_data['actual'].max(), model_data['prediction'].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        name='Perfect Prediction'
                    )
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Error distribution
                fig = px.histogram(
                    model_data,
                    x='error',
                    nbins=50,
                    title='Error Distribution',
                    color_discrete_sequence=['#3366CC'],
                    opacity=0.7,
                )
                
                # Add vertical line at zero
                fig.add_vline(
                    x=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Zero Error",
                    annotation_position="top"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Error by magnitude
        st.markdown("### Error by Prediction Magnitude")
        
        if selected_model in ["All models", "Compare Models"]:
            # Create bins for actual values
            # Use manual bins to avoid pandas Interval serialization issues
            min_val = df['actual'].min()
            max_val = df['actual'].max()
            bins = np.linspace(min_val, max_val, 11)  # 11 points to create 10 bins
            labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
            
            df['magnitude_bin'] = pd.cut(df['actual'], bins=bins, labels=labels)
            
            # Calculate mean error by bin and model
            bin_errors = df.groupby(['model', 'magnitude_bin'])['error'].mean().reset_index()
            
            fig = px.bar(
                bin_errors, 
                x='magnitude_bin', 
                y='error', 
                color='model',
                title='Average Error by Value Magnitude',
                barmode='group',
                category_orders={"magnitude_bin": labels}  # Maintain bin order
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            
            # Create bins with string labels to avoid pandas Interval objects
            min_val = model_data['actual'].min()
            max_val = model_data['actual'].max()
            bins = np.linspace(min_val, max_val, 11)  # 11 points to create 10 bins
            labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
            
            model_data['magnitude_bin'] = pd.cut(model_data['actual'], bins=bins, labels=labels)
            
            bin_errors = model_data.groupby('magnitude_bin')['error'].agg(['mean', 'count']).reset_index()
            
            fig = px.bar(
                bin_errors, 
                x='magnitude_bin', 
                y='mean', 
                title='Average Error by Value Magnitude',
                text='count',
                color='mean',
                color_continuous_scale='RdYlGn_r',
                category_orders={"magnitude_bin": labels}  # Maintain bin order
            )
            fig.update_layout(height=400, xaxis={'tickangle': 45})  # Angled labels for better readability
            st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Time Series Analysis":
    if df.empty:
        st.warning("No data available for time series analysis.")
    else:
        st.markdown("## ⏱️ Time Series Analysis")
        
        # Time series plot of predictions vs actuals
        st.markdown("### Predictions vs Actuals Over Time")
        
        if selected_model in ["All models", "Compare Models"]:
            # For multiple models, choose a representative time period
            # Get the most recent week of data
            recent_data = df.sort_values('datetime').groupby('model').tail(24*7)  # Last 7 days (hourly data)
            
            fig = px.line(
                recent_data, 
                x='datetime', 
                y=['actual', 'prediction'],
                color='model',
                title='Recent Predictions vs Actuals (Last 7 Days)',
                labels={'value': 'Value', 'datetime': 'Date & Time'},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            
            # Sort by datetime and limit to a reasonable number of points for visualization
            model_data = model_data.sort_values('datetime')
            if len(model_data) > 500:
                step = len(model_data) // 500
                model_data = model_data.iloc[::step]
                
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=model_data['datetime'],
                y=model_data['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Add predicted values
            fig.add_trace(go.Scatter(
                x=model_data['datetime'],
                y=model_data['prediction'],
                mode='lines',
                name='Prediction',
                line=dict(color='red')
            ))
            
            # Add error bands
            fig.add_trace(go.Scatter(
                x=model_data['datetime'],
                y=model_data['actual'] + model_data['error'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=model_data['datetime'],
                y=model_data['actual'] - model_data['error'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(200, 200, 200, 0.2)',
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip',
                name='Error Margin'
            ))
            
            fig.update_layout(
                title='Predictions vs Actuals Over Time',
                xaxis_title='Date & Time',
                yaxis_title='Value',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Error by time components
        st.markdown("### Error Analysis by Time Components")
        
        # Create a 2x2 grid of time component error analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Error by hour of day
            if selected_model in ["All models", "Compare Models"]:
                hour_errors = df.groupby(['model', 'hour'])['error'].mean().reset_index()
                fig = px.line(
                    hour_errors, 
                    x='hour', 
                    y='error', 
                    color='model',
                    title='Error by Hour of Day',
                    markers=True
                )
            else:
                model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
                hour_errors = model_data.groupby('hour')['error'].mean().reset_index()
                fig = px.bar(
                    hour_errors, 
                    x='hour', 
                    y='error',
                    title='Error by Hour of Day',
                    color='error',
                    color_continuous_scale='RdYlGn_r'
                )
                
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Error by month
            if selected_model in ["All models", "Compare Models"]:
                month_errors = df.groupby(['model', 'month'])['error'].mean().reset_index()
                fig = px.line(
                    month_errors, 
                    x='month', 
                    y='error', 
                    color='model',
                    title='Error by Month',
                    markers=True
                )
            else:
                model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
                month_errors = model_data.groupby('month')['error'].mean().reset_index()
                fig = px.bar(
                    month_errors, 
                    x='month', 
                    y='error',
                    title='Error by Month',
                    color='error',
                    color_continuous_scale='RdYlGn_r'
                )
                
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Error by day of week
            if selected_model in ["All models", "Compare Models"]:
                weekday_errors = df.groupby(['model', 'weekday'])['error'].mean().reset_index()
                fig = px.line(
                    weekday_errors, 
                    x='weekday', 
                    y='error', 
                    color='model',
                    title='Error by Day of Week (0=Monday, 6=Sunday)',
                    markers=True
                )
            else:
                model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
                weekday_errors = model_data.groupby('weekday')['error'].mean().reset_index()
                fig = px.bar(
                    weekday_errors, 
                    x='weekday', 
                    y='error',
                    title='Error by Day of Week (0=Monday, 6=Sunday)',
                    color='error',
                    color_continuous_scale='RdYlGn_r'
                )
                
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Error heatmap (Hour vs Weekday)
            if selected_model in ["All models", "Compare Models"]:
                # For multiple models, just use the first one for the heatmap
                if 'champion' in champion_info:
                    heatmap_data = df[df['model'] == champion_info['champion']]
                else:
                    first_model = df['model'].unique()[0]
                    heatmap_data = df[df['model'] == first_model]
            else:
                heatmap_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            
            # Create pivot table for heatmap
            heat = heatmap_data.pivot_table(
                index='weekday', 
                columns='hour', 
                values='error', 
                aggfunc='mean'
            )
            
            if not heat.empty and not heat.isnull().all().all():
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heat.index = [weekday_names[i] if i < len(weekday_names) else f"Day {i}" for i in heat.index]
                
                fig = px.imshow(
                    heat,
                    title='Error Heatmap (Hour vs Weekday)',
                    labels=dict(x="Hour of Day", y="Day of Week", color="Avg Error"),
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for error heatmap.")

elif view_mode == "Error Analysis":
    if df.empty:
        st.warning("No data available for error analysis.")
    else:
        st.markdown("## 🔍 Error Analysis")
        
        # Error distribution
        st.markdown("### Error Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of errors
            if selected_model in ["All models", "Compare Models"]:
                fig = px.histogram(
                    df, 
                    x='error',
                    color='model',
                    nbins=30,
                    opacity=0.7,
                    title='Error Distribution by Model',
                    barmode='overlay'
                )
            else:
                model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
                fig = px.histogram(
                    model_data, 
                    x='error',
                    nbins=30,
                    title='Error Distribution',
                    color_discrete_sequence=['#3366CC'],
                )
                
                # Add vertical line at zero
                fig.add_vline(
                    x=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Zero Error",
                    annotation_position="top"
                )
                
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Box plot of errors
            if selected_model in ["All models", "Compare Models"]:
                fig = px.box(
                    df, 
                    y='error',
                    x='model',
                    title='Error Distribution Box Plot'
                )
            else:
                model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
                
                # For single model, do a box plot by weekday
                fig = px.box(
                    model_data, 
                    y='error',
                    x='weekday',
                    title='Error Distribution by Day of Week (0=Monday, 6=Sunday)'
                )
                
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Error metrics table
        st.markdown("### Error Metrics Analysis")
        
        if selected_model in ["All models", "Compare Models"]:
            # Create a more detailed metrics table
            detailed_metrics = df.groupby('model').agg({
                'error': ['mean', 'std', 'min', 'max'],
                'mape': 'mean',
                'rmse': 'mean',
                'mae': 'mean',
                'accuracy': 'mean'
            }).reset_index()
            
            # Flatten the column hierarchy
            detailed_metrics.columns = [
                f'{col[0]}_{col[1]}' if col[1] else col[0] 
                for col in detailed_metrics.columns
            ]
            
            # Clean up column names
            column_renames = {
                'error_mean': 'Avg Error',
                'error_std': 'Error StdDev',
                'error_min': 'Min Error',
                'error_max': 'Max Error',
                'mape_mean': 'MAPE',
                'rmse_mean': 'RMSE',
                'mae_mean': 'MAE',
                'accuracy_mean': 'Accuracy'
            }
            detailed_metrics = detailed_metrics.rename(columns=column_renames)
            
            # Round the values
            numeric_cols = [col for col in detailed_metrics.columns if col != 'model']
            detailed_metrics[numeric_cols] = detailed_metrics[numeric_cols].round(2)
            
            # Highlight champion model if available
            if 'champion' in champion_info:
                detailed_metrics['Is Champion'] = detailed_metrics['model'] == champion_info['champion']
            
            st.dataframe(
                detailed_metrics,
                use_container_width=True,
                hide_index=True
            )
        else:
            model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            
            # For single model, show errors by time period
            time_metrics = model_data.groupby('weekday').agg({
                'error': ['mean', 'std', 'count'],
                'mape': 'mean',
                'rmse': 'mean',
            }).reset_index()
            
            # Flatten the column hierarchy
            time_metrics.columns = [
                f'{col[0]}_{col[1]}' if col[1] else col[0] 
                for col in time_metrics.columns
            ]
            
            # Add weekday names
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            time_metrics['Day'] = time_metrics['weekday'].apply(
                lambda x: weekday_names[x] if x < len(weekday_names) else f"Day {x}"
            )
            
            # Clean up and reorder columns
            column_renames = {
                'error_mean': 'Avg Error',
                'error_std': 'Error StdDev',
                'error_count': 'Count',
                'mape_mean': 'MAPE',
                'rmse_mean': 'RMSE',
            }
            time_metrics = time_metrics.rename(columns=column_renames)
            
            # Round the values
            numeric_cols = [col for col in time_metrics.columns if col not in ['weekday', 'Day']]
            time_metrics[numeric_cols] = time_metrics[numeric_cols].round(2)
            
            display_cols = ['Day', 'Avg Error', 'Error StdDev', 'MAPE', 'RMSE', 'Count']
            st.dataframe(
                time_metrics[display_cols],
                use_container_width=True,
                hide_index=True
            )
        
        # Large errors analysis
        st.markdown("### Large Errors Analysis")
        
        # Find the top 10 largest errors
        if selected_model in ["All models", "Compare Models"]:
            largest_errors = df.nlargest(10, 'error')
        else:
            model_data = df if selected_model == "Champion Only" else df[df['model'] == selected_model]
            largest_errors = model_data.nlargest(10, 'error')
            
        if not largest_errors.empty:
            # Display a table of the largest errors
            st.dataframe(
                largest_errors[['datetime', 'model', 'actual', 'prediction', 'error']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No error data available.")

# ─── Data Table (optional) ────────────────────────────────────────────────────
with st.expander("View Raw Prediction Data"):
    if not df.empty:
        st.dataframe(
            df[['datetime', 'model', 'actual', 'prediction', 'error'] + 
               [m for m in selected_metrics if m in df.columns]]
            .sort_values(['model', 'datetime'])
            .reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.warning("No data to display in the predictions table.")

st.markdown("---")
st.caption("© Gate Token Hourly Prediction System - Interactive Evaluation Dashboard")
