import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from config import DATA_DIR, FORECAST_HORIZON

# Page setup
st.set_page_config(page_title="Real-Time Predictions", layout="wide")

# Dark theme CSS matching sample
st.markdown(
    """
    <style>
      .reportview-container .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
      .stApp { background-color: #1E1E1E; }
      .stHeader, .stSidebar { background-color: #1E1E1E; }
      h1, h2, h3, h4, h5, h6, p, div, span { color: #E0E0E0; }
      .metric-label { color: #A0A0A0 !important; font-size:1rem; }
      .stMetricValue { font-size:2rem !important; }
      .stMetricDelta { font-size:1rem !important; }
    </style>
    """, unsafe_allow_html=True
)

# Title area with spacing
st.markdown("# ⏱️ Real-Time Hourly Forecast (Next 24h)")
st.markdown("---")

# Load predictions
pred_path = os.path.join(DATA_DIR, 'final_output', 'real_time_predictions.csv')
if not os.path.exists(pred_path):
    st.error("No real-time predictions found. Run the prediction agent first.")
    st.stop()

df = pd.read_csv(pred_path, parse_dates=['datetime'])

# Compute summary stats
min_val = df['prediction'].min()
max_val = df['prediction'].max()
avg_val = df['prediction'].mean()

# Metrics row in 3 columns
col1, col2, col3 = st.columns(3)
col1.metric(label="Min Forecast", value=f"{min_val:.0f}", delta=None)
col2.metric(label="Avg Forecast", value=f"{avg_val:.0f}", delta=None)
col3.metric(label="Max Forecast", value=f"{max_val:.0f}", delta=None)

st.markdown("---")

# Forecast line chart (Plotly) in full width
times = df['datetime']
values = df['prediction']
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=values, mode='lines+markers', line=dict(color='#00CC96'), marker=dict(size=6)))
fig.update_layout(
    plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E',
    font_color='#E0E0E0', margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#444444')
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Table styled with alternating row colors
st.subheader("Detailed Hourly Predictions")

def style_table(df_table):
    return (
        df_table.style
        .set_properties(**{'background-color': '#2E2E2E', 'color': '#E0E0E0', 'border-color': '#444444'})
        .set_table_styles([{'selector': 'th', 'props': [('background-color', '#3E3E3E'), ('color', '#FFFFFF')]}])
        .apply(lambda x: ['background: #252525' if i%2==0 else '' for i in range(len(x))], axis=0)
    )

display_df = df.copy()
display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
st.write(style_table(display_df.set_index('datetime'))
         .to_html(escape=False), unsafe_allow_html=True)

st.markdown("---")
st.write("© Your Company — Live Hourly Forecast")
