import os
import glob
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
from config import DATA_DIR, TOP_IF_SLIDER

# Page config and dark theme CSS
st.set_page_config(page_title="EDA Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #0E1117; color: #FFFFFF;}
    .sidebar .sidebar-content {background-color: #16181D; color: #FFFFFF;}
    h1, h2, h3, h4, h5, h6 {color: #FFFFFF;}
    </style>
    """, unsafe_allow_html=True
)

# Sidebar controls
stage = st.sidebar.radio(
    "Select Stage", ["Raw", "Preprocessed", "Engineered", "Encoded", "Feature Impact"]
)

# Date range filter in sidebar
start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Helper to filter by datetime column
def filter_by_date(df):
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        mask = (df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))
        return df.loc[mask]
    return df

# Data loaders per stage
@st.cache_data
def load_raw():
    files_csv = glob.glob(os.path.join(DATA_DIR, 'raw', '*.csv'))
    files_xlsx = glob.glob(os.path.join(DATA_DIR, 'raw', '*.xlsx'))
    df_list = [pd.read_csv(f) for f in files_csv if os.path.exists(f)]
    df_list += [pd.read_excel(f) for f in files_xlsx if os.path.exists(f)]
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    # Filter by datetime for last 2 months if sidebar filter is not set
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        # If sidebar filter is default, use last 2 months
        if start_date == date(2022, 1, 1) and end_date == date.today():
            end = df['datetime'].max()
            start = end - pd.DateOffset(months=2)
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
        else:
            mask = (df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))
            df = df.loc[mask]
    return df

@st.cache_data
def load_preprocessed():
    path = os.path.join(DATA_DIR, 'preprocessed', 'preprocessed.csv')
    df = pd.read_csv(path, parse_dates=['datetime']) if os.path.exists(path) else pd.DataFrame()
    return filter_by_date(df)

@st.cache_data
def load_engineered():
    path = os.path.join(DATA_DIR, 'features', 'train_features.csv')
    df = pd.read_csv(path, parse_dates=['datetime']) if os.path.exists(path) else pd.DataFrame()
    return filter_by_date(df)

@st.cache_data
def load_encoded():
    path = os.path.join(DATA_DIR, 'encoded_input', 'train_input_classic.csv')
    df = pd.read_csv(path, parse_dates=['datetime']) if os.path.exists(path) else pd.DataFrame()
    return filter_by_date(df)

# Main content panels
if stage == "Raw":
    st.title("Raw Data Overview")
    df = load_raw()
    # Convert all object columns to string for Arrow compatibility
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    # Layout: columns for table and charts
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(), use_container_width=True, height=300)
    with col2:
        if df.empty:
            st.info("No raw data available.")
        else:
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.write("No missing values detected.")
            else:
                st.write("Missing values per column:")
                st.bar_chart(missing, use_container_width=True)
    st.markdown("---")
    if not df.empty and 'TokenCount' in df.columns:
        st.write("TokenCount distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(df['TokenCount'].dropna(), bins=50)
        ax.set_xlabel('TokenCount')
        st.pyplot(fig, use_container_width=True)

elif stage == "Preprocessed":
    st.title("Preprocessed Data")
    df = load_preprocessed()
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head(), use_container_width=True, height=300)
    st.markdown("---")
    if 'TokenCount' in df.columns:
        st.write("Post-clean TokenCount distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(df['TokenCount'], bins=50)
        st.pyplot(fig, use_container_width=True)

elif stage == "Engineered":
    st.title("Engineered Features")
    df = load_engineered()
    if not df.empty:
        col1, col2 = st.columns([1,2])
        with col1:
            st.write("Correlation with TokenCount:")
            corr = df.select_dtypes(include='number').corr()['TokenCount'].sort_values(ascending=False)
            st.bar_chart(corr, use_container_width=True)
        with col2:
            st.write("Sample of engineered features:")
            st.dataframe(df.head(), use_container_width=True, height=300)
    st.markdown("---")

elif stage == "Encoded":
    st.title("Encoded Features")
    df = load_encoded()
    if not df.empty:
        st.write("Latent feature sample:")
        latent_cols = [c for c in df.columns if c.startswith('latent_')]
        st.dataframe(df[latent_cols].head())
        if 'recon_error' in df.columns:
            st.write("Reconstruction error distribution")
            fig, ax = plt.subplots()
            ax.hist(df['recon_error'], bins=50)
            st.pyplot(fig)

elif stage == "Feature Impact":
    st.title("Feature Impact (What-If Analysis)")
    df = load_engineered()
    st.write("Adjust top features to see impact on prediction")
    sliders = {}
    for feat in TOP_IF_SLIDER:
        if feat in df.columns:
            sliders[feat] = st.slider(
                feat,
                float(df[feat].min()),
                float(df[feat].max()),
                float(df[feat].median())
            )
    st.write("Selected feature values:", sliders)
    # Future: integrate model prediction display

# Footer
st.markdown("---")
st.write("EDA Dashboard — End of panel")
