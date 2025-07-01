import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import plotly.express as px
import datetime

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_FILE_PATH

mpl.rcParams['agg.path.chunksize'] = 10000

# --- PAGE CONFIG ---
st.set_page_config(page_title="Token Count EDA Dashboard", layout="wide")
st.title("Token Count EDA Dashboard")

# --- DATA LOADING ---
EDA_PATH = os.path.join('data', 'preprocessed', 'preprocessed_features_for_eda.xlsx')
if not os.path.exists(EDA_PATH):
    st.error(f"EDA file not found: {EDA_PATH}. Please run feature engineering first.")
    st.stop()
try:
    df = pd.read_excel(EDA_PATH, parse_dates=['MoveDate'])
except:
    df = pd.read_csv(EDA_PATH, parse_dates=['MoveDate'])

# # Debug output for troubleshooting empty dashboard
# st.write("## Debug Info: DataFrame after load")
# st.write("Shape:", df.shape)
# st.write("Columns:", df.columns.tolist())
# st.write(df.head())
# if 'MoveHour' in df.columns:
#     st.write("MoveHour unique:", df['MoveHour'].unique())
#     st.write("MoveHour dtype:", df['MoveHour'].dtype)
#     # Fix MoveHour to always be integer (not float)
#     df['MoveHour'] = pd.to_numeric(df['MoveHour'], errors='coerce').fillna(-1).astype(int)
#     st.write("MoveHour unique after int conversion:", df['MoveHour'].unique())
#     st.write(df[['MoveDate', 'MoveHour']].head(20))

# Drop rows with missing MoveDate or year/month, and convert year/month to int
if 'MoveDate' in df.columns:
    df = df[df['MoveDate'].notna()]
if 'year' in df.columns:
    df = df[df['year'].notna()]
    df['year'] = df['year'].astype(int)
if 'month' in df.columns:
    df = df[df['month'].notna()]
    df['month'] = df['month'].astype(int)

# Tabs
tabs = st.tabs(["Time Series","Hourly","Monthly","Categorical","Cyclic"])

# Helper for analysis text
analysis = {
    "time_series": "This plot shows the TokenCount trend at an hourly level. Peaks and lows are highlighted, helping management quickly spot operational surges or lulls. Use the filters to focus on specific periods.",
    "hourly": "This plot displays the average TokenCount for each hour of the day, aggregated over the selected period. It helps identify which hours are busiest or slowest, supporting resource planning.",
    "monthly": "This plot shows the average TokenCount for each month, allowing management to spot seasonal trends and plan for high/low demand periods.",
    "categorical": "This boxplot compares TokenCount distributions by MoveType. It helps management understand which operation types drive the most or least activity.",
    "cyclic": "This scatter plot visualizes TokenCount by hour of day, with bubble size and color indicating volume. It makes it easy to spot time-of-day patterns and outliers in activity."
}

# --- Improve plot layout for all plots ---
def compact_fig():
    fig, ax = plt.subplots(figsize=(8, 3))
    plt.tight_layout(pad=2)
    return fig, ax

# 1. Time Series Tab
with tabs[0]:
    st.subheader("TokenCount Over Time (Hourly Granularity)")
    st.markdown(f"<div style='background-color:#f0f4f8;padding:8px 12px;border-radius:6px;margin-bottom:10px;'><b>Analysis:</b> {analysis['time_series']}</div>", unsafe_allow_html=True)
    # Filters
    sel_yrs = []
    if 'year' in df.columns:
        yrs = sorted(df.year.unique())
        default_year = [yrs[-1]] if yrs else []
        sel_yrs = st.multiselect("Select Year", yrs, default=default_year, key="ts_year")
    sel_mths = []
    if 'month' in df.columns:
        mths = sorted(df.month.unique())
        # Default to latest month in latest year
        if sel_yrs:
            latest_year = sel_yrs[-1]
            mths_in_latest_year = sorted(df[df['year'] == latest_year]['month'].unique())
            default_month = [mths_in_latest_year[-1]] if mths_in_latest_year else []
        else:
            default_month = [mths[-1]] if mths else []
        sel_mths = st.multiselect("Select Month", mths, default=default_month, key="ts_month")
    df_ts = df.copy()
    if sel_yrs:
        df_ts = df_ts[df_ts.year.isin(sel_yrs)]
    if sel_mths:
        df_ts = df_ts[df_ts.month.isin(sel_mths)]
    # --- FIX: Drop rows with missing MoveDate or MoveHour before plotting ---
    df_ts = df_ts[df_ts['MoveDate'].notna() & df_ts['MoveHour'].notna()]
    if not df_ts.empty:
        df_ts['date_hour'] = df_ts['MoveDate'].dt.strftime('%Y-%m-%d') + ' ' + df_ts['MoveHour'].astype(int).astype(str).str.zfill(2) + ':00'
        df_ts['date_hour'] = pd.to_datetime(df_ts['date_hour'], format='%Y-%m-%d %H:%M', errors='coerce')
        hourly_trend = df_ts.groupby('date_hour')['TokenCount'].sum().reset_index()
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_trend['date_hour'], y=hourly_trend['TokenCount'],
                                 mode='lines+markers',
                                 line=dict(color='#1f77b4', width=3),
                                 marker=dict(size=7, color='#1f77b4', line=dict(width=1, color='white')),
                                 name='TokenCount'))
        # Highlight max and min
        if not hourly_trend.empty:
            max_idx = hourly_trend['TokenCount'].idxmax()
            min_idx = hourly_trend['TokenCount'].idxmin()
            max_x = hourly_trend.loc[max_idx, 'date_hour']
            max_y = hourly_trend.loc[max_idx, 'TokenCount']
            min_x = hourly_trend.loc[min_idx, 'date_hour']
            min_y = hourly_trend.loc[min_idx, 'TokenCount']
            import numpy as np
            if isinstance(max_x, pd.Timestamp):
                max_x = max_x.to_pydatetime()
            elif isinstance(max_x, np.datetime64):
                max_x = pd.to_datetime(max_x).to_pydatetime()
            if isinstance(min_x, pd.Timestamp):
                min_x = min_x.to_pydatetime()
            elif isinstance(min_x, np.datetime64):
                min_x = pd.to_datetime(min_x).to_pydatetime()
            max_x_str = max_x.strftime('%Y-%m-%d %H:%M') if hasattr(max_x, 'strftime') else str(max_x)
            min_x_str = min_x.strftime('%Y-%m-%d %H:%M') if hasattr(min_x, 'strftime') else str(min_x)
            fig.add_trace(go.Scatter(x=[max_x], y=[max_y], mode='markers+text',
                                     marker=dict(size=16, color='red'),
                                     text=['Peak'], textposition='top center', name='Peak'))
            fig.add_trace(go.Scatter(x=[min_x], y=[min_y], mode='markers+text',
                                     marker=dict(size=16, color='green'),
                                     text=['Lowest'], textposition='bottom center', name='Lowest'))
        fig.update_layout(
            title='TokenCount Trend at Hourly Level',
            xaxis_title='Date & Hour',
            yaxis_title='TokenCount',
            template='plotly_white',
            hovermode='x unified',
            font=dict(size=14),
            margin=dict(l=30, r=30, t=60, b=30),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='background-color:#eaf6fb;padding:10px;border-radius:8px;'>
        <b>What does this show?</b><br>
        <ul>
        <li><b>Hourly granularity:</b> Each point is the total TokenCount for a specific hour of a specific day.</li>
        <li>Hover over any point to see the exact value and time.</li>
        <li>Peak and low hours are highlighted with vertical lines.</li>
        <li>Use the filters above to focus on specific years or months.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No data to display for selected period.")

# 2. Hourly Tab
with tabs[1]:
    st.subheader("Average TokenCount by Hour")
    st.markdown(f"<div style='background-color:#f0f4f8;padding:8px 12px;border-radius:6px;margin-bottom:10px;'><b>Analysis:</b> {analysis['hourly']}</div>", unsafe_allow_html=True)
    sel_yrs_hr = []
    if 'year' in df.columns:
        yrs_hr = sorted(df.year.unique())
        default_year_hr = [yrs_hr[-1]] if yrs_hr else []
        sel_yrs_hr = st.multiselect("Select Year", yrs_hr, default=default_year_hr, key="hr_year")
    sel_mths_hr = []
    if 'month' in df.columns:
        mths_hr = sorted(df.month.unique())
        if sel_yrs_hr:
            latest_year_hr = sel_yrs_hr[-1]
            mths_in_latest_year_hr = sorted(df[df['year'] == latest_year_hr]['month'].unique())
            default_month_hr = [mths_in_latest_year_hr[-1]] if mths_in_latest_year_hr else []
        else:
            default_month_hr = [mths_hr[-1]] if mths_hr else []
        sel_mths_hr = st.multiselect("Select Month", mths_hr, default=default_month_hr, key="hr_month")
    sel_hrs = []
    if 'MoveHour' in df.columns:
        hrs = sorted(df.MoveHour.unique())
        sel_hrs = st.multiselect("Select Hour(s)", hrs, default=hrs, key="hr_hours")
    df_hr = df.copy()
    if sel_yrs_hr:
        df_hr = df_hr[df_hr.year.isin(sel_yrs_hr)]
    if sel_mths_hr:
        df_hr = df_hr[df_hr.month.isin(sel_mths_hr)]
    if sel_hrs:
        df_hr = df_hr[df_hr.MoveHour.isin(sel_hrs)]
    if not df_hr.empty and 'MoveHour' in df_hr.columns:
        import plotly.express as px
        hourly = df_hr.groupby('MoveHour').TokenCount.mean().reset_index()
        fig = px.bar(hourly, x='MoveHour', y='TokenCount',
            labels={'MoveHour': 'Hour', 'TokenCount': 'Avg TokenCount'},
            template='plotly_white',
            hover_data={'MoveHour': True, 'TokenCount': ':.2f'})
        fig.update_traces(marker_color='#1f77b4')
        fig.update_layout(xaxis=dict(dtick=1), hoverlabel=dict(bgcolor="white", font_size=14), height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hourly data for selected filters.")

# 3. Monthly Tab
with tabs[2]:
    st.subheader("Monthly Average TokenCount")
    st.markdown(f"<div style='background-color:#f0f4f8;padding:8px 12px;border-radius:6px;margin-bottom:10px;'><b>Analysis:</b> {analysis['monthly']}</div>", unsafe_allow_html=True)
    sel_yrs_mon = []
    if 'year' in df.columns:
        yrs_mon = sorted(df.year.unique())
        default_year_mon = [yrs_mon[-1]] if yrs_mon else []
        sel_yrs_mon = st.multiselect("Select Year", yrs_mon, default=default_year_mon, key="mon_year")
    sel_mths_mon = []
    if 'month' in df.columns:
        mths_mon = sorted(df.month.unique())
        if sel_yrs_mon:
            latest_year_mon = sel_yrs_mon[-1]
            mths_in_latest_year_mon = sorted(df[df['year'] == latest_year_mon]['month'].unique())
            default_month_mon = [mths_in_latest_year_mon[-1]] if mths_in_latest_year_mon else []
        else:
            default_month_mon = [mths_mon[-1]] if mths_mon else []
        sel_mths_mon = st.multiselect("Select Month", mths_mon, default=default_month_mon, key="mon_month")
    df_mon = df.copy()
    if sel_yrs_mon:
        df_mon = df_mon[df_mon.year.isin(sel_yrs_mon)]
    if sel_mths_mon:
        df_mon = df_mon[df_mon.month.isin(sel_mths_mon)]
    if not df_mon.empty:
        import plotly.express as px
        # Ensure 'year' and 'month' columns exist
        if 'year' not in df_mon.columns:
            df_mon['year'] = df_mon['MoveDate'].dt.year
        if 'month' not in df_mon.columns:
            df_mon['month'] = df_mon['MoveDate'].dt.month
        monthly = df_mon.groupby(['year','month']).TokenCount.mean().reset_index()
        monthly['ym'] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
        fig = px.line(monthly, x='ym', y='TokenCount', markers=True,
                      labels={'ym': 'Year-Month', 'TokenCount': 'Avg TokenCount'},
                      title='Monthly Average TokenCount')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data for selected filters.")

# 4. Categorical Tab
with tabs[3]:
    st.subheader("TokenCount by MoveType")
    st.markdown(f"<div style='background-color:#f0f4f8;padding:8px 12px;border-radius:6px;margin-bottom:10px;'><b>Analysis:</b> {analysis['categorical']}</div>", unsafe_allow_html=True)
    sel_yrs_cat = []
    if 'year' in df.columns:
        yrs_cat = sorted(df.year.unique())
        default_year_cat = [yrs_cat[-1]] if yrs_cat else []
        sel_yrs_cat = st.multiselect("Select Year", yrs_cat, default=default_year_cat, key="cat_year")
    sel_mths_cat = []
    if 'month' in df.columns:
        mths_cat = sorted(df.month.unique())
        if sel_yrs_cat:
            latest_year_cat = sel_yrs_cat[-1]
            mths_in_latest_year_cat = sorted(df[df['year'] == latest_year_cat]['month'].unique())
            default_month_cat = [mths_in_latest_year_cat[-1]] if mths_in_latest_year_cat else []
        else:
            default_month_cat = [mths_cat[-1]] if mths_cat else []
        sel_mths_cat = st.multiselect("Select Month", mths_cat, default=default_month_cat, key="cat_month")
    sel_types = []
    if 'MoveType' in df.columns:
        types = sorted(df.MoveType.unique())
        sel_types = st.multiselect("Select MoveType", types, default=types, key="cat_type")
    df_cat = df.copy()
    if sel_yrs_cat:
        df_cat = df_cat[df_cat.year.isin(sel_yrs_cat)]
    if sel_mths_cat:
        df_cat = df_cat[df_cat.month.isin(sel_mths_cat)]
    if sel_types:
        df_cat = df_cat[df_cat.MoveType.isin(sel_types)]
    if not df_cat.empty and 'MoveType' in df_cat.columns:
        import plotly.express as px
        fig = px.box(df_cat, x='MoveType', y='TokenCount', points='all',
            color='MoveType',
            labels={'MoveType': 'MoveType', 'TokenCount': 'TokenCount'},
            template='plotly_white',
            hover_data=['MoveType', 'TokenCount'])
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(showlegend=False, hoverlabel=dict(bgcolor="white", font_size=14), height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical data for selected types.")
    if all(c in df_cat.columns for c in ['TerminalID','Desig']):
        st.subheader("Pivot: Terminal vs Designation")
        pivot = df_cat.pivot_table(index='TerminalID', columns='Desig', values='TokenCount', aggfunc='mean')
        st.dataframe(pivot)

# 5. Cyclic Tab
with tabs[4]:
    st.subheader("Cyclic Hour Encoding (bubble size ~ TokenCount)")
    st.markdown(f"<div style='background-color:#f0f4f8;padding:8px 12px;border-radius:6px;margin-bottom:10px;'><b>Analysis:</b> {analysis['cyclic']}</div>", unsafe_allow_html=True)
    sel_yrs_cyc = []
    if 'year' in df.columns:
        yrs_cyc = sorted(df.year.unique())
        default_year_cyc = [yrs_cyc[-1]] if yrs_cyc else []
        sel_yrs_cyc = st.multiselect("Select Year", yrs_cyc, default=default_year_cyc, key="cyc_year")
    sel_mths_cyc = []
    if 'month' in df.columns:
        mths_cyc = sorted(df.month.unique())
        if sel_yrs_cyc:
            latest_year_cyc = sel_yrs_cyc[-1]
            mths_in_latest_year_cyc = sorted(df[df['year'] == latest_year_cyc]['month'].unique())
            default_month_cyc = [mths_in_latest_year_cyc[-1]] if mths_in_latest_year_cyc else []
        else:
            default_month_cyc = [mths_cyc[-1]] if mths_cyc else []
        sel_mths_cyc = st.multiselect("Select Month", mths_cyc, default=default_month_cyc, key="cyc_month")
    sel_hrs_c = []
    if 'MoveHour' in df.columns:
        hrs_c = sorted(df.MoveHour.unique())
        sel_hrs_c = st.multiselect("Select Hour(s)", hrs_c, default=hrs_c, key="cyc_hours")
    df_cyc = df.copy()
    if sel_yrs_cyc:
        df_cyc = df_cyc[df_cyc.year.isin(sel_yrs_cyc)]
    if sel_mths_cyc:
        df_cyc = df_cyc[df_cyc.month.isin(sel_mths_cyc)]
    if sel_hrs_c:
        df_cyc = df_cyc[df_cyc.MoveHour.isin(sel_hrs_c)]
    if not df_cyc.empty and 'hour_sin' in df_cyc.columns:
        import plotly.express as px
        fig = px.scatter(df_cyc, x='MoveHour', y='TokenCount', size='TokenCount', color='MoveHour',
            color_continuous_scale='viridis',
            labels={'MoveHour': 'Hour of Day (0-23)', 'TokenCount': 'TokenCount'},
            template='plotly_white',
            hover_data=['MoveHour', 'TokenCount'])
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(coloraxis_colorbar=dict(title='Hour'), hoverlabel=dict(bgcolor="white", font_size=14), height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each point represents a time of day (hour), with bubble size proportional to TokenCount. Hover for details.")
    else:
        st.info("No cyclic data for selected filters.")
