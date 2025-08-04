# Shared Streamlit components for EDA dashboard
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DATA_DIR, TEST_SPLIT_MONTHS, VALIDATION_SPLIT_MONTHS
from config import DATA_DIR, TEST_SPLIT_MONTHS, VALIDATION_SPLIT_MONTHS

def show_data_split_visualization():
    """
    Display the time-based train/validation/test split visualization in the EDA dashboard
    """
    st.header("📊 Temporal Data Split Visualization")
    st.markdown("""
    This visualization shows how the data is divided into train, validation, and test sets based on time.
    - **Training data**: Used for model training 
    - **Validation data**: Used for hyperparameter tuning and early stopping
    - **Test data**: Used only for final model evaluation
    """)
    
    # Check if split files exist
    split_files = ['train.csv', 'validation.csv', 'test.csv']
    files_exist = all(os.path.exists(os.path.join(DATA_DIR, 'preprocessed', file)) for file in split_files)
    
    if not files_exist:
        st.warning("⚠️ Data split files not found. Run the splitting agent first.")
        if st.button("Generate Sample Visualization"):
            st.info("This would show a sample visualization with random data.")
            # For future implementation if needed
        return
    
    # Load data
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed', 'train.csv'), parse_dates=['datetime'])
        validation = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed', 'validation.csv'), parse_dates=['datetime'])
        test = pd.read_csv(os.path.join(DATA_DIR, 'preprocessed', 'test.csv'), parse_dates=['datetime'])
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Train Set", f"{len(train):,} samples", f"{len(train)/len(train.datetime.dt.date.unique()):.1f} samples/day")
            st.caption(f"From {train.datetime.min().date()} to {train.datetime.max().date()}")
        
        with col2:
            st.metric("Validation Set", f"{len(validation):,} samples", f"{len(validation)/len(validation.datetime.dt.date.unique()):.1f} samples/day")
            st.caption(f"From {validation.datetime.min().date()} to {validation.datetime.max().date()}")
        
        with col3:
            st.metric("Test Set", f"{len(test):,} samples", f"{len(test)/len(test.datetime.dt.date.unique()):.1f} samples/day")
            st.caption(f"From {test.datetime.min().date()} to {test.datetime.max().date()}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Determine the correct token count column name (could be 'token_count' or 'TokenCount')
        token_col = 'TokenCount' if 'TokenCount' in train.columns else 'token_count'
        if token_col not in train.columns:
            st.error(f"Error: Column '{token_col}' not found in the data. Available columns: {', '.join(train.columns)}")
            return
            
        st.info(f"Using column '{token_col}' for token count data")
        
        # Create resampled time series for better visualization
        train_daily = train.set_index('datetime')[token_col].resample('D').mean()
        validation_daily = validation.set_index('datetime')[token_col].resample('D').mean()
        test_daily = test.set_index('datetime')[token_col].resample('D').mean()
        
        # Plot data
        ax.plot(train_daily.index, train_daily, 'b-', alpha=0.7, label=f'Train ({len(train)} samples)')
        ax.plot(validation_daily.index, validation_daily, 'g-', alpha=0.7, label=f'Validation ({len(validation)} samples)')
        ax.plot(test_daily.index, test_daily, 'r-', alpha=0.7, label=f'Test ({len(test)} samples)')
        
        # Add vertical lines for split points
        val_cutoff = validation.datetime.min()
        test_cutoff = test.datetime.min()
        
        ax.axvline(x=val_cutoff, color='g', linestyle='--', alpha=0.7, 
                  label=f'Validation Cutoff: {val_cutoff.strftime("%Y-%m-%d")}')
        ax.axvline(x=test_cutoff, color='r', linestyle='--', alpha=0.7, 
                  label=f'Test Cutoff: {test_cutoff.strftime("%Y-%m-%d")}')
        
        # Add labels and legend
        ax.set_title(f'Temporal Data Split: Train, Validation ({VALIDATION_SPLIT_MONTHS} months), Test ({TEST_SPLIT_MONTHS} months)', 
                    fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Token Count (Daily Average)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        
        # Format x-axis to show dates clearly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Show additional statistics
        with st.expander("📈 Distribution Statistics"):
            stats_cols = st.columns(3)
            
            with stats_cols[0]:
                st.subheader("Train Set")
                st.write(f"Mean: {train.token_count.mean():.2f}")
                st.write(f"Std: {train.token_count.std():.2f}")
                st.write(f"Min: {train.token_count.min():.2f}")
                st.write(f"Max: {train.token_count.max():.2f}")
            
            with stats_cols[1]:
                st.subheader("Validation Set")
                st.write(f"Mean: {validation.token_count.mean():.2f}")
                st.write(f"Std: {validation.token_count.std():.2f}")
                st.write(f"Min: {validation.token_count.min():.2f}")
                st.write(f"Max: {validation.token_count.max():.2f}")
            
            with stats_cols[2]:
                st.subheader("Test Set")
                st.write(f"Mean: {test.token_count.mean():.2f}")
                st.write(f"Std: {test.token_count.std():.2f}")
                st.write(f"Min: {test.token_count.min():.2f}")
                st.write(f"Max: {test.token_count.max():.2f}")
            
            # Show distribution overlays
            st.subheader("Token Count Distribution by Split")
            dist_fig, dist_ax = plt.subplots(figsize=(10, 6))
            dist_ax.hist(train.token_count, alpha=0.5, bins=30, label='Train')
            dist_ax.hist(validation.token_count, alpha=0.5, bins=30, label='Validation')
            dist_ax.hist(test.token_count, alpha=0.5, bins=30, label='Test')
            dist_ax.set_xlabel('Token Count')
            dist_ax.set_ylabel('Frequency')
            dist_ax.legend()
            st.pyplot(dist_fig)
    
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.code(str(e))
