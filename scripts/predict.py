import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def predict(model_path, data_path, output_path):
    """
    Predict tokens using the specified model and save the predictions to a file.
    """
    # Load model
    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(data_path)

    # Find date/hour columns
    date_col = None
    for col in ['Date', 'date', 'MoveDate']:
        if col in df.columns:
            date_col = col
            break

    hour_col = None
    for col in ['Hour', 'hour', 'MoveHour']:
        if col in df.columns:
            hour_col = col
            break

    # Select numeric features
    exclude_cols = [date_col, hour_col]
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)

    # Predict
    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.round(y_pred).astype(int)

    # Build output DataFrame
    out_df = pd.DataFrame({
        'Date': df[date_col] if date_col else '',
        'Hour': df[hour_col] if hour_col else '',
        'PredictedTokens': y_pred
    })

    # Save predictions
    out_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    model_path = 'models/Linear_Regression_model.pkl'  # Change as needed
    data_path = 'data/preprocessed/preprocessed_features.csv'
    output_path = 'outputs/predictions.csv'
    predict(model_path, data_path, output_path)
