import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from agents.config import MODEL_PATH, PREPROCESSED_FEATURES_PATH, MODEL_FEATURES_PATH, REPORTS_DIR
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
try:
    import lightgbm as lgb
    has_lgb = True
except ImportError:
    has_lgb = False

mpl.rcParams['agg.path.chunksize'] = 10000  # Fix OverflowError for large plots

# Paths
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load model and data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(PREPROCESSED_FEATURES_PATH, parse_dates=['MoveDate'])

# Load feature names
if os.path.exists(MODEL_FEATURES_PATH):
    import json
    with open(MODEL_FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
else:
    feature_names = [col for col in df.columns if col not in ['TokenCount', 'MoveDate']]

# Prepare X, y
X = df[feature_names].copy()
y = df['TokenCount']

# Predict
y_pred = model.predict(X)
df['y_pred'] = y_pred

# 1. Actual vs Predicted (Time Series)
plt.figure(figsize=(16,5))
plt.plot(df['MoveDate'], y, label='Actual', alpha=0.7)
plt.plot(df['MoveDate'], y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted TokenCount (Time Series)')
plt.xlabel('Date')
plt.ylabel('TokenCount')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'actual_vs_predicted_timeseries.png'))
plt.close()

# 2. Scatter: Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.3, label='Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
plt.xlabel('Actual TokenCount')
plt.ylabel('Predicted TokenCount')
plt.title('Actual vs Predicted TokenCount (Scatter)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'actual_vs_predicted_scatter.png'))
plt.close()

# 3. Residuals plot
residuals = y - y_pred
plt.figure(figsize=(12,4))
plt.plot(df['MoveDate'], residuals, alpha=0.7, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Error')
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residual (Actual - Predicted)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'residuals_timeseries.png'))
plt.close()

# 4. Histogram of residuals
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50, alpha=0.7, label='Residuals')
plt.axvline(0, color='red', linestyle='--', label='Zero Error')
plt.title('Distribution of Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'residuals_histogram.png'))
plt.close()

# 5. MAE/RMSE by Month
if 'month' in df.columns and 'year' in df.columns:
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    grouped = df.groupby('year_month').apply(lambda g: pd.Series({
        'MAE': np.mean(np.abs(g['TokenCount'] - g['y_pred'])),
        'RMSE': np.sqrt(np.mean((g['TokenCount'] - g['y_pred'])**2))
    }))
    grouped = grouped.reset_index()
    plt.figure(figsize=(12,5))
    plt.plot(grouped['year_month'], grouped['MAE'], label='MAE')
    plt.plot(grouped['year_month'], grouped['RMSE'], label='RMSE')
    plt.title('MAE and RMSE by Month')
    plt.xlabel('Year-Month')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'mae_rmse_by_month.png'))
    plt.close()

# 6. Boxplot: Actual vs Predicted by Month
if 'month' in df.columns and 'year' in df.columns:
    import seaborn as sns
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    plt.figure(figsize=(16,6))
    ax = plt.gca()
    sns.boxplot(x='year_month', y='TokenCount', data=df, color='skyblue', showfliers=False, width=0.5, boxprops=dict(alpha=.5), ax=ax)
    sns.boxplot(x='year_month', y='y_pred', data=df, color='orange', showfliers=False, width=0.3, boxprops=dict(alpha=.5), ax=ax)
    plt.title('Actual vs Predicted TokenCount by Month (Boxplot)')
    plt.xlabel('Year-Month')
    plt.ylabel('TokenCount')
    plt.xticks(rotation=45)
    # Add custom legend manually
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor='skyblue', edgecolor='black', label='Actual'),
                      Patch(facecolor='orange', edgecolor='black', label='Predicted')]
    plt.legend(handles=legend_patches)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'boxplot_actual_vs_predicted_by_month.png'))
    plt.close()

# 7. Feature Importance (if available)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center', label='Importance')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'))
    plt.close()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment('TokenCount Regression Experiments')

# Define models to test
models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Lasso Regression", Lasso()),
    ("Random Forest Regression", RandomForestRegressor(n_estimators=100, random_state=42)),
]
if has_lgb:
    models.insert(0, ("LightGBM Regression", lgb.LGBMRegressor(random_state=42)))

for model_name, model in models:
    with mlflow.start_run(run_name=model_name):
        # Fit model
        model.fit(X, y)
        y_pred = model.predict(X)
        df['y_pred'] = y_pred
        # Metrics
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('R2', r2)
        # Save model
        mlflow.sklearn.log_model(model, 'model')
        # 1. Actual vs Predicted (Time Series)
        plt.figure(figsize=(16,5))
        plt.plot(df['MoveDate'], y, label='Actual', alpha=0.7)
        plt.plot(df['MoveDate'], y_pred, label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted TokenCount (Time Series) - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('TokenCount')
        plt.legend()
        plt.tight_layout()
        plot1_path = os.path.join(REPORTS_DIR, f'actual_vs_predicted_timeseries_{model_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot1_path)
        mlflow.log_artifact(plot1_path, artifact_path='plots')
        plt.close()
        # 2. Scatter: Actual vs Predicted
        plt.figure(figsize=(6,6))
        plt.scatter(y, y_pred, alpha=0.3, label='Predicted')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
        plt.xlabel('Actual TokenCount')
        plt.ylabel('Predicted TokenCount')
        plt.title(f'Actual vs Predicted TokenCount (Scatter) - {model_name}')
        plt.legend()
        plt.tight_layout()
        plot2_path = os.path.join(REPORTS_DIR, f'actual_vs_predicted_scatter_{model_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot2_path)
        mlflow.log_artifact(plot2_path, artifact_path='plots')
        plt.close()
        # 3. Residuals plot
        residuals = y - y_pred
        plt.figure(figsize=(12,4))
        plt.plot(df['MoveDate'], residuals, alpha=0.7, label='Residuals')
        plt.axhline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f'Residuals Over Time - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.legend()
        plt.tight_layout()
        plot3_path = os.path.join(REPORTS_DIR, f'residuals_timeseries_{model_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot3_path)
        mlflow.log_artifact(plot3_path, artifact_path='plots')
        plt.close()
        # 4. Histogram of residuals
        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=50, alpha=0.7, label='Residuals')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f'Distribution of Residuals - {model_name}')
        plt.xlabel('Residual (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plot4_path = os.path.join(REPORTS_DIR, f'residuals_histogram_{model_name.replace(" ", "_").lower()}.png')
        plt.savefig(plot4_path)
        mlflow.log_artifact(plot4_path, artifact_path='plots')
        plt.close()
        # 5. MAE/RMSE by Month
        if 'month' in df.columns and 'year' in df.columns:
            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            grouped = df.groupby('year_month').apply(lambda g: pd.Series({
                'MAE': np.mean(np.abs(g['TokenCount'] - g['y_pred'])),
                'RMSE': np.sqrt(np.mean((g['TokenCount'] - g['y_pred'])**2))
            }))
            grouped = grouped.reset_index()
            plt.figure(figsize=(12,5))
            plt.plot(grouped['year_month'], grouped['MAE'], label='MAE')
            plt.plot(grouped['year_month'], grouped['RMSE'], label='RMSE')
            plt.title(f'MAE and RMSE by Month - {model_name}')
            plt.xlabel('Year-Month')
            plt.ylabel('Error')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plot5_path = os.path.join(REPORTS_DIR, f'mae_rmse_by_month_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(plot5_path)
            mlflow.log_artifact(plot5_path, artifact_path='plots')
            plt.close()
        # 6. Boxplot: Actual vs Predicted by Month
        if 'month' in df.columns and 'year' in df.columns:
            import seaborn as sns
            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            plt.figure(figsize=(16,6))
            ax = plt.gca()
            sns.boxplot(x='year_month', y='TokenCount', data=df, color='skyblue', showfliers=False, width=0.5, boxprops=dict(alpha=.5), ax=ax)
            sns.boxplot(x='year_month', y='y_pred', data=df, color='orange', showfliers=False, width=0.3, boxprops=dict(alpha=.5), ax=ax)
            plt.title(f'Actual vs Predicted TokenCount by Month (Boxplot) - {model_name}')
            plt.xlabel('Year-Month')
            plt.ylabel('TokenCount')
            plt.xticks(rotation=45)
            from matplotlib.patches import Patch
            legend_patches = [Patch(facecolor='skyblue', edgecolor='black', label='Actual'),
                              Patch(facecolor='orange', edgecolor='black', label='Predicted')]
            plt.legend(handles=legend_patches)
            plt.tight_layout()
            plot6_path = os.path.join(REPORTS_DIR, f'boxplot_actual_vs_predicted_by_month_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(plot6_path)
            mlflow.log_artifact(plot6_path, artifact_path='plots')
            plt.close()
        # 7. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10,6))
            plt.title(f'Feature Importances - {model_name}')
            plt.bar(range(len(importances)), importances[indices], align='center', label='Importance')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plot7_path = os.path.join(REPORTS_DIR, f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(plot7_path)
            mlflow.log_artifact(plot7_path, artifact_path='plots')
            plt.close()
        print(f'All plots and results for {model_name} saved and logged to MLflow.')

print('All model tests complete. Results and plots are logged to MLflow.')
