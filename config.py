import os

# Directory settings
DATA_DIR = os.getenv('DATA_DIR', 'data')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')

# Ingestion source directory (optional override via .env)
INGESTION_SOURCE_DIR = os.getenv('INGESTION_SOURCE_DIR', os.path.join(DATA_DIR, 'incoming'))

# UAE public holidays list (ISO format)
# This list can be extended or overridden
# UAE_HOLIDAYS = [
#     '2022-12-02',  # UAE National Day
#     '2023-01-01',  # New Year's Day
#     '2023-06-28',  # Eid al-Adha (approximate)
#     '2023-06-29',  # Eid al-Adha
#     '2023-12-02',  # UAE National Day
#     # Add additional holiday dates as needed
# ]

UAE_HOLIDAYS = []

# Top features for the "What-If" slider in dashboards
TOP_IF_SLIDER = ['roll_mean_6', 'lag_1', 'is_friday']

# Test split configuration: number of months for holdout
TEST_SPLIT_MONTHS = 6

# Forecast horizon in hours
FORECAST_HORIZON = 24

# Autoencoder configuration
AE_LATENT_DIM = 16
AE_HIDDEN_DIMS = [128, 64]
AE_REGULARIZATION = 1e-4
AE_VALIDATION_SPLIT = 0.1
AE_EARLY_STOPPING_PATIENCE = 10
AE_MAX_EPOCHS = 100
AE_BATCH_SIZE = 32

# Training defaults for models (fallback hyperparameters)
DEFAULT_HYPERPARAMS = {
    'rf': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
    'xgb': {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42},
    'catboost': {'iterations': 500, 'learning_rate': 0.1, 'depth': 6, 'random_seed': 42, 'verbose': False},
    'lgbm': {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42},
    'mlp': {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001, 'max_iter': 200, 'random_state': 42},
    'lstm': {'units': 50, 'epochs': 50, 'batch_size': 32, 'patience': 10},
}
