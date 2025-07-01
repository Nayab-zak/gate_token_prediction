import os
from dotenv import load_dotenv

# Load environment variables from .env file (only once, when config is imported)
load_dotenv()

# Centralized config access
INPUT_FILE_PATH = os.getenv("INPUT_FILE_PATH", "data/Token_Input_data_desig.xlsx")
PREPROCESSED_FEATURES_PATH = os.getenv("PREPROCESSED_FEATURES_PATH", "data/preprocessed/preprocessed_features.csv")
PREPROCESSED_FEATURES_EDA_PATH = os.getenv("PREPROCESSED_FEATURES_EDA_PATH", "data/preprocessed/preprocessed_features_for_eda.xlsx")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
MODEL_FEATURES_PATH = os.getenv("MODEL_FEATURES_PATH", "models/best_model_features.json")
BEST_PARAMS_PATH = os.getenv("BEST_PARAMS_PATH", "models/best_params.json")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")
BASELINE_METRICS_PATH = os.getenv("BASELINE_METRICS_PATH", "outputs/baseline_metrics.json")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "exp1")
DASHBOARD_DATA_PATH = os.getenv("DASHBOARD_DATA_PATH", "data/preprocessed/preprocessed_features.csv")
EDA_PATH = os.getenv("EDA_PATH", "data/preprocessed/preprocessed_features_for_eda.xlsx")
