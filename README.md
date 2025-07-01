# Predictive Modeling Token Prediction

## Project Overview
This project implements a modular, reproducible pipeline for token prediction using a series of agents for data loading, sanitization, preprocessing, feature engineering, modeling, and evaluation. The workflow is orchestrated via CLI scripts and a `manage.sh` utility for end-to-end automation.

## Pipeline Flow

1. **Data Preparation (`prep`)**
   - Run: `./manage.sh prep`
   - Loads and sanitizes raw data using `DataLoaderAgent` and `SanitizationAgent`.
   - Output: Cleaned/preprocessed data in `data/preprocessed/`.

2. **Feature Engineering (`features`)**
   - Run: `./manage.sh features`
   - Applies `FeatureEngineerAgent` to generate features (lags, rolling stats, holiday flags, etc.).
   - Output: Feature matrix in `data/preprocessed/`.

3. **EDA (`eda`)**
   - Run: `./manage.sh eda`
   - Launches an interactive dashboard for exploratory data analysis using Streamlit.

4. **Model Training (`train`)**
   - Run: `./manage.sh train`
   - Trains baseline and advanced models (GLM, LightGBM, XGBoost) using `ModelingAgent`.
   - Saves best model and parameters to `models/`.

5. **Evaluation (`evaluate`)**
   - Run: `./manage.sh evaluate`
   - Evaluates model on holdout data using `EvaluationAgent`.
   - Generates metrics, slice breakdowns, rolling backtests, and PSI drift analysis.
   - Outputs reports and metrics to `outputs/` and `reports/`.

6. **Plots/Visualizations (`plots`)**
   - Run: `./manage.sh plots`
   - Generates additional plots (if implemented in `scripts/plots.py`).

7. **Prediction (`predict`)**
   - Run: `./manage.sh predict`
   - Runs prediction on new data using the trained model.

## CLI Orchestration
All steps above are orchestrated via `manage.sh`:

```bash
./manage.sh [prep|features|eda|train|evaluate|plots|predict]
```

## Logging & Reproducibility
- All agents use robust logging (file + console) via `utils/logger.py`.
- Footsteps tracking for pipeline steps.
- Test coverage for all major agents in `tests/`.
- Configuration and feature documentation in `config/`.

## Directory Structure
- `agents/` - Modular pipeline agents
- `scripts/` - CLI scripts for each pipeline step
- `data/` - Raw and processed data
- `models/` - Saved models and parameters
- `outputs/` - Metrics and results
- `logs/` - Centralized logs
- `tests/` - Unit tests
- `utils/` - Utilities (logging, feature serialization, PSI, etc.)

## Summary of Flow
- The pipeline is fully modular and reproducible.
- Each step is test-covered and can be run independently or as a workflow.
- Logging and error handling are robust.
- CLI (`manage.sh`) enables easy orchestration for end users.

For more details, see the docstrings in each agent and the `docs/` folder.
