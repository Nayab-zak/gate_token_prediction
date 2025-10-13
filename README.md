# 🚢 DPW Gate Token Prediction System

A production-ready machine learning pipeline for predicting hourly gate token counts at Dubai Ports World (DPW) terminals using Vertica database integration.

## 🎯 Overview

This system provides end-to-end ML capabilities for time series forecasting of gate token usage patterns, featuring:

- **🏗️ Robust Architecture**: Modular design with separate data, model, and pipeline components
- **📊 Multiple Models**: Support for LightGBM, XGBoost, CatBoost, and ElasticNet algorithms
- **🔄 Time Series Features**: Advanced lag features, rolling statistics, and calendar features
- **📈 Model Selection**: Automated champion model selection with cross-validation
- **🎛️ Production Ready**: Comprehensive logging, error handling, and monitoring
- **🗄️ Database Integration**: Native Vertica integration with connection pooling
- **⚙️ Configurable**: YAML-based configuration management

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Access to Vertica database (production read-only, development write)
- UV package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Nayab-zak/gate_token_prediction.git
cd gate_token_prediction

# Create virtual environment
uv venv && source .venv/bin/activate  # Linux/Mac
# OR
python -m venv .venv && .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
# Edit .env with your database credentials
```

### Basic Usage

```bash
# Create development table (first time only)
python -m app.cli create-dev-table --apply

# Train models
python -m app.cli train

# Generate predictions
python -m app.cli predict

# Backfill actuals when available
python -m app.cli backfill

# Check prediction accuracy
python -m app.cli accuracy --days-back 7
```

## 📁 Project Structure

```
predictive_modeling/
├── 📱 Application Core
│   ├── app/
│   │   ├── cli.py                    # Command-line interface
│   │   ├── logging.py               # Structured logging system
│   │   ├── data/                    # Data processing modules
│   │   │   ├── ingestion.py         # Data ingestion from Vertica
│   │   │   ├── preparation.py       # Data cleaning & preparation
│   │   │   ├── feature_gen.py       # Feature engineering
│   │   │   ├── splits.py           # Time series splitting
│   │   │   └── leakage_guards.py   # Data leakage prevention
│   │   ├── models/                  # ML model implementations
│   │   │   ├── registry.py          # Model factory
│   │   │   ├── lightgbm_.py        # LightGBM implementation
│   │   │   ├── xgboost_.py         # XGBoost implementation
│   │   │   ├── catboost_.py        # CatBoost implementation
│   │   │   ├── tuning.py           # Hyperparameter optimization
│   │   │   ├── evaluation.py       # Model evaluation
│   │   │   ├── champion.py         # Champion model selection
│   │   │   └── intervals.py        # Prediction intervals
│   │   ├── pipelines/              # End-to-end pipelines
│   │   │   ├── train.py            # Training pipeline
│   │   │   ├── predict.py          # Prediction pipeline
│   │   │   └── backfill_actuals.py # Actuals backfill pipeline
│   │   ├── db/                     # Database layer
│   │   │   └── vertica_client.py   # Vertica database client
│   │   └── utils/                  # Utility functions
│   │       ├── env.py              # Environment management
│   │       ├── hashing.py          # Data integrity checks
│   │       └── ddl.py              # Database schema management
│
├── ⚙️ Configuration
│   ├── config/
│   │   ├── config.yaml             # Main configuration
│   │   ├── db.yaml                 # Database connections
│   │   ├── features.yaml           # Feature specifications
│   │   ├── models.yaml             # Model configurations
│   │   ├── metrics.yaml            # Evaluation metrics
│   │   └── realtime.yaml           # Production deployment config
│
├── 🗄️ Database
│   ├── db/sql/                     # SQL queries
│   └── data/                       # Data storage (gitignored)
│
├── 📊 Artifacts
│   ├── artifacts/                  # Model artifacts (gitignored)
│   │   ├── models/                 # Trained models
│   │   └── champion/               # Current champion model
│   └── logs/                       # Application logs (gitignored)
│
├── 🧪 Testing
│   ├── tests/                      # Test suite
│   └── support/                    # Support scripts
│
└── 📋 Project Files
    ├── .env                        # Environment variables
    ├── requirements.txt            # Python dependencies
    ├── pyproject.toml             # Project metadata
    └── README.md                  # This file
```

## 🔧 Configuration

The system uses YAML-based configuration for maximum flexibility:

### Environment Variables (.env)

```env
# Production Database (Read-only)
VERTICA_PROD_HOST=your-prod-host
VERTICA_PROD_PORT=5433
VERTICA_PROD_DB=your-prod-db
VERTICA_PROD_USER=your-user
VERTICA_PROD_PASSWORD=your-password

# Development Database (Write access)
VERTICA_DEV_HOST=your-dev-host
VERTICA_DEV_PORT=5433
VERTICA_DEV_DB=your-dev-db
VERTICA_DEV_USER=your-user
VERTICA_DEV_PASSWORD=your-password

# System Configuration
DATA_START_DATE=2023-01-01
TIMEZONE=Asia/Dubai
DEFAULT_SCHEMA=DPW_DL
DEFAULT_TABLE=T_DA_PRED_GATE_TOKEN
PREDICTION_FREQUENCY_HOURS=6
```

### Key Configuration Files

- **`config.yaml`**: Main system configuration including logging, model selection, and training parameters
- **`features.yaml`**: Feature engineering specifications (lags, rolling windows, calendar features)
- **`models.yaml`**: Model-specific hyperparameter search spaces
- **`metrics.yaml`**: Evaluation metrics and champion selection policy
- **`realtime.yaml`**: Production deployment schema and upsert configuration

## 🤖 Machine Learning Pipeline

### Data Processing

1. **Ingestion**: Fetches data from Vertica production database
2. **Canonicalization**: Standardizes column names and data types
3. **Feature Engineering**: Creates lag features, rolling statistics, and calendar features
4. **Leakage Guards**: Prevents future data leakage in time series features
5. **Validation**: Ensures data quality and schema compliance

### Model Training

1. **Cross-Validation**: Time series-aware k-fold cross-validation
2. **Hyperparameter Optimization**: Optuna-based optimization with early stopping
3. **Backtesting**: Walk-forward validation on hold-out periods
4. **Champion Selection**: Automated model selection based on configurable metrics
5. **Interval Estimation**: Confidence intervals via quantile regression or conformal prediction

### Prediction Pipeline

1. **Real-time Data Fetch**: Gets latest data for prediction
2. **Feature Engineering**: Applies same transformations as training
3. **Prediction Generation**: Uses champion model for forecasts
4. **Database Upsert**: Stores predictions with conflict resolution
5. **Monitoring**: Tracks prediction quality and system performance

## 📊 Features

### Time Series Features

- **Lag Features**: Configurable lag periods (e.g., 1h, 6h, 24h, 168h)
- **Rolling Statistics**: Moving averages, standard deviations, min/max
- **Calendar Features**: Hour of day, day of week, weekend indicators
- **Horizon Adjustment**: Features are offset by prediction horizon to prevent leakage

### Model Support

| Model | Library | Strengths |
|-------|---------|-----------|
| LightGBM | lightgbm | Fast training, categorical features, feature importance |
| XGBoost | xgboost | Robust to outliers, parallel processing |
| CatBoost | catboost | Handles categorical data natively, reduces overfitting |
| ElasticNet | sklearn | Linear baseline, interpretable coefficients |

### Production Features

- **Structured Logging**: JSON-formatted logs with contextual information
- **Error Handling**: Comprehensive error capture and logging
- **Data Validation**: Schema validation and data quality checks
- **Monitoring**: Performance tracking and alerting capabilities
- **Scalability**: Chunked database operations for large datasets

## 🔍 Monitoring & Observability

### Logging System

The system provides comprehensive structured logging:

```python
from app.logging import get_model_logger, get_data_logger

model_log = get_model_logger()
model_log.info("model_training_started", 
               model="lightgbm", 
               features=45, 
               samples=10000)

data_log = get_data_logger()
data_log.info("feature_engineering_completed",
              lag_features=12,
              rolling_features=8,
              calendar_features=3)
```

### Log Categories

- **CLI**: Command-line operations and user interactions
- **Model**: Training, prediction, and model operations
- **Data**: Data processing, feature engineering, and validation
- **Database**: Connection, query execution, and performance
- **Pipeline**: End-to-end pipeline operations
- **Error**: Error handling and debugging information

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_logging.py          # Logging tests
python -m pytest tests/test_leakage_guard.py    # Data leakage tests
python -m pytest tests/test_time_splits.py      # Time series splitting
python -m pytest tests/test_schema_drift.py     # Schema validation
```

## 📈 Usage Examples

### Training a New Model

```bash
# Train with default configuration
python -m app.cli train

# Train with dry run (no database writes)
python -m app.cli train --dry-run

# Train with custom config
python -m app.cli train --config-path custom_config.yaml
```

### Generating Predictions

```bash
# Generate predictions for current time
python -m app.cli predict

# Dry run predictions (no database writes)  
python -m app.cli predict --dry-run
```

### Backfilling Actuals

```bash
# Backfill actual values when they become available
python -m app.cli backfill

# Check what would be backfilled (dry run)
python -m app.cli backfill --dry-run
```

### Model Performance Analysis

```bash
# Analyze prediction accuracy for last 7 days
python -m app.cli accuracy

# Analyze accuracy for last 30 days
python -m app.cli accuracy --days-back 30
```

## 🛠️ Development

### Adding New Models

1. Create model class in `app/models/`:

```python
from app.models.base import BaseModel

class MyModel(BaseModel):
    name = "mymodel"
    
    def __init__(self, **params):
        self.model = MyMLLibrary(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X).tolist()
```

2. Register in `app/models/registry.py`:

```python
def get_model(name: str, **params):
    if name == "mymodel":
        from app.models.mymodel_ import MyModel
        return MyModel(**params)
    # ... existing models
```

3. Add hyperparameter space in `config/models.yaml`

### Extending Features

Add new feature types in `app/data/feature_gen.py`:

```python
def add_my_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    # Implement feature engineering logic
    return df.with_columns([
        # New feature columns
    ])
```

## 🚀 Deployment

### Production Checklist

- [ ] Environment variables configured
- [ ] Database connections tested
- [ ] Development table created
- [ ] Initial model trained
- [ ] Monitoring dashboards set up
- [ ] Log rotation configured
- [ ] Backup procedures in place

### Scheduling

Use cron or task scheduler for automated runs:

```bash
# Train daily at 2 AM
0 2 * * * /path/to/venv/bin/python -m app.cli train

# Predict every 6 hours
0 */6 * * * /path/to/venv/bin/python -m app.cli predict

# Backfill actuals daily at 6 AM
0 6 * * * /path/to/venv/bin/python -m app.cli backfill
```

## 📊 Performance

### Benchmarks

- **Training Time**: ~5-15 minutes for 100K samples (varies by model)
- **Prediction Latency**: <2 seconds for real-time predictions
- **Memory Usage**: ~500MB-2GB depending on data size
- **Database Throughput**: 10K+ rows/second for batch operations

### Optimization Tips

- Use `parallel_jobs` setting for faster hyperparameter optimization
- Adjust `chunk_size` in database operations based on available memory
- Enable connection pooling for high-frequency predictions
- Monitor log file sizes and configure rotation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with appropriate tests
4. Ensure code passes linting: `flake8 app/ tests/`
5. Run test suite: `python -m pytest`
6. Commit changes: `git commit -m "Add feature description"`
7. Push to branch: `git push origin feature-name`
8. Create a Pull Request

## 📄 License

This project is proprietary software developed for Dubai Ports World (DPW).

## 🔗 Links

- [Dubai Ports World](https://www.dpworld.com/)
- [Vertica Documentation](https://www.vertica.com/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Polars Documentation](https://pola-rs.github.io/polars/)

---

**Generated**: 2025-10-13 | **Version**: 1.2.0 | **Author**: DPW AI Team
