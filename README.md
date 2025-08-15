# Gate Token Prediction Pipeline

A production-ready machine learning pipeline for predicting gate token counts at Dubai Ports World terminals.

## 🏗️ Architecture Overview

This project uses a **hybrid architecture** that combines the benefits of modular, standalone agents with centralized orchestration:

### **Standalone Agents** (Modular Components)
- Each agent handles a specific pipeline stage
- Can be run independently for testing/debugging
- Fault-isolated and easy to maintain
- Located in `agents/` directory

### **Pipeline Orchestrator** (Centralized Control)
- Coordinates execution across all agents
- Provides error handling and monitoring
- Validates data flow between stages
- Generates comprehensive pipeline reports

## 🚀 Quick Start

### **Recommended: Use the Pipeline Orchestrator**

```bash
# Run the complete ML pipeline
./manage.sh pipeline

# Run pipeline with deployment to Vertica
./manage.sh pipeline-deploy

# Run a specific stage only
./manage.sh stage preprocessing
./manage.sh stage training
```

### **Available Pipeline Stages**
1. **ingestion** - Extract data from Vertica database
2. **preprocessing** - Clean and standardize raw data  
3. **splitting** - Split data temporally + feature engineering (prevents data leakage)
4. **training** - Train CatBoost model and generate predictions
5. **evaluation** - Generate evaluation reports and visualizations
6. **deployment** - Push predictions back to Vertica (disabled by default)

## 📊 Complete Workflow

### **Historical Training Pipeline**
```bash
# Option 1: Full automated pipeline
./manage.sh pipeline --mode history

# Option 2: Step-by-step execution
./manage.sh stage ingestion
./manage.sh stage preprocessing  
./manage.sh stage splitting
./manage.sh stage training
./manage.sh stage evaluation
./manage.sh stage deployment  # Optional
```

### **Real-time Prediction Pipeline**
```bash
# For real-time predictions (no training needed)
./manage.sh realtime
```

## 🛠️ Configuration

The pipeline is configured through environment variables in a `.env` file:

### **Database Connection**
```env
VERTICA_HOST=your-host
VERTICA_PORT=5433
VERTICA_DB=your-db
VERTICA_USER=your-user
VERTICA_PASSWORD=your-password
```

### **Pipeline Settings**
```env
INGEST_MODE=history                    # history | realtime
TABLE_NAME=DPW_DL.TBL_GATE_TOKENS     # Source table
TIMEZONE=Asia/Dubai                    # Local timezone

# Feature Engineering
FE_HORIZON_HOURS=1                     # Prediction horizon
FE_WINDOWS=3,6,12,24                   # Rolling window sizes
FE_KEYS=TerminalID,MoveType,Desig      # Grouping keys

# Training
TRAIN_OBJECTIVE=Poisson                # Loss function for count data
TRAIN_ITERATIONS=2000                  # CatBoost iterations
```

## 📁 Directory Structure

```
├── agents/                    # Individual ML agents (standalone)
│   ├── ingestion_agent.py    # Database extraction
│   ├── preprocessing_agent.py # Data cleaning
│   ├── split_agent.py        # Data splitting + feature engineering
│   ├── training_agent.py     # Model training
│   ├── evaluate_agent.py     # Model evaluation
│   └── vertica_push_agent.py # Deployment
├── utils/                     # Shared utilities
├── data/                      # Pipeline data
│   ├── input_raw/            # Raw extracted data
│   ├── preprocessed/         # Cleaned data
│   ├── features/             # Feature engineered data
│   └── _reports/             # Evaluation reports
├── models/                    # Trained models and predictions
├── pipeline_orchestrator.py  # Main orchestrator
├── config.py                 # Configuration management
└── manage.sh                 # CLI interface
```

## 🧪 Testing

```bash
# Test the orchestrator setup
python test_orchestrator.py

# Test individual agents
python agents/preprocessing_agent.py
python agents/training_agent.py
```

## 🎯 Key Features

### **Data Leakage Prevention**
- Uses temporal splitting (not random splitting)
- Features are engineered **separately** for each split
- Only uses historical data within each time period

### **Production Ready**
- Comprehensive error handling and logging
- Performance monitoring and metrics
- Batch processing for large datasets
- Configurable deployment strategies

### **Flexibility**
- Run full pipeline or individual stages
- Support for both historical training and real-time prediction
- Easy to add new agents or modify existing ones

## 🚨 Important Notes

### **Data Leakage Warning**
❌ **Don't use**: `./manage.sh features` (deprecated)  
✅ **Use instead**: `./manage.sh split` or `./manage.sh stage splitting`

The splitting agent prevents data leakage by splitting data **before** feature engineering.

### **Deployment Safety**
The deployment stage is **disabled by default** to prevent accidental production deployments. Enable it explicitly:

```bash
./manage.sh pipeline-deploy
# or
./manage.sh stage deployment  # (after enabling in orchestrator)
```

## 📈 Pipeline Flow

```
[Vertica Database] → [Ingestion] → [Raw CSV] → [Preprocessing] → [Clean CSV] 
     ↓
[Split Agent] → [Train/Valid/Test Features] → [Training] → [Model + Predictions]
     ↓
[Evaluation] → [Reports] + [Vertica Push] → [Production Table]
```

## 🤖 Model Details

- **Algorithm**: CatBoost Regressor
- **Objective**: Poisson (optimized for count data)
- **Features**: Calendar, lags, rolling statistics, context, quality flags
- **Validation**: Time-based splits with early stopping
- **Metrics**: RMSE, MAE, WAPE, SMAPE

## 🔍 Monitoring

Pipeline execution generates detailed reports:
- **Pipeline Report**: Overall execution status and timing
- **QA Report**: Data quality metrics after preprocessing  
- **Training Metrics**: Model performance on train/valid/test
- **Feature Importance**: Most predictive features
- **Evaluation Plots**: Actual vs predicted visualizations

## 💡 Best Practices

1. **Always use the orchestrator** for production runs
2. **Test individual agents** during development
3. **Monitor pipeline reports** for issues
4. **Review feature importance** for model insights
5. **Validate predictions** before deployment

## 📞 Support

For questions or issues, check:
1. Pipeline reports in `data/_reports/`
2. Agent logs for specific errors
3. Configuration settings in `.env`
4. This documentation for best practices