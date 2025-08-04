# Epochs and Iterations Configuration

This document explains how to configure training epochs and iterations for the models in the gate token prediction system.

## Overview

All model training epochs and iterations are now configurable from a central location: `config.py`. This allows for:

1. Easy adjustment of training intensity for all models
2. Quick experimentation with different training durations
3. Consistent configuration across all agents
4. Fast testing with reduced epochs/iterations

## Available Configuration Parameters

### Neural Network Models

```python
# Neural Network models
LSTM_CLASSIC_EPOCHS = 50        # Number of epochs for LSTM with classic features
LSTM_AUGMENTED_EPOCHS = 50      # Number of epochs for LSTM with augmented features
MLP_CLASSIC_EPOCHS = 200        # Number of epochs for MLP with classic features
MLP_AUGMENTED_EPOCHS = 200      # Number of epochs for MLP with augmented features
```

### Tree-based Models

```python
# Tree-based model iterations
XGB_CLASSIC_ITERATIONS = 300    # Number of trees for XGBoost with classic features
XGB_AUGMENTED_ITERATIONS = 300  # Number of trees for XGBoost with augmented features
LGBM_CLASSIC_ITERATIONS = 300   # Number of trees for LightGBM with classic features
LGBM_AUGMENTED_ITERATIONS = 300 # Number of trees for LightGBM with augmented features
CATBOOST_CLASSIC_ITERATIONS = 500  # Number of trees for CatBoost with classic features
CATBOOST_AUGMENTED_ITERATIONS = 500 # Number of trees for CatBoost with augmented features
RF_CLASSIC_ESTIMATORS = 100     # Number of trees for Random Forest with classic features
RF_AUGMENTED_ESTIMATORS = 100   # Number of trees for Random Forest with augmented features
```

### Autoencoder Configuration

```python
# Autoencoder configuration
AE_MAX_EPOCHS = 100            # Number of epochs for autoencoder training
```

## Usage Recommendations

### For Development/Testing

Set epochs to low values (1-5) to quickly test the pipeline:

```python
LSTM_CLASSIC_EPOCHS = 1
XGB_CLASSIC_ITERATIONS = 1
# etc...
```

### For Production Training

Use higher values for thorough training:

```python
LSTM_CLASSIC_EPOCHS = 50
XGB_CLASSIC_ITERATIONS = 300
# etc...
```

### For Hyperparameter Tuning

When performing hyperparameter tuning, adjust these values based on the complexity of the dataset and the specific model requirements. Models with early stopping (like neural networks and gradient boosting) will automatically stop training when performance plateaus on the validation set.

## Implementation Details

Each training agent now imports its specific epoch/iteration parameter from config.py and uses it when creating the model. For example:

```python
from config import LSTM_CLASSIC_EPOCHS

# In hyperparameter loading function
params = {
    'units': 50,
    'epochs': LSTM_CLASSIC_EPOCHS,
    'batch_size': 32,
    'patience': 10
}
```

## Default Values

The default values are set in the `DEFAULT_HYPERPARAMS` dictionary in `config.py`, which ensures consistency across the system:

```python
DEFAULT_HYPERPARAMS = {
    'rf': {'n_estimators': RF_CLASSIC_ESTIMATORS, ...},
    'xgb': {'n_estimators': XGB_CLASSIC_ITERATIONS, ...},
    'lstm': {'epochs': LSTM_CLASSIC_EPOCHS, ...},
    # etc...
}
```
