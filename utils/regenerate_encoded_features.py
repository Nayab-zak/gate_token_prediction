#!/usr/bin/env python3
"""
Regenerate encoded features using new chronological data splits
This script creates scaled features and encoded embeddings that align with the new data splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_wide_data():
    """Load the wide format data splits"""
    try:
        data_dir = Path("data/preprocessed")
        
        wide_train = pd.read_csv(data_dir / "wide_train.csv")
        wide_val = pd.read_csv(data_dir / "wide_val.csv") 
        wide_test = pd.read_csv(data_dir / "wide_test.csv")
        
        logger.info(f"Loaded wide data - Train: {len(wide_train)}, Val: {len(wide_val)}, Test: {len(wide_test)}")
        return wide_train, wide_val, wide_test
        
    except Exception as e:
        logger.error(f"Error loading wide data: {str(e)}")
        raise

def prepare_features(df):
    """Extract features from wide data (exclude timestamp)"""
    if 'timestamp' in df.columns:
        return df.drop('timestamp', axis=1)
    return df

def fit_scaler(X_train):
    """Fit StandardScaler on training data"""
    try:
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Save scaler
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        with open(models_dir / "scaler_new.pkl", 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info("Fitted and saved StandardScaler")
        return scaler
        
    except Exception as e:
        logger.error(f"Error fitting scaler: {str(e)}")
        raise

def scale_data(scaler, X_train, X_val, X_test):
    """Scale all data splits"""
    try:
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Scaled data - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    except Exception as e:
        logger.error(f"Error scaling data: {str(e)}")
        raise

def create_autoencoder(input_dim, encoding_dim=64):
    """Create autoencoder model"""
    try:
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(512, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(256, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(128, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(128, activation='relu')(encoded)
        decoded = layers.Dense(256, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(512, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Full autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info(f"Created autoencoder - Input: {input_dim}, Encoding: {encoding_dim}")
        return autoencoder, encoder
        
    except Exception as e:
        logger.error(f"Error creating autoencoder: {str(e)}")
        raise

def train_autoencoder(autoencoder, X_train_scaled, X_val_scaled):
    """Train autoencoder model"""
    try:
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=100,
            batch_size=128,
            validation_data=(X_val_scaled, X_val_scaled),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Save model
        models_dir = Path("models")
        autoencoder.save(models_dir / "autoencoder_new.h5")
        
        # Save history
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(models_dir / "training_history_new.csv", index=False)
        
        logger.info("Trained and saved autoencoder")
        return history
        
    except Exception as e:
        logger.error(f"Error training autoencoder: {str(e)}")
        raise

def generate_embeddings(encoder, X_train_scaled, X_val_scaled, X_test_scaled, 
                       train_timestamps, val_timestamps, test_timestamps):
    """Generate embeddings for all data splits"""
    try:
        # Generate embeddings
        Z_train = encoder.predict(X_train_scaled, verbose=0)
        Z_val = encoder.predict(X_val_scaled, verbose=0)
        Z_test = encoder.predict(X_test_scaled, verbose=0)
        
        # Create DataFrames with timestamps
        embed_cols = [f"embed_{i}" for i in range(Z_train.shape[1])]
        
        train_df = pd.DataFrame(Z_train, columns=embed_cols)
        train_df['timestamp'] = train_timestamps
        train_df = train_df[['timestamp'] + embed_cols]
        
        val_df = pd.DataFrame(Z_val, columns=embed_cols)
        val_df['timestamp'] = val_timestamps
        val_df = val_df[['timestamp'] + embed_cols]
        
        test_df = pd.DataFrame(Z_test, columns=embed_cols)
        test_df['timestamp'] = test_timestamps
        test_df = test_df[['timestamp'] + embed_cols]
        
        # Save embeddings
        encoded_dir = Path("data/encoded_input")
        encoded_dir.mkdir(exist_ok=True)
        
        train_df.to_csv(encoded_dir / "Z_train.csv", index=False)
        val_df.to_csv(encoded_dir / "Z_val.csv", index=False)
        test_df.to_csv(encoded_dir / "Z_test.csv", index=False)
        
        logger.info(f"Generated embeddings - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def main():
    """Main execution"""
    try:
        logger.info("Starting feature regeneration process")
        
        # Load wide data
        wide_train, wide_val, wide_test = load_wide_data()
        
        # Extract timestamps
        train_timestamps = wide_train['timestamp'].values
        val_timestamps = wide_val['timestamp'].values
        test_timestamps = wide_test['timestamp'].values
        
        # Prepare features (remove timestamp)
        X_train = prepare_features(wide_train)
        X_val = prepare_features(wide_val)
        X_test = prepare_features(wide_test)
        
        logger.info(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Fit scaler and scale data
        scaler = fit_scaler(X_train)
        X_train_scaled, X_val_scaled, X_test_scaled = scale_data(scaler, X_train, X_val, X_test)
        
        # Create and train autoencoder
        input_dim = X_train_scaled.shape[1]
        autoencoder, encoder = create_autoencoder(input_dim, encoding_dim=64)
        
        logger.info("Training autoencoder...")
        history = train_autoencoder(autoencoder, X_train_scaled, X_val_scaled)
        
        # Generate embeddings
        train_df, val_df, test_df = generate_embeddings(
            encoder, X_train_scaled, X_val_scaled, X_test_scaled,
            train_timestamps, val_timestamps, test_timestamps
        )
        
        logger.info("✅ Feature regeneration completed successfully!")
        logger.info(f"Generated embeddings:")
        logger.info(f"  - Train: {train_df.shape[0]} samples, {train_df.shape[1]-1} features")
        logger.info(f"  - Val: {val_df.shape[0]} samples, {val_df.shape[1]-1} features") 
        logger.info(f"  - Test: {test_df.shape[0]} samples, {test_df.shape[1]-1} features")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature regeneration failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
