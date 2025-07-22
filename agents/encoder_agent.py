#!/usr/bin/env python3
"""
Encoder Agent - Train autoencoder and create embeddings
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class EncoderAgent:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "encoder_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EncoderAgent')
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def load_scaled_data(self, file_path):
        """Load scaled data and separate features from timestamp"""
        try:
            self.logger.info(f"Loading scaled data from: {file_path}")
            df = pd.read_csv(file_path)
            
            # Separate timestamp and features
            if 'timestamp' in df.columns:
                timestamps = df['timestamp']
                features = df.drop('timestamp', axis=1)
            else:
                timestamps = None
                features = df
            
            self.logger.info(f"Loaded {len(features)} rows with {features.shape[1]} features")
            return features, timestamps
            
        except Exception as e:
            self.logger.error(f"Error loading scaled data: {str(e)}")
            raise
    
    def build_autoencoder(self, input_dim, encoding_dim=64):
        """Build dense autoencoder architecture with better numerical stability"""
        try:
            self.logger.info(f"Building autoencoder - Input dim: {input_dim}, Encoding dim: {encoding_dim}")
            
            # Input layer
            input_layer = keras.Input(shape=(input_dim,))
            
            # Encoder layers with batch normalization for stability
            x = layers.Dense(512, kernel_initializer='glorot_uniform')(input_layer)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)
            
            x = layers.Dense(256, kernel_initializer='glorot_uniform')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)
            
            x = layers.Dense(128, kernel_initializer='glorot_uniform')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Encoding layer without activation to allow negative values
            encoded = layers.Dense(encoding_dim, kernel_initializer='glorot_uniform', name='encoded')(x)
            
            # Decoder layers
            x = layers.Dense(128, kernel_initializer='glorot_uniform')(encoded)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)
            
            x = layers.Dense(256, kernel_initializer='glorot_uniform')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.1)(x)
            
            x = layers.Dense(512, kernel_initializer='glorot_uniform')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Output layer
            decoded = layers.Dense(input_dim, kernel_initializer='glorot_uniform', activation='linear')(x)
            
            # Full autoencoder model
            autoencoder = keras.Model(input_layer, decoded, name='autoencoder')
            
            # Encoder model (for extracting embeddings)
            encoder = keras.Model(input_layer, encoded, name='encoder')
            
            # Compile with lower learning rate for stability
            optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            autoencoder.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"Autoencoder architecture:")
            autoencoder.summary(print_fn=self.logger.info)
            
            return autoencoder, encoder
            
        except Exception as e:
            self.logger.error(f"Error building autoencoder: {str(e)}")
            raise
    
    def validate_data(self, X_train, X_val, X_test):
        """Validate data for training issues"""
        try:
            self.logger.info("Validating input data")
            
            for name, data in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
                # Check for NaN values
                nan_count = np.isnan(data).sum()
                if nan_count > 0:
                    self.logger.warning(f"{name} data has {nan_count} NaN values")
                    # Replace NaN with 0
                    data[np.isnan(data)] = 0
                
                # Check for infinite values
                inf_count = np.isinf(data).sum()
                if inf_count > 0:
                    self.logger.warning(f"{name} data has {inf_count} infinite values")
                    # Replace inf with finite values
                    data[np.isinf(data)] = 0
                
                # Check for extremely large values
                max_val = np.abs(data).max()
                if max_val > 1e6:
                    self.logger.warning(f"{name} data has very large values (max: {max_val})")
                
                self.logger.info(f"{name} data - Shape: {data.shape}, Range: [{data.min():.4f}, {data.max():.4f}]")
            
            return X_train, X_val, X_test
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            raise

    def train_autoencoder(self, autoencoder, X_train, X_val, model_path, epochs=100, batch_size=256):
        """Train autoencoder with early stopping"""
        try:
            self.logger.info("Training autoencoder")
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Train model
            history = autoencoder.fit(
                X_train, X_train,  # Autoencoder predicts its input
                validation_data=(X_val, X_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            best_val_loss = min(history.history['val_loss']) if history.history['val_loss'] else float('inf')
            self.logger.info(f"Training completed. Best val_loss: {best_val_loss:.6f}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error training autoencoder: {str(e)}")
            raise
    
    def create_embeddings(self, encoder, features, output_path, timestamps=None):
        """Create embeddings using encoder"""
        try:
            self.logger.info(f"Creating embeddings and saving to: {output_path}")
            
            # Generate embeddings
            embeddings = encoder.predict(features, verbose=0)
            
            # Create DataFrame with embeddings
            embedding_cols = [f'embed_{i}' for i in range(embeddings.shape[1])]
            embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
            
            # Add timestamp if available
            if timestamps is not None:
                embeddings_df['timestamp'] = timestamps.values
                # Reorder columns to have timestamp first
                cols = ['timestamp'] + embedding_cols
                embeddings_df = embeddings_df[cols]
            
            # Save embeddings
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Saved embeddings with shape {embeddings_df.shape} to {output_path}")
            return embeddings_df
            
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def run(self, train_path, val_path, test_path, model_path, output_dir, encoding_dim=64, epochs=100):
        """Main execution method"""
        try:
            self.logger.info("Starting encoder training process")
            
            # Load scaled data
            X_train, train_timestamps = self.load_scaled_data(train_path)
            X_val, val_timestamps = self.load_scaled_data(val_path) 
            X_test, test_timestamps = self.load_scaled_data(test_path)
            
            # Convert to numpy arrays and validate
            X_train_np = X_train.values
            X_val_np = X_val.values
            X_test_np = X_test.values
            
            # Validate data
            X_train_np, X_val_np, X_test_np = self.validate_data(X_train_np, X_val_np, X_test_np)
            
            # Build autoencoder
            autoencoder, encoder = self.build_autoencoder(X_train_np.shape[1], encoding_dim)
            
            # Train autoencoder
            history = self.train_autoencoder(autoencoder, X_train_np, X_val_np, model_path, epochs)
            
            # Create output paths
            output_paths = {
                'train': Path(output_dir) / "Z_train.csv",
                'val': Path(output_dir) / "Z_val.csv",
                'test': Path(output_dir) / "Z_test.csv"
            }
            
            # Create embeddings for all splits
            for split_name, features, timestamps, out_path in [
                ('train', X_train_np, train_timestamps, output_paths['train']),
                ('val', X_val_np, val_timestamps, output_paths['val']),
                ('test', X_test_np, test_timestamps, output_paths['test'])
            ]:
                self.create_embeddings(encoder, features, out_path, timestamps)
            
            # Save training history
            history_path = Path(model_path).parent / "training_history.csv"
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(history_path, index=False)
            self.logger.info(f"Saved training history to: {history_path}")
            
            self.logger.info("Encoder training process completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Encoder training process failed: {str(e)}")
            return False

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Encoder Agent')
    parser.add_argument('--train-path', required=True, help='Path to training scaled CSV')
    parser.add_argument('--val-path', required=True, help='Path to validation scaled CSV')
    parser.add_argument('--test-path', required=True, help='Path to test scaled CSV')
    parser.add_argument('--model-path', required=True, help='Path to save autoencoder model')
    parser.add_argument('--output-dir', required=True, help='Output directory for embeddings')
    parser.add_argument('--encoding-dim', type=int, default=64, help='Encoding dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    args = parser.parse_args()
    
    agent = EncoderAgent()
    success = agent.run(
        args.train_path, args.val_path, args.test_path,
        args.model_path, args.output_dir,
        args.encoding_dim, args.epochs
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
