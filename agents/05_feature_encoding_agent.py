import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configure TensorFlow to use CPU only (disable GPU completely to avoid OOM)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


def setup_logger():
    logger = logging.getLogger('05_feature_encoding_agent')
    logger.setLevel(logging.INFO)
    os.makedirs('logs/agents', exist_ok=True)
    fh = logging.FileHandler('logs/agents/05_feature_encoding_agent.log')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_features(split):
    path = os.path.join(DATA_DIR, 'features', f'{split}_features.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_csv(path, parse_dates=['datetime'])


def build_autoencoder(input_dim, latent_dim=16):
    """Build autoencoder - CPU only version"""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(latent)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    autoencoder = Model(inputs, outputs, name='autoencoder')
    encoder = Model(inputs, latent, name='encoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    print("✅ Autoencoder built successfully on CPU")
    return autoencoder, encoder


def process_encoding(split, scaler, encoder_model, autoencoder_model, logger):
    df = load_features(split)
    logger.info(f"Loaded {split} features, shape={df.shape}")

    # One-hot encode categorical columns
    cat_cols = ['MoveType', 'TerminalID', 'Desig']
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Keep datetime and target
    datetime_idx = df_enc['datetime'].reset_index(drop=True)
    y = df_enc['TokenCount'].values
    X = df_enc.drop(columns=['datetime', 'TokenCount']).values

    # Drop all non-numeric columns before scaling (must match training)
    non_numeric = ['datetime', 'MoveDate', 'outlier_flag', 'TokenCount']
    drop_cols = [col for col in non_numeric if col in df_enc.columns]
    X_to_scale = df_enc.drop(columns=drop_cols).values
    X_scaled = scaler.transform(X_to_scale)

    # Autoencoder outputs - CPU only
    latents = encoder_model.predict(X_scaled)
    recon = autoencoder_model.predict(X_scaled)
    
    recon_err = np.mean(np.square(X_scaled - recon), axis=1).reshape(-1, 1)

    # Prepare classic vs augmented arrays
    X_classic = X_scaled
    X_augmented = np.hstack([X_scaled, latents, recon_err])

    # Build meaningful column names based on original features
    # Base features: 45 features from the original CSV (excluding categorical and metadata)
    base_feature_names = [
        'MoveHour', 'outlier_flag', 'is_weekend', 'is_friday', 'is_holiday',
        'hour_of_day', 'day_of_week', 'month',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_4h', 'lag_5h', 'lag_6h', 
        'lag_7h', 'lag_8h', 'lag_9h', 'lag_10h', 'lag_11h', 'lag_12h',
        'lag_13h', 'lag_14h', 'lag_15h', 'lag_16h', 'lag_17h', 'lag_18h',
        'lag_19h', 'lag_20h', 'lag_21h', 'lag_22h', 'lag_23h', 'lag_24h',
        'roll_mean_3h', 'roll_std_3h', 'roll_mean_6h', 'roll_std_6h',
        'roll_mean_12h', 'roll_std_12h', 'roll_mean_24h', 'roll_std_24h',
        'diff_1h', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    # Categorical features: Based on one-hot encoding with drop_first=True
    # Expected: MoveType=[In,Out]->1, TerminalID=[T1,T2,T3,T4]->3, Desig=[EXP,FCL,MT,T/S]->3
    # But actual encoded data has 51 features, so we'll adapt dynamically
    expected_categorical_features = [
        'MoveType_Out',  # Reference: In (drop_first=True)
        'TerminalID_T2', 'TerminalID_T3', 'TerminalID_T4',  # Reference: T1
        'Desig_FCL', 'Desig_MT', 'Desig_T/S'  # Reference: EXP
    ]
    
    # Calculate expected vs actual
    expected_total = len(base_feature_names) + len(expected_categorical_features)
    actual_total = X_classic.shape[1]
    
    logger.info(f"Feature count analysis:")
    logger.info(f"  Base features: {len(base_feature_names)}")
    logger.info(f"  Expected categorical features: {len(expected_categorical_features)}")
    logger.info(f"  Expected total: {expected_total}")
    logger.info(f"  Actual total: {actual_total}")
    logger.info(f"  Difference: {expected_total - actual_total}")
    
    # Handle the column count mismatch gracefully
    if expected_total == actual_total:
        cols_classic = base_feature_names + expected_categorical_features
        logger.info("✅ Feature count matches expected. Using meaningful column names.")
    else:
        # Use generic names but log the issue
        cols_classic = [f'feature_{i}' for i in range(actual_total)]
        logger.warning(f"⚠️ Feature count mismatch! Using generic column names.")
        logger.warning(f"   This may indicate changes in categorical values or encoding logic.")
    
    df_classic = pd.DataFrame(X_classic, columns=cols_classic)
    cols_aug = cols_classic + [f'latent_{i}' for i in range(latents.shape[1])] + ['recon_error']
    df_augmented = pd.DataFrame(X_augmented, columns=cols_aug)

    # Add datetime and target
    df_classic.insert(0, 'datetime', datetime_idx)
    df_classic['TokenCount'] = y
    df_augmented.insert(0, 'datetime', datetime_idx)
    df_augmented['TokenCount'] = y

    # Round all float columns to 4 decimals before saving
    float_cols_classic = df_classic.select_dtypes(include=['float']).columns
    df_classic[float_cols_classic] = df_classic[float_cols_classic].round(4)
    float_cols_aug = df_augmented.select_dtypes(include=['float']).columns
    df_augmented[float_cols_aug] = df_augmented[float_cols_aug].round(4)

    # Save encoded inputs & outputs
    in_dir = os.path.join(DATA_DIR, 'encoded_input')
    out_dir = os.path.join(DATA_DIR, 'encoded_output')
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Classic
    in_classic = os.path.join(in_dir, f'{split}_input_classic.csv')
    out_classic = os.path.join(out_dir, f'{split}_output_classic.csv')
    df_classic.to_csv(in_classic, index=False)
    df_classic[['datetime', 'TokenCount']].to_csv(out_classic, index=False)
    logger.info(f"Saved classic input to {in_classic} and output to {out_classic}")

    # Augmented
    in_aug = os.path.join(in_dir, f'{split}_input_augmented.csv')
    out_aug = os.path.join(out_dir, f'{split}_output_augmented.csv')
    df_augmented.to_csv(in_aug, index=False)
    df_augmented[['datetime', 'TokenCount']].to_csv(out_aug, index=False)
    logger.info(f"Saved augmented input to {in_aug} and output to {out_aug}")


def main():
    logger = setup_logger()
    logger.info("Starting feature encoding with autoencoder...")

    # Force CPU-only training to avoid GPU memory issues
    print("🔧 Using CPU-only mode to avoid GPU memory issues")

    # Load train features
    df_train = load_features('train')

    # One-hot then scale on train
    cat_cols = ['MoveType', 'TerminalID', 'Desig']
    df_train_enc = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
    # Convert boolean columns to int so they are included in scaling
    for col in ['is_weekend', 'is_friday', 'is_holiday']:
        if col in df_train_enc.columns:
            df_train_enc[col] = df_train_enc[col].astype(int)
    # Drop only non-numeric columns that cannot be encoded
    non_numeric = ['datetime', 'MoveDate', 'outlier_flag']
    X_train = df_train_enc.drop(columns=[col for col in non_numeric if col in df_train_enc.columns] + ['TokenCount']).values
    scaler = StandardScaler().fit(X_train)

    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder(X_train.shape[1])
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    logger.info(f"Training autoencoder on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    logger.info("Autoencoder configuration:")
    logger.info(f"  - Input dimension: {X_train.shape[1]}")
    logger.info(f"  - Latent dimension: 16")
    logger.info(f"  - Early stopping patience: 15 epochs")
    logger.info(f"  - Learning rate reduction patience: 8 epochs")
    logger.info(f"  - Maximum epochs: 100")
    
    # CPU-only training
    history = autoencoder.fit(
        scaler.transform(X_train), scaler.transform(X_train),
        epochs=100, batch_size=32, validation_split=0.15,
        callbacks=[early_stop, reduce_lr], verbose=1
    )
    logger.info("✅ Autoencoder training completed on CPU")
    
    final_val_loss = min(history.history['val_loss'])
    final_epoch = len(history.history['val_loss'])
    logger.info(f"✅ Autoencoder training completed successfully!")
    logger.info(f"  - Final epoch: {final_epoch}")
    logger.info(f"  - Final validation loss: {final_val_loss:.6f}")
    logger.info(f"  - Training stopped via: {'Early stopping' if final_epoch < 100 else 'Max epochs reached'}")

    # Process train and test splits
    process_encoding('train', scaler, encoder, autoencoder, logger)
    process_encoding('test', scaler, encoder, autoencoder, logger)

    logger.info("Feature encoding completed.")


if __name__ == '__main__':
    main()
