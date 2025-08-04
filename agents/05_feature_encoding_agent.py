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


def process_encoding(split, scaler, encoder_model, autoencoder_model, logger, reference_levels=None):
    """
    Process feature encoding for a data split (train, validation, or test)
    
    Args:
        split: String identifier ('train', 'validation', 'test')
        scaler: Fitted StandardScaler
        encoder_model: Trained encoder model
        autoencoder_model: Trained autoencoder model
        logger: Logger instance
        reference_levels: Dict of reference levels for categorical variables (if None, will be inferred)
    """
    df = load_features(split)
    logger.info(f"Loaded {split} features, shape={df.shape}")

    # One-hot encode categorical columns - with consistent reference levels
    cat_cols = ['MoveType', 'TerminalID', 'Desig']
    
    # To ensure consistent encoding across train/validation/test, we need to:
    # 1. Generate all possible categories first
    # 2. Create dummies with all categories specified
    
    # Define reference levels - either use provided ones or infer from training data
    if reference_levels:
        move_types = reference_levels['move_types']
        terminals = reference_levels['terminals'] 
        desig_types = reference_levels['desig_types']
        logger.info("Using provided reference levels for categorical encoding")
    else:
        # Define reference levels based on the training data
        move_types = ['In', 'Out']
        terminals = ['T1', 'T2', 'T3', 'T4']
        desig_types = ['EXP', 'FCL', 'MT', 'T/S']
        logger.info("Using default reference levels for categorical encoding")
    
    # Create explicit dummies with consistent references
    df_enc = df.copy()
    
    # Handle each categorical column separately
    df_enc = pd.get_dummies(df_enc, columns=['MoveType'], prefix=['MoveType'])
    df_enc = pd.get_dummies(df_enc, columns=['TerminalID'], prefix=['TerminalID'])
    df_enc = pd.get_dummies(df_enc, columns=['Desig'], prefix=['Desig'])
    
    # Ensure all expected columns exist (for consistent feature counts)
    for move_type in move_types:
        col = f'MoveType_{move_type}'
        if col not in df_enc.columns:
            df_enc[col] = 0
    
    for terminal in terminals:
        col = f'TerminalID_{terminal}'
        if col not in df_enc.columns:
            df_enc[col] = 0
            
    for desig in desig_types:
        col = f'Desig_{desig}'
        if col not in df_enc.columns:
            df_enc[col] = 0
    
    # Drop the first category for each to match original behavior
    if 'MoveType_In' in df_enc.columns:
        df_enc = df_enc.drop(columns=['MoveType_In'])
    if 'TerminalID_T1' in df_enc.columns:
        df_enc = df_enc.drop(columns=['TerminalID_T1'])
    if 'Desig_EXP' in df_enc.columns:
        df_enc = df_enc.drop(columns=['Desig_EXP'])
    
    # Keep datetime and target
    datetime_idx = df_enc['datetime'].reset_index(drop=True)
    y = df_enc['TokenCount'].values
    X = df_enc.drop(columns=['datetime', 'TokenCount']).values

    # Drop all non-numeric columns before scaling (must match training)
    non_numeric = ['datetime', 'MoveDate', 'outlier_flag', 'TokenCount']
    drop_cols = [col for col in non_numeric if col in df_enc.columns]
    X_to_scale = df_enc.drop(columns=drop_cols).values
    
    # Log column count and column names to help debug
    logger.info(f"[{split}] Feature count before scaling: {X_to_scale.shape[1]}")
    df_cols = df_enc.drop(columns=drop_cols).columns.tolist()
    logger.info(f"[{split}] Columns: {df_cols}")
    
    # For validation/test sets, ensure they have exactly the same columns as the training data
    if split != 'train':
        # Get training columns by loading training data with same processing
        train_df = load_features('train')
        train_df_enc = train_df.copy()
        
        # Process train with same categorical handling
        train_df_enc = pd.get_dummies(train_df_enc, columns=['MoveType'], prefix=['MoveType'])
        train_df_enc = pd.get_dummies(train_df_enc, columns=['TerminalID'], prefix=['TerminalID'])
        train_df_enc = pd.get_dummies(train_df_enc, columns=['Desig'], prefix=['Desig'])
        
        # Apply same column dropping
        for move_type in move_types:
            col = f'MoveType_{move_type}'
            if col not in train_df_enc.columns:
                train_df_enc[col] = 0
        
        for terminal in terminals:
            col = f'TerminalID_{terminal}'
            if col not in train_df_enc.columns:
                train_df_enc[col] = 0
                
        for desig in desig_types:
            col = f'Desig_{desig}'
            if col not in train_df_enc.columns:
                train_df_enc[col] = 0
        
        # Drop the first category for each to match original behavior
        if 'MoveType_In' in train_df_enc.columns:
            train_df_enc = train_df_enc.drop(columns=['MoveType_In'])
        if 'TerminalID_T1' in train_df_enc.columns:
            train_df_enc = train_df_enc.drop(columns=['TerminalID_T1'])
        if 'Desig_EXP' in train_df_enc.columns:
            train_df_enc = train_df_enc.drop(columns=['Desig_EXP'])
        
        # Get train columns after processing
        train_non_numeric = ['datetime', 'MoveDate', 'outlier_flag', 'TokenCount']
        train_drop_cols = [col for col in train_non_numeric if col in train_df_enc.columns]
        train_cols = train_df_enc.drop(columns=train_drop_cols).columns.tolist()
        
        logger.info(f"[train] Reference columns: {train_cols}")
        
        # Find columns in validation/test not in train
        extra_cols = [col for col in df_cols if col not in train_cols]
        if extra_cols:
            logger.warning(f"[{split}] Extra columns found: {extra_cols}")
            # Drop extra columns
            df_enc = df_enc.drop(columns=extra_cols)
            
        # Find columns in train not in validation/test
        missing_cols = [col for col in train_cols if col not in df_cols]
        if missing_cols:
            logger.warning(f"[{split}] Missing columns found: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                df_enc[col] = 0
        
        # Re-compute X_to_scale after column adjustments
        X_to_scale = df_enc.drop(columns=drop_cols).values
        logger.info(f"[{split}] Adjusted feature count: {X_to_scale.shape[1]}")
    
    # Apply scaling
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
    
    # One-hot encode with consistent reference levels for training too
    df_train_enc = df_train.copy()
    
    # Define reference levels for all datasets - this will be used for train, validation and test
    # We explicitly specify all possible values to ensure consistent encoding
    move_types = ['In', 'Out']
    terminals = ['T1', 'T2', 'T3', 'T4', 'R1']  # Added R1 terminal that appears in validation/test
    desig_types = ['EXP', 'FCL', 'MT', 'T/S']
    
    # Store reference levels for consistent use across all splits
    reference_levels = {
        'move_types': move_types,
        'terminals': terminals,
        'desig_types': desig_types
    }
    
    logger.info(f"Reference levels for categorical encoding:")
    logger.info(f"  - MoveType: {move_types}")
    logger.info(f"  - TerminalID: {terminals}")
    logger.info(f"  - Desig: {desig_types}")
    
    # Create explicit dummies with consistent references
    df_train_enc = pd.get_dummies(df_train_enc, columns=['MoveType'], prefix=['MoveType'])
    df_train_enc = pd.get_dummies(df_train_enc, columns=['TerminalID'], prefix=['TerminalID'])
    df_train_enc = pd.get_dummies(df_train_enc, columns=['Desig'], prefix=['Desig'])
    
    # Ensure all expected columns exist
    for move_type in move_types:
        col = f'MoveType_{move_type}'
        if col not in df_train_enc.columns:
            df_train_enc[col] = 0
    
    for terminal in terminals:
        col = f'TerminalID_{terminal}'
        if col not in df_train_enc.columns:
            df_train_enc[col] = 0
            
    for desig in desig_types:
        col = f'Desig_{desig}'
        if col not in df_train_enc.columns:
            df_train_enc[col] = 0
    
    # Drop the first category for each to match original behavior
    if 'MoveType_In' in df_train_enc.columns:
        df_train_enc = df_train_enc.drop(columns=['MoveType_In'])
    if 'TerminalID_T1' in df_train_enc.columns:
        df_train_enc = df_train_enc.drop(columns=['TerminalID_T1'])
    if 'Desig_EXP' in df_train_enc.columns:
        df_train_enc = df_train_enc.drop(columns=['Desig_EXP'])

    # Convert boolean columns to int so they are included in scaling
    for col in ['is_weekend', 'is_friday', 'is_holiday']:
        if col in df_train_enc.columns:
            df_train_enc[col] = df_train_enc[col].astype(int)
            
    # Drop only non-numeric columns that cannot be encoded
    non_numeric = ['datetime', 'MoveDate', 'outlier_flag']
    X_train = df_train_enc.drop(columns=[col for col in non_numeric if col in df_train_enc.columns] + ['TokenCount']).values
    
    # Log feature count before scaling to help debug
    logger.info(f"[train] Feature count before scaling: {X_train.shape[1]}")
    
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

    # Process train, validation, and test splits
    # Pass reference levels to ensure consistency across all datasets
    process_encoding('train', scaler, encoder, autoencoder, logger, reference_levels)
    process_encoding('validation', scaler, encoder, autoencoder, logger, reference_levels)
    process_encoding('test', scaler, encoder, autoencoder, logger, reference_levels)

    logger.info("Feature encoding completed for train, validation, and test datasets.")


if __name__ == '__main__':
    main()
