"""
Data loading utilities for Spaceship Titanic dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from config import Config

logger = logging.getLogger(__name__)

def load_train_test_data():
    """
    Load training and test datasets.
    
    Returns:
        tuple: (train_df, test_df) pandas DataFrames
    """
    try:
        train_df = pd.read_csv(Config.RAW_DATA_DIR / 'train.csv')
        test_df = pd.read_csv(Config.RAW_DATA_DIR / 'test.csv')
        logger.info("✅ Data loaded successfully from local files")
    except FileNotFoundError:
        logger.warning("📥 Local files not found, attempting to download from Kaggle...")
        # You can implement Kaggle download logic here if needed
        raise FileNotFoundError("Please ensure train.csv and test.csv are in data/raw/ directory")
    
    logger.info(f"📊 Training data shape: {train_df.shape}")
    logger.info(f"📈 Test data shape: {test_df.shape}")
    
    return train_df, test_df

def prepare_train_val_test_split(train_df):
    """
    Prepare train/validation/test splits.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Separate features and target
    X = train_df.drop('Transported', axis=1)
    y = train_df['Transported'].astype(int)  # Convert boolean to int
    
    # Split the data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    logger.info(f"📊 DATA SPLITS:")
    logger.info(f"• Training: {X_train.shape[0]:,} samples")
    logger.info(f"• Validation: {X_val.shape[0]:,} samples")
    logger.info(f"• Test: {X_test.shape[0]:,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test