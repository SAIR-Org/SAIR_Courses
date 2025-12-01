"""
Preprocessing pipeline for Spaceship Titanic dataset.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from .feature_engineering import SpaceshipFeatureEngineer

logger = logging.getLogger(__name__)

def create_preprocessing_pipeline(X_sample):
    """
    Create preprocessing pipeline based on data sample.
    
    Args:
        X_sample: Sample DataFrame to determine feature types
        
    Returns:
        Pipeline: Full preprocessing pipeline
    """
    # Apply feature engineering to determine feature types
    feature_engineer = SpaceshipFeatureEngineer()
    X_engineered = feature_engineer.fit_transform(X_sample)
    
    # Identify feature types
    numerical_features = X_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"🔢 Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"🔤 Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Create full pipeline with feature engineering
    full_pipeline = Pipeline([
        ('feature_engineer', SpaceshipFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def preprocess_data(X_train, X_val, X_test, pipeline):
    """
    Apply preprocessing pipeline to data.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        pipeline: Preprocessing pipeline
        
    Returns:
        tuple: Processed X_train, X_val, X_test and fitted pipeline
    """
    logger.info("🔄 Applying preprocessing pipeline...")
    
    # Fit and transform training data
    X_train_proc = pipeline.fit_transform(X_train)
    
    # Transform validation and test data
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)
    
    logger.info(f"✅ Preprocessing complete!")
    logger.info(f"📊 Processed data shapes:")
    logger.info(f"   • X_train: {X_train_proc.shape}")
    logger.info(f"   • X_val: {X_val_proc.shape}")
    logger.info(f"   • X_test: {X_test_proc.shape}")
    
    return X_train_proc, X_val_proc, X_test_proc, pipeline