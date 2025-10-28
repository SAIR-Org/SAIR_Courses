"""
Data processing module for Spaceship Titanic ML Pipeline.

Handles data loading, feature engineering, and preprocessing.
"""

from .load_data import load_train_test_data, prepare_train_val_test_split
from .feature_engineering import SpaceshipFeatureEngineer
from .preprocessing import create_preprocessing_pipeline, preprocess_data

__all__ = [
    'load_train_test_data',
    'prepare_train_val_test_split', 
    'SpaceshipFeatureEngineer',
    'create_preprocessing_pipeline',
    'preprocess_data'
]