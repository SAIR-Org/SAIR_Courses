"""
Spaceship Titanic ML Pipeline

A production-ready machine learning pipeline for the Kaggle Spaceship Titanic competition.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import key classes for easy access
from Pipeline.config.config import Config
from Pipeline.data.feature_engineering import SpaceshipFeatureEngineer
from Pipeline.models.base_model import get_advanced_models

# Define what gets imported with "from Pipeline import *"
__all__ = [
    'Config',
    'SpaceshipFeatureEngineer', 
    'get_advanced_models'
]