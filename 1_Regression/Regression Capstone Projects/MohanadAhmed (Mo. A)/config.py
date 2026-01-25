import warnings                             # Suppress warnings for cleaner outputs
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                       # Advanced visualization (correlation matrix)
import joblib                               # Save/load large models and preprocessing objects
import json                                 # Handle JSON configs and outputs
from datetime import datetime               # Timestamping for logs
import os                                   # File system operations
import time                                 # Time tracking for experiments
from IPython.display import display

# 🧰 Scikitlearn libraries - expanded for advanced ML workflows
from sklearn.model_selection import (
       train_test_split,                   # Split data into train/test sets
       cross_val_score,                    # Cross-validation scoring
       GridSearchCV,                       # Hyperparameter tuning (grid search)
       RandomizedSearchCV                 # Hyperparameter tuning (randomized search)
)
from sklearn.preprocessing import (
       StandardScaler,                     # Feature scaling (zero-mean, unit variance)
       RobustScaler,                       # Scaling robust to outliers
)

# Build modular pipelines
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
     f_regression,                        # Scoring function for regression (measures that relationship using a statistical F-test between each feature and the target)
     SelectKBest,                          # Univariate feature selection (selects the top K input features that have the strongest individual relationship with the target variable)
     RFE                                   # Recursive feature elimination (repeatedly trains a model and removes the least important features step by step until only the best ones remain)
)

from sklearn.linear_model import (
      LinearRegression,
      Lasso,                             #L1-regularized
      Ridge,                             #L2-regularized
      ElasticNet                        # Combination of L1 and L2 regularization
)
from sklearn.ensemble import (
    RandomForestRegressor,              # Ensemble of decision trees
    GradientBoostingRegressor,          # Boosted trees for regression
    VotingRegressor                     # Combine multiple regressors
)

from sklearn.svm import SVR
from sklearn.metrics import (
     mean_squared_error,
     r2_score,
     mean_absolute_error
)
from sklearn.inspection import (
    permutation_importance,       # Feature importance (measures how much model performance drops when the values of one feature are randomly shuffled)
    PartialDependenceDisplay      # Partial dependence plots (shows how the predicted output changes when one (or two) features vary, while others are averaged out)
)

# 🧪 Advanced model tracking with MLflow
import mlflow                  # Experiment tracking
import mlflow.sklearn          # Log sklearn models
from mlflow.models.signature import infer_signature  # Auto-capture input/output schema (data shape and type) for reproducible deployment




#=====================================================
#🎛️ Configuration & Reproducibility
#=====================================================
class Config:
    # Reproducibility - Critical for production!
    RANDOM_STATE = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    CV_FOLDS = 5
    N_JOBS = -1

    # Model directories - Organized project structure
    MODEL_DIR = "models"
    EXPERIMENT_DIR = "../src/experiments"

    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

config = Config()

# Initialize MLflow for experiment tracking
mlflow.set_tracking_uri(f"file:///{os.path.abspath(config.EXPERIMENT_DIR)}")
experiment_name = 'power_plant'
mlflow.set_experiment(experiment_name)
