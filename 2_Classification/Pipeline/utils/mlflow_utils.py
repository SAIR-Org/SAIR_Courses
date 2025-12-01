"""
MLflow utility functions.
"""
import mlflow
from config import Config

def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(f"file://{Config.EXPERIMENT_DIR.absolute()}")
    experiment_name = "spaceship_titanic_classification"
    mlflow.set_experiment(experiment_name)