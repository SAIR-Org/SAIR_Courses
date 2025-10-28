"""
Configuration settings for Spaceship Titanic ML pipeline.
"""
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Reproducibility
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODEL_DIR: Path = BASE_DIR / "models"
    PRODUCTION_MODEL_DIR: Path = MODEL_DIR / "production"
    EXPERIMENT_DIR: Path = BASE_DIR / "spaceship_experiments"
    SUBMISSION_DIR: Path = BASE_DIR / "submissions"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.PRODUCTION_MODEL_DIR,
            cls.EXPERIMENT_DIR,
            cls.SUBMISSION_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")