# 🚀 Spaceship Titanic ML Pipeline

A production-ready, modular machine learning project for the Kaggle Spaceship Titanic competition. This project demonstrates industry best practices for building, training, and deploying classification models.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [MLflow Tracking](#mlflow-tracking)
- [For Students](#for-students)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements a complete machine learning pipeline for predicting which passengers were transported to an alternate dimension aboard the Spaceship Titanic.

**Key Features:**
- ✅ Modular, reusable code architecture
- ✅ Advanced feature engineering
- ✅ Multiple model comparisons (7+ algorithms)
- ✅ Hyperparameter tuning with cross-validation
- ✅ MLflow experiment tracking
- ✅ Production-ready preprocessing pipelines
- ✅ Comprehensive documentation

---

## 📁 Project Structure

```
spaceship-titanic-ml/
├── project/                      # Main Python package
│   ├── __init__.py
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── config.py             # Project settings and constants
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── load_data.py          # Data loading utilities
│   │   ├── feature_engineering.py # SpaceshipFeatureEngineer
│   │   └── preprocessing.py      # Preprocessing pipelines
│   ├── models/                   # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── base_models.py        # Model definitions
│   │   ├── train_model.py        # Training orchestration
│   │   ├── evaluate_model.py     # Evaluation metrics
│   │   └── hyperparameter_tuning.py # Tuning logic
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── mlflow_utils.py       # MLflow helpers
│       ├── visualization.py      # Plotting functions
│       └── metrics.py            # Custom metrics
├── data/                         # Data directory
│   ├── raw/                      # Original datasets
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/                # Processed datasets
├── models/                       # Saved model artifacts
│   └── production/               # Production-ready models
├── notebooks/                    # Jupyter notebooks
│   ├── original_notebook.ipynb
│   └── exploratory_analysis.ipynb
├── tests/                        # Unit tests
├── run_pipeline.py               # Main pipeline runner
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── README.md                     # This file
└── .gitignore
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Set Up Project Structure
```bash
# Create directory structure
mkdir -p spaceship-titanic-ml/{project/{config,data,models,utils},data/{raw,processed},models/production,notebooks,tests}
cd spaceship-titanic-ml

# Place the CSV files in data/raw/
# Download from Kaggle: https://www.kaggle.com/competitions/spaceship-titanic/data
# Copy train.csv and test.csv to data/raw/
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Project as Package
```bash
pip install -e .
```

This allows you to import modules from anywhere: `from project.data import load_data`

---

## 🚀 Quick Start

### Run Complete Pipeline
Execute the entire ML pipeline from start to finish:

```bash
python run_pipeline.py --mode full
```

### Run Specific Steps
```bash
# Data processing only
python run_pipeline.py --mode preprocessing

# Model training only
python run_pipeline.py --mode training

# Model evaluation
python run_pipeline.py --mode evaluation

# Generate submission (optional)
python run_pipeline.py --mode submission
```

### Expected Output
```
🚀 SPACESHIP TITANIC ML PIPELINE
================================================================================
Mode: full
Timestamp: 2024-01-15 10:30:00
================================================================================
Setting up environment...
✅ Environment setup complete
================================================================================
STEP 1: DATA PREPROCESSING
================================================================================
Loading training and test data...
✅ Data loaded successfully from local files
📊 Training data shape: (8693, 14)
📈 Test data shape: (4277, 13)
...
🎉 PIPELINE COMPLETE!
```

---

## 📖 Detailed Usage

### Pipeline Modes

| Mode | Description | Output |
|------|-------------|---------|
| `full` | Complete pipeline from data loading to submission | Trained models, submissions, MLflow experiments |
| `preprocessing` | Data loading, cleaning, and feature engineering | Processed datasets, preprocessing pipeline |
| `training` | Model training and validation | Trained models, performance metrics |
| `evaluation` | Model evaluation on test set | Evaluation reports, confusion matrices |
| `submission` | Generate Kaggle submission file | submission_YYYYMMDD_HHMMSS.csv |

### 1️⃣ Data Preparation

#### Load Data
```python
from project.data.load_data import load_train_test_data

train_df, test_df = load_train_test_data()
print(f"Training data: {train_df.shape}, Test data: {test_df.shape}")
```

#### Feature Engineering
```python
from project.data.feature_engineering import SpaceshipFeatureEngineer

feature_engineer = SpaceshipFeatureEngineer()
X_engineered = feature_engineer.fit_transform(train_df.drop('Transported', axis=1))
print(f"Engineered features: {X_engineered.shape[1]}")
```

#### Preprocessing Pipeline
```python
from project.data.preprocessing import create_preprocessing_pipeline

pipeline = create_preprocessing_pipeline(X_engineered)
X_processed = pipeline.fit_transform(X_engineered)
```

### 2️⃣ Model Training

#### Train All Models
```python
from project.models.train_model import train_all_models

results, trained_models = train_all_models(
    X_train, y_train,
    X_val, y_val,
    track_mlflow=True
)

# Select best model
best_name = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
print(f"Best model: {best_name}")
```

### 3️⃣ Model Evaluation

```python
from project.models.evaluate_model import comprehensive_evaluation

metrics = comprehensive_evaluation(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    model_name=best_name
)
```

### 4️⃣ Generate Kaggle Submission

```python
from project.models.train_model import generate_submission

submission_df = generate_submission(
    model=best_model,
    pipeline=pipeline,
    test_df=test_df,
    output_path='submissions/final_submission.csv'
)
```

---

## 🏗️ Pipeline Architecture

### Data Flow
```
Raw Data → Feature Engineering → Preprocessing → Model Training → Evaluation → Submission
```

### Component Overview

1. **Data Loading** (`project/data/load_data.py`)
   - Load CSV files from `data/raw/`
   - Split into train/validation/test sets
   - Handle missing data detection

2. **Feature Engineering** (`project/data/feature_engineering.py`)
   - Extract group information from PassengerId
   - Decompose Cabin into deck/number/side
   - Create spending and family features
   - Add age groups and alone indicators

3. **Preprocessing** (`project/data/preprocessing.py`)
   - Numerical: Median imputation + Robust scaling
   - Categorical: Mode imputation + One-hot encoding
   - Full pipeline with feature engineering

4. **Model Training** (`project/models/train_model.py`)
   - 7+ classification algorithms
   - Cross-validation
   - MLflow experiment tracking
   - Ensemble methods

5. **Evaluation** (`project/models/evaluate_model.py`)
   - Accuracy, Precision, Recall, F1, ROC-AUC
   - Confusion matrices
   - Classification reports

---

## 📚 Module Documentation

### `project.config.config`
**Purpose**: Centralized configuration management

**Key Components**:
- `Config` class with all project settings
- Path configurations for data, models, experiments
- Reproducibility settings (random state, CV folds)

### `project.data.load_data`
**Purpose**: Data loading and initial splitting

**Key Functions**:
- `load_train_test_data()`: Load datasets from raw directory
- `prepare_train_val_test_split()`: Create train/val/test splits

### `project.data.feature_engineering`
**Purpose**: Advanced feature creation

**Key Classes**:
- `SpaceshipFeatureEngineer`: Custom transformer for domain-specific features
  - Group features from PassengerId
  - Cabin decomposition
  - Spending patterns
  - Age groups and family indicators

### `project.data.preprocessing`
**Purpose**: Data preprocessing pipelines

**Key Functions**:
- `create_preprocessing_pipeline()`: Build complete preprocessing pipeline
- `preprocess_data()`: Apply pipeline to datasets

### `project.models.base_models`
**Purpose**: Model definitions and configurations

**Key Functions**:
- `get_advanced_models()`: Returns portfolio of classification models
- Includes: Logistic Regression, Random Forest, XGBoost, LightGBM, Ensembles

### `project.models.train_model`
**Purpose**: Model training orchestration

**Key Functions**:
- `train_all_models()`: Train complete model portfolio
- `evaluate_classification_model()`: Comprehensive model evaluation
- `select_best_model()`: Choose best performing model
- `generate_submission()`: Create Kaggle submission file

### `project.models.evaluate_model`
**Purpose**: Model evaluation and visualization

**Key Functions**:
- `comprehensive_evaluation()`: Full evaluation suite
- `plot_confusion_matrix()`: Visualization of model performance

### `project.utils.mlflow_utils`
**Purpose**: MLflow experiment tracking

**Key Functions**:
- `setup_mlflow()`: Configure MLflow tracking

---

## ⚙️ Configuration

The project is configured through `project/config/config.py`:

```python
class Config:
    # Reproducibility
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1  # Use all available cores
    
    # Paths (automatically created)
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
```

### Customizing Configuration
You can modify these settings directly in the config file or override them in your scripts:

```python
from project.config.config import Config

# Temporarily override settings
Config.RANDOM_STATE = 123
Config.CV_FOLDS = 10
```

---

## 📊 MLflow Tracking

All experiments are automatically tracked with MLflow for reproducibility and comparison.

### View Experiments
```bash
mlflow ui --backend-store-uri spaceship_experiments
```
Then open http://localhost:5000 in your browser.

### What's Tracked
- **Parameters**: Model hyperparameters and configuration
- **Metrics**: Training/validation accuracy, precision, recall, F1, ROC-AUC
- **Artifacts**: Saved models, preprocessing pipelines, confusion matrices
- **Metadata**: Model versions, training time, git commits

### Example MLflow Usage
```python
import mlflow

with mlflow.start_run(run_name="experiment_1"):
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 200)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.83)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
```

---

## 🎓 For Students: Learning Objectives

This project demonstrates key machine learning engineering concepts:

### 1. Modular Code Architecture
- **Separation of Concerns**: Data, models, and utilities in separate modules
- **Reusability**: Components can be used independently
- **Maintainability**: Easy to update individual components

### 2. Production ML Practices
- **Reproducibility**: Fixed random seeds, version control
- **Experiment Tracking**: MLflow for model management
- **Pipeline Automation**: End-to-end workflow orchestration

### 3. Model Development Lifecycle
- **Exploratory Data Analysis**: Understanding data patterns
- **Feature Engineering**: Creating meaningful predictors
- **Model Selection**: Comparing multiple algorithms
- **Hyperparameter Tuning**: Optimizing model performance
- **Evaluation**: Comprehensive performance assessment

### 4. Adapting for Your Projects

#### Replace Dataset
```python
# In project/data/load_data.py
def load_your_data():
    return pd.read_csv('your_data.csv'), pd.read_csv('your_test.csv')
```

#### Custom Feature Engineering
```python
# In project/data/feature_engineering.py
class YourFeatureEngineer:
    def transform(self, X):
        # Your domain-specific features
        X_eng = X.copy()
        X_eng['new_feature'] = X_eng['feature1'] / X_eng['feature2']
        return X_eng
```

#### Add New Models
```python
# In project/models/base_models.py
from your_model_library import YourModel

models['Your Model'] = YourModel(
    param1=value1,
    param2=value2
)
```

### 5. Key Learning Outcomes
After studying this project, you should be able to:

✅ Build modular ML pipelines  
✅ Implement professional feature engineering  
✅ Compare multiple models systematically  
✅ Track experiments with MLflow  
✅ Create reproducible research  
✅ Prepare competition submissions  
✅ Structure projects for collaboration  

---

## 🐛 Troubleshooting

### Common Issues and Solutions

**Issue**: `ModuleNotFoundError: No module named 'project'`  
**Solution**: Run `pip install -e .` from the project root directory

**Issue**: `FileNotFoundError` for train.csv or test.csv  
**Solution**: Ensure CSV files are in `data/raw/` directory

**Issue**: MLflow UI not showing experiments  
**Solution**: Check that `spaceship_experiments` directory exists and contains files

**Issue**: Memory errors during training  
**Solution**: Reduce `N_JOBS` in config or use smaller model subsets

**Issue**: Submission format errors  
**Solution**: Verify PassengerId and Transported columns are present and correctly formatted

### Debug Mode
Run pipeline with detailed logging:
```bash
python run_pipeline.py --mode full 2>&1 | tee pipeline.log
```

### Performance Tips
- Use `--mode preprocessing` first to cache processed data
- For large datasets, consider using `N_JOBS=1` to reduce memory usage
- Processed data is saved as .npy files for faster reloading

---

## 📝 License

MIT License - Feel free to use this project for learning, teaching, and commercial purposes.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Create a new issue with detailed description

---

## 🎉 Acknowledgments

- Kaggle for hosting the Spaceship Titanic competition
- Scikit-learn, XGBoost, and LightGBM development teams
- MLflow community for excellent experiment tracking
- All contributors and users of this template

---

## 🔄 Version History

- **v1.0.0** (2024-01-15)
  - Initial release with complete ML pipeline
  - Feature engineering and preprocessing
  - Multi-model training and evaluation
  - MLflow integration
  - Kaggle submission generation

---

**Happy Learning and Building! 🚀**

*Remember: The best way to learn is by doing. Don't hesitate to experiment with the code and adapt it to your own projects!*