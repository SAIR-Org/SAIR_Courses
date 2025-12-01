from setuptools import setup, find_packages

setup(
    name="spaceship-titanic-ml",
    version="1.0.0",
    description="Machine Learning Pipeline for Spaceship Titanic Kaggle Competition",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "mlflow>=2.0.0",
        "joblib>=1.1.0",
        "plotly>=5.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
    ],
)