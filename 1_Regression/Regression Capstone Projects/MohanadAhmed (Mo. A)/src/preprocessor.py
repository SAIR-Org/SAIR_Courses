
from eda import *
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Clip extreme values using IQR method (robust). Works on DataFrames."""

    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            self.feature_names_ = X.columns
        else:
            X_values = X
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]

        self.lower_bounds_ = []
        self.upper_bounds_ = []

        for i in range(X_values.shape[1]):
            Q1 = np.percentile(X_values[:, i], 25)
            Q3 = np.percentile(X_values[:, i], 75)
            IQR = Q3 - Q1
            self.lower_bounds_.append(Q1 - self.factor * IQR)
            self.upper_bounds_.append(Q3 + self.factor * IQR)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values.copy()
        else:
            X_values = X.copy()

        for i in range(X_values.shape[1]):
            X_values[:, i] = np.clip(X_values[:, i],
                                     self.lower_bounds_[i],
                                     self.upper_bounds_[i])
        return pd.DataFrame(X_values, columns=self.feature_names_)  # Return as DataFrame to keep column names












preprocessor = Pipeline([
    ('outlier_handler', OutlierHandler(factor=1.5)),  # Handle outliers
    ('scaler', RobustScaler())  # Robust to outliers (better than StandardScaler)
])

# Apply preprocessing pipeline
print("🔄 Applying preprocessing pipeline...")
X_processed = preprocessor.fit_transform(X, Y)

print("✅ ADVANCED PREPROCESSING PIPELINE BUILT!")
print(f"📊 Processed data shape: {X_processed.shape}")
