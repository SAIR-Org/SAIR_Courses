# utils.py - Enhanced Version
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering for housing price prediction.
    Creates interaction features and distance metrics.
    """
    
    # California's approximate center
    CA_CENTER_LAT = 36.5
    CA_CENTER_LON = -119.5
    
    def fit(self, X, y=None):
        """No fitting required for this transformer"""
        return self

    def transform(self, X):
        """
        Create engineered features:
        1. Distance from CA center
        2. Rooms to bedrooms ratio
        3. Income to rooms ratio
        4. Population to occupancy ratio
        5. Income × Rooms interaction
        6. Geographic quadrant encoding
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X.copy()
        
        # Distance from California center
        lat, lon = X_array[:, 6], X_array[:, 7]
        distance_from_center = np.sqrt(
            (lat - self.CA_CENTER_LAT)**2 + 
            (lon - self.CA_CENTER_LON)**2
        )
        
        # Rooms to bedrooms ratio (avoid division by zero)
        rooms_to_bedrooms = X_array[:, 2] / (X_array[:, 3] + 1e-8)
        
        # Income to rooms ratio
        income_to_rooms = X_array[:, 0] / (X_array[:, 2] + 1e-8)
        
        # Population to occupancy ratio
        population_to_occupancy = X_array[:, 4] / (X_array[:, 5] + 1e-8)
        
        # Income × Rooms interaction
        income_rooms_interaction = X_array[:, 0] * X_array[:, 2]
        
        # Geographic quadrant (North-South vs East-West)
        quadrant = (lat > self.CA_CENTER_LAT).astype(int) * 2 + \
                   (lon > self.CA_CENTER_LON).astype(int)
        
        # Stack all features
        X_engineered = np.column_stack([
            X_array,
            distance_from_center,
            rooms_to_bedrooms,
            income_to_rooms,
            population_to_occupancy,
            income_rooms_interaction,
            quadrant
        ])
        
        return X_engineered


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detects and handles outliers using IQR (Interquartile Range) method.
    Clips extreme values to bounds defined by quartiles.
    """
    
    def __init__(self, factor=1.5):
        """
        Parameters:
        -----------
        factor : float, default=1.5
            IQR multiplier for outlier bounds (1.5 is standard)
        """
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """
        Calculate outlier bounds for each feature using IQR method.
        
        Bounds are defined as:
        - Lower: Q1 - factor * IQR
        - Upper: Q3 + factor * IQR
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        n_features = X_array.shape[1]
        
        self.lower_bounds_ = []
        self.upper_bounds_ = []
        
        for i in range(n_features):
            Q1 = np.percentile(X_array[:, i], 25)
            Q3 = np.percentile(X_array[:, i], 75)
            IQR = Q3 - Q1
            
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            
            self.lower_bounds_.append(lower)
            self.upper_bounds_.append(upper)
        
        return self

    def transform(self, X):
        """Clip values to outlier bounds for each feature"""
        X_array = X.values if isinstance(X, pd.DataFrame) else X.copy()
        
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("Transformer must be fitted before transform()")
        
        for i in range(X_array.shape[1]):
            X_array[:, i] = np.clip(
                X_array[:, i],
                self.lower_bounds_[i],
                self.upper_bounds_[i]
            )
        
        return X_array

    def get_outlier_stats(self, X):
        """
        Get statistics about outliers detected.
        
        Returns:
        --------
        dict : Statistics about outliers per feature
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        stats = {}
        
        for i in range(X_array.shape[1]):
            lower_outliers = np.sum(X_array[:, i] < self.lower_bounds_[i])
            upper_outliers = np.sum(X_array[:, i] > self.upper_bounds_[i])
            stats[f'Feature_{i}'] = {
                'lower_outliers': lower_outliers,
                'upper_outliers': upper_outliers,
                'total_outliers': lower_outliers + upper_outliers
            }
        
        return stats