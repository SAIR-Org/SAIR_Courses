# utils.py snippet
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_eng = X.values if isinstance(X, pd.DataFrame) else X.copy()
        X_eng = np.column_stack([
            X_eng,
            np.sqrt((X_eng[:,6]-36.5)**2 + (X_eng[:,7]+119.5)**2),  # distance feature
            X_eng[:,2]/(X_eng[:,3]+1e-8),  # rooms/bedrooms
            X_eng[:,0]/(X_eng[:,2]+1e-8),  # income/rooms
            X_eng[:,4]/(X_eng[:,5]+1e-8),  # population/occupancy
            X_eng[:,0]*X_eng[:,2],          # income*rooms
            (X_eng[:,6]>36.5).astype(int)*2 + (X_eng[:,7]>-119.5).astype(int)  # quadrant
        ])
        return X_eng


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5): self.factor = factor
    def fit(self, X, y=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.lower_bounds_ = [np.percentile(X_arr[:,i],25) - self.factor*(np.percentile(X_arr[:,i],75)-np.percentile(X_arr[:,i],25)) for i in range(X_arr.shape[1])]
        self.upper_bounds_ = [np.percentile(X_arr[:,i],75) + self.factor*(np.percentile(X_arr[:,i],75)-np.percentile(X_arr[:,i],25)) for i in range(X_arr.shape[1])]
        return self
    def transform(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X.copy()
        for i in range(X_arr.shape[1]):
            X_arr[:,i] = np.clip(X_arr[:,i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_arr
