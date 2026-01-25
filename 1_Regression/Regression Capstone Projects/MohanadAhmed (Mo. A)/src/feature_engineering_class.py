from sklearn.base import BaseEstimator, TransformerMixin
from config import *
from eda import *

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering using domain knowledge"""

    def __init__(self):
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # DOMAIN-DRIVEN FEATURE ENGINEERING
        # step 1 : Calculate Saturation Vapor Pressure (P_sat) using Magnus Formula, result in milibar (100Pa)
        P_sat = 6.112 * np.exp((17.67 * X['AT']) / (243.5 + X['AT']))
        # step 2 : Calculate Actual Vapor Pressure (P_v)
        P_v = P_sat * (X['RH'] / 100.0)
        # Physics based features:
        X['AirDensity'] = X['AP'] / (X['AT'] + 273.15)  # K = ℃ + 273.15
        X['MoistureMoleFraction'] = P_v / X['AP']
        X['DryMassFlowIndex'] = (X['AP'] - P_v) / (X['AT'] + 273.15)  # scaled by temperature (in K) to find the actual density of the working fluid

        self.feature_names = list(X.columns)
        return X

    def fit_transform(self, X, y=None):
        """Mimics sklearn's fit_transform"""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self):
        return self.feature_names

# Test our feature engineering
engineer = AdvancedFeatureEngineer()
X_engineered = engineer.fit_transform(X)

print(f"\n🎯 FEATURE ENGINEERING COMPLETE!")
print(f"• Original features: {X.shape[1]} (from previous work)")
print(f"• Engineered features: {X_engineered.shape[1]} (NEW!)")
print(f"• New features created: {engineer.feature_names[-3:]}")