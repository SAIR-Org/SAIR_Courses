"""
Feature engineering for Spaceship Titanic dataset.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SpaceshipFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering for Spaceship Titanic dataset"""
    
    def __init__(self):
        self.feature_names = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_eng = X.copy()
        
        # Extract information from PassengerId
        X_eng['GroupId'] = X_eng['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
        X_eng['GroupSize'] = X_eng.groupby('GroupId')['GroupId'].transform('count')
        
        # Extract deck/num/side from Cabin
        X_eng[['CabinDeck', 'CabinNum', 'CabinSide']] = X_eng['Cabin'].str.split('/', expand=True)
        X_eng['CabinNum'] = pd.to_numeric(X_eng['CabinNum'], errors='coerce')
        
        # Create total spending feature
        spending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        X_eng['TotalSpending'] = X_eng[spending_features].sum(axis=1)
        X_eng['HasSpending'] = (X_eng['TotalSpending'] > 0).astype(int)
        
        # Age groups
        X_eng['AgeGroup'] = pd.cut(X_eng['Age'], 
                                  bins=[0, 12, 18, 30, 50, 100], 
                                  labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        
        # Family features
        X_eng['IsAlone'] = (X_eng['GroupSize'] == 1).astype(int)
        
        # Drop original columns
        columns_to_drop = ['PassengerId', 'Cabin', 'Name']
        X_eng = X_eng.drop([col for col in columns_to_drop if col in X_eng.columns], axis=1)
        
        self.feature_names = list(X_eng.columns)
        return X_eng
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self):
        return self.feature_names