# =============================================================================
# File: src/data/preprocessor.py
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocess data for DNN training."""
    
    def __init__(self, config):
        self.config = config
        self.val_split = config['training']['val_split']
        self.random_seed = config['training']['random_seed']
    
    def prepare_dataset(self, X_train, y_train, X_test, y_test, dataset_name):
        """Preprocess and split dataset."""
        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32) 
        
        # Convert RGB to grayscale for CIFAR-10
        if dataset_name == 'cifar10' and X_train.ndim == 4 and X_train.shape[-1] == 3:
            X_train = np.mean(X_train, axis=-1)
            X_test = np.mean(X_test, axis=-1) 
        
        # Normalize to [0, 1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Flatten for DNN
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # One-hot encode labels
        n_classes = 10
        y_train_onehot = np.zeros((len(y_train), n_classes))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        y_test_onehot = np.zeros((len(y_test), n_classes))
        y_test_onehot[np.arange(len(y_test)), y_test] = 1
        
        # Create validation split
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train_onehot,
            test_size=self.val_split,
            random_state=self.random_seed,
            stratify=y_train
        )
        
        return {
            'X_train': X_train_final,
            'y_train': y_train_final,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test_onehot
        }

