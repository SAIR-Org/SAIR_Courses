"""Test data loading and preprocessing"""
import pytest
import numpy as np
from pathlib import Path
from src.data.loader import DataLoader
from src.data.preprocesser import DataPreprocessor


class TestDataLoader:
    """Test data loader"""
    
    def test_init(self):
        """Test initialization"""
        config = {
            'directories': {'data': './test_data'},
            'datasets': {
                'enabled': ['mnist'],
                'mnist': {'urls': {}, 'classes': []}
            }
        }
        
        loader = DataLoader(config)
        assert loader.data_dir == Path('./test_data') / 'mnist'
    
    def test_download_retry(self, mocker):
        """Test download retry logic"""
        # Mock urllib to simulate failures
        mock_urlopen = mocker.patch('src.data.loader.urllib.request.urlopen')
        mock_urlopen.side_effect = Exception("Network error")
        
        config = {'directories': {'data': './test_data'}}
        loader = DataLoader(config)
        
        result = loader.download_with_retry(
            ['http://example.com/data.gz'],
            Path('test.gz'),
            retries=2
        )
        
        assert not result
        assert mock_urlopen.call_count == 2


class TestDataPreprocessor:
    """Test data preprocessor"""
    
    def test_transform_grayscale(self):
        """Test RGB to grayscale conversion"""
        config = {'preprocessing': {'rgb_to_grayscale': True}}
        preprocessor = DataPreprocessor(config)
        
        # Create fake RGB image
        X = np.random.rand(10, 32, 32, 3).astype(np.float32)
        X_transformed, _ = preprocessor.transform(X, dataset_name='cifar10')
        
        assert X_transformed.shape == (10, 32, 32, 1)
        assert X_transformed.dtype == np.float32
    
    def test_transform_flatten(self):
        """Test flattening"""
        config = {'preprocessing': {'flatten': True}}
        preprocessor = DataPreprocessor(config)
        
        X = np.random.rand(10, 28, 28).astype(np.float32)
        X_transformed, _ = preprocessor.transform(X)
        
        assert X_transformed.shape == (10, 784)
    
    def test_one_hot_encode(self):
        """Test one-hot encoding"""
        config = {'preprocessing': {}}
        preprocessor = DataPreprocessor(config)
        
        X = np.random.rand(5, 10)
        y = np.array([0, 1, 2, 0, 1])
        
        X_transformed, y_transformed = preprocessor.transform(X, y)
        
        assert y_transformed.shape == (5, 3)
        assert np.all(y_transformed[0] == [1, 0, 0])
        assert np.all(y_transformed[1] == [0, 1, 0])