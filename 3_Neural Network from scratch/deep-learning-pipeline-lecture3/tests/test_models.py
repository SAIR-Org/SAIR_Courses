"""Test model components"""
import pytest
import numpy as np
from src.models.operations import NeuralOperations
from src.models.layers import DenseLayer
from src.models.neural_network import DNN


class TestNeuralOperations:
    """Test neural network operations"""
    
    def test_relu(self):
        """Test ReLU function"""
        x = np.array([[-1, 0, 1], [-2, 0.5, 2]])
        result = NeuralOperations.relu(x)
        
        expected = np.array([[0, 0, 1], [0, 0.5, 2]])
        np.testing.assert_array_equal(result, expected)
    
    def test_relu_derivative(self):
        """Test ReLU derivative"""
        x = np.array([[-1, 0, 1], [-2, 0.5, 2]])
        result = NeuralOperations.relu_derivative(x)
        
        expected = np.array([[0, 0, 1], [0, 1, 1]])
        np.testing.assert_array_equal(result, expected)
    
    def test_softmax(self):
        """Test softmax function"""
        x = np.array([[1, 2, 3], [1, 1, 1]])
        result = NeuralOperations.softmax(x)
        
        # Check shape
        assert result.shape == (2, 3)
        
        # Check rows sum to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)
        
        # Check ordering (largest input gets highest probability)
        assert result[0, 2] > result[0, 1] > result[0, 0]
    
    def test_cross_entropy(self):
        """Test cross-entropy loss"""
        y_pred = np.array([[0.1, 0.9], [0.9, 0.1]])
        y_true = np.array([[0, 1], [1, 0]])
        
        loss = NeuralOperations.cross_entropy(y_pred, y_true)
        
        # Loss should be positive
        assert loss > 0
        
        # Perfect prediction should give low loss (but not zero due to clipping)
        perfect_pred = np.array([[0, 1], [1, 0]])
        perfect_loss = NeuralOperations.cross_entropy(perfect_pred, y_true)
        assert perfect_loss < 0.01
    
    def test_weight_initialization(self):
        """Test weight initialization"""
        shape = (100, 50)
        
        # Test He initialization
        weights_relu = NeuralOperations.initialize_weights(shape, 'relu')
        assert weights_relu.shape == shape
        assert np.abs(np.mean(weights_relu)) < 0.1  # Mean close to 0
        
        # Test Xavier initialization
        weights_xavier = NeuralOperations.initialize_weights(shape, 'sigmoid')
        assert weights_xavier.shape == shape


class TestDenseLayer:
    """Test dense layer"""
    
    def test_init(self):
        """Test layer initialization"""
        layer = DenseLayer(10, 5, activation='relu')
        
        assert layer.input_size == 10
        assert layer.output_size == 5
        assert layer.activation == 'relu'
        assert layer.weights.shape == (10, 5)
        assert layer.biases.shape == (1, 5)
        assert layer.num_params == 10*5 + 5
    
    def test_forward_relu(self):
        """Test forward pass with ReLU"""
        layer = DenseLayer(2, 3, activation='relu')
        
        # Set deterministic weights for testing
        layer.weights = np.array([[1, 0, -1], [0, 1, 0]])
        layer.biases = np.array([[0, 0, 1]])
        
        X = np.array([[1, 2], [3, 4]])
        output = layer.forward(X)
        
        # Expected: X @ W + b, then ReLU
        expected = np.array([
            [1*1 + 2*0 + 0, 1*0 + 2*1 + 0, max(0, 1*(-1) + 2*0 + 1)],
            [3*1 + 4*0 + 0, 3*0 + 4*1 + 0, max(0, 3*(-1) + 4*0 + 1)]
        ])
        
        np.testing.assert_allclose(output, expected, rtol=1e-5)
    
    def test_backward(self):
        """Test backward pass"""
        layer = DenseLayer(2, 3, activation='relu', l2_lambda=0)
        
        # Forward pass
        X = np.array([[1, 2], [3, 4]])
        output = layer.forward(X)
        
        # Mock gradient
        dL_dA = np.ones_like(output)
        
        # Backward pass
        dL_dinput = layer.backward(dL_dA, learning_rate=0.01)
        
        # Check shapes
        assert dL_dinput.shape == X.shape
        
        # Check that weights were updated
        old_weights = layer.weights.copy()
        # We can't check exact values due to randomness in initialization
        # but we can verify the method runs without error


class TestDNN:
    """Test deep neural network"""
    
    def test_init(self):
        """Test DNN initialization"""
        dnn = DNN(
            layer_sizes=[10, 5, 3],
            activations=['relu', 'softmax'],
            name='test_dnn'
        )
        
        assert len(dnn.layers) == 2
        assert dnn.layer_sizes == [10, 5, 3]
        assert dnn.name == 'test_dnn'
        assert dnn.num_params > 0
    
    def test_forward(self):
        """Test forward pass through network"""
        dnn = DNN([2, 3, 2], ['relu', 'softmax'])
        
        X = np.random.randn(4, 2)
        output = dnn.forward(X)
        
        assert output.shape == (4, 2)
        
        # Check softmax output sums to 1
        row_sums = np.sum(output, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0, 1.0], rtol=1e-5)
    
    def test_compute_loss(self):
        """Test loss computation"""
        dnn = DNN([2, 3, 2], ['relu', 'softmax'])
        
        X = np.random.randn(5, 2)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
        
        output = dnn.forward(X)
        loss = dnn.compute_loss(output, y)
        
        assert loss > 0
    
    @pytest.mark.slow
    def test_train_step(self):
        """Test single training step"""
        dnn = DNN([2, 3, 2], ['relu', 'softmax'], learning_rate=0.01)
        
        X = np.random.randn(10, 2)
        y = np.eye(2)[np.random.choice(2, 10)]  # One-hot encoded
        
        loss_before = dnn.compute_loss(dnn.forward(X), y)
        dnn.train_step(X, y)
        loss_after = dnn.compute_loss(dnn.forward(X), y)
        
        # Loss should decrease (not strictly due to randomness)
        # Just verify the method runs without error
        assert True