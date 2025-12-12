# =============================================================================
# File: src/models/dnn.py
# =============================================================================

import numpy as np
import pickle

class NeuralMath:
    """Neural network math operations."""
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def cross_entropy(y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def cross_entropy_gradient(y_pred, y_true):
        return y_pred - y_true
    
    @staticmethod
    def initialize_weights(shape, activation='relu'):
        if activation == 'relu':
            std = np.sqrt(2.0 / shape[0])
        else:
            std = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.randn(*shape) * std 
    

class DenseLayer:
    """Single dense layer."""
    
    def __init__(self, input_size, output_size, activation='relu', l2_lambda=0.0001):
        self.weights = NeuralMath.initialize_weights((input_size, output_size), activation)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.l2_lambda = l2_lambda
        self.input_cache = None
        self.output_cache = None
    
    def forward(self, X):
        self.input_cache = X
        Z = X @ self.weights + self.biases
        
        if self.activation == 'relu':
            A = NeuralMath.relu(Z)
        elif self.activation == 'softmax':
            A = NeuralMath.softmax(Z)
        else:
            A = Z
        
        self.output_cache = A
        return A
    
    def backward(self, dL_dA, learning_rate):
        batch_size = self.input_cache.shape[0]
        
        if self.activation == 'relu':
            dL_dZ = dL_dA * NeuralMath.relu_derivative(self.output_cache)
        else:
            dL_dZ = dL_dA
        
        dL_dW = (self.input_cache.T @ dL_dZ) / batch_size
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True) / batch_size
        
        if self.l2_lambda > 0:
            dL_dW += self.l2_lambda * self.weights / batch_size
        
        self.weights -= learning_rate * dL_dW
        self.biases -= learning_rate * dL_db
        
        return dL_dZ @ self.weights.T
    
    @property
    def num_params(self):
        return self.weights.size + self.biases.size


class DNN:
    """Deep Neural Network."""
    
    def __init__(self, layer_sizes, activations, learning_rate=0.001, 
                 l2_lambda=0.0001, name="DNN"):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.name = name
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                layer_sizes[i], layer_sizes[i + 1],
                activations[i], l2_lambda
            )
            self.layers.append(layer)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    @property
    def num_params(self):
        return sum(layer.num_params for layer in self.layers)
    
    def forward(self, X):
        activations = X
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations
    
    def compute_loss(self, y_pred, y_true):
        loss = NeuralMath.cross_entropy(y_pred, y_true)
        if self.l2_lambda > 0:
            reg_loss = sum(np.sum(layer.weights ** 2) for layer in self.layers)
            loss += (self.l2_lambda / (2 * y_true.shape[0])) * reg_loss
        return loss
    
    def compute_accuracy(self, y_pred, y_true):
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(pred_labels == true_labels)
    
    def train_step(self, X_batch, y_batch):
        y_pred = self.forward(X_batch)
        loss_grad = NeuralMath.cross_entropy_gradient(y_pred, y_batch)
        
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)
        
        loss = self.compute_loss(y_pred, y_batch)
        accuracy = self.compute_accuracy(y_pred, y_batch)
        
        return loss, accuracy
    
    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y)
        accuracy = self.compute_accuracy(y_pred, y)
        return loss, accuracy
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1), y_pred
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)