import numpy as np
from enum import Enum

class ActivationType(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"
    LINEAR = "linear"

class Activation:
    """
    Activation function class that provides various activation functions and their derivatives
    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.power(np.tanh(x), 2)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        # Shift x for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x):
        # For softmax, derivative is typically handled directly in the loss function
        return np.ones_like(x)
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def get_activation_function(activation_type: ActivationType):
        if activation_type == ActivationType.SIGMOID:
            return Activation.sigmoid, Activation.sigmoid_derivative
        elif activation_type == ActivationType.TANH:
            return Activation.tanh, Activation.tanh_derivative
        elif activation_type == ActivationType.RELU:
            return Activation.relu, Activation.relu_derivative
        elif activation_type == ActivationType.LEAKY_RELU:
            return Activation.leaky_relu, Activation.leaky_relu_derivative
        elif activation_type == ActivationType.SOFTMAX:
            return Activation.softmax, Activation.softmax_derivative
        elif activation_type == ActivationType.LINEAR:
            return Activation.linear, Activation.linear_derivative
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}") 