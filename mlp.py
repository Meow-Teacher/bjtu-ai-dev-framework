# mlp_neural_network.py
import numpy as np
import threading
import queue
import time
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
from enum import Enum
import pickle
import os

class ActivationType(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"
    LINEAR = "linear"

class InitializationType(Enum):
    ZERO = "zero"
    RANDOM = "random"
    XAVIER = "xavier"
    HE = "he"

class RegularizationType(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"

class OptimizerType(Enum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    RMSPROP = "rmsprop"
    ADAM = "adam"

class StopCriteriaType(Enum):
    MAX_EPOCHS = "max_epochs"
    MIN_LOSS = "min_loss"
    EARLY_STOPPING = "early_stopping"
    CONVERGENCE = "convergence"

class LossType(Enum):
    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"

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

class Loss:
    """
    Loss function class that provides various loss functions and their derivatives
    """
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.shape[0]
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred / y_true.shape[0]
    
    @staticmethod
    def get_loss_function(loss_type: LossType):
        if loss_type == LossType.MSE:
            return Loss.mse, Loss.mse_derivative
        elif loss_type == LossType.BINARY_CROSS_ENTROPY:
            return Loss.binary_cross_entropy, Loss.binary_cross_entropy_derivative
        elif loss_type == LossType.CROSS_ENTROPY:
            return Loss.cross_entropy, Loss.cross_entropy_derivative
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

class Initializer:
    """
    Weight initializer class that provides various initialization strategies
    """
    @staticmethod
    def zero(input_size, output_size):
        return np.zeros((input_size, output_size))
    
    @staticmethod
    def random(input_size, output_size):
        return np.random.randn(input_size, output_size) * 0.01
    
    @staticmethod
    def xavier(input_size, output_size):
        # Good for sigmoid/tanh activations
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))
    
    @staticmethod
    def he(input_size, output_size):
        # Good for ReLU activations
        return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    
    @staticmethod
    def get_initializer(init_type: InitializationType):
        if init_type == InitializationType.ZERO:
            return Initializer.zero
        elif init_type == InitializationType.RANDOM:
            return Initializer.random
        elif init_type == InitializationType.XAVIER:
            return Initializer.xavier
        elif init_type == InitializationType.HE:
            return Initializer.he
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}")

class Regularizer:
    """
    Regularization class that provides L1, L2 and elastic net regularization
    """
    @staticmethod
    def none(weights, lambda_param=0.0):
        return 0.0, np.zeros_like(weights)
    
    @staticmethod
    def l1(weights, lambda_param=0.01):
        l1_loss = lambda_param * np.sum(np.abs(weights))
        l1_grad = lambda_param * np.sign(weights)
        return l1_loss, l1_grad
    
    @staticmethod
    def l2(weights, lambda_param=0.01):
        l2_loss = 0.5 * lambda_param * np.sum(np.square(weights))
        l2_grad = lambda_param * weights
        return l2_loss, l2_grad
    
    @staticmethod
    def elastic_net(weights, lambda_param=0.01, l1_ratio=0.5):
        l1_lambda = lambda_param * l1_ratio
        l2_lambda = lambda_param * (1 - l1_ratio)
        
        l1_loss, l1_grad = Regularizer.l1(weights, l1_lambda)
        l2_loss, l2_grad = Regularizer.l2(weights, l2_lambda)
        
        return l1_loss + l2_loss, l1_grad + l2_grad
    
    @staticmethod
    def get_regularizer(reg_type: RegularizationType):
        if reg_type == RegularizationType.NONE:
            return Regularizer.none
        elif reg_type == RegularizationType.L1:
            return Regularizer.l1
        elif reg_type == RegularizationType.L2:
            return Regularizer.l2
        elif reg_type == RegularizationType.ELASTIC_NET:
            return Regularizer.elastic_net
        else:
            raise ValueError(f"Unsupported regularization type: {reg_type}")

class Optimizer:
    """Base optimizer class"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, weights, gradients):
        """Update rule to be implemented by subclasses"""
        raise NotImplementedError
    
    def reset(self):
        """Reset optimizer state"""
        pass

class SGDOptimizer(Optimizer):
    """Standard SGD optimizer"""
    def update(self, weights, gradients):
        return weights - self.learning_rate * gradients

class MomentumOptimizer(Optimizer):
    """SGD with momentum optimizer"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def update(self, weights, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return weights + self.velocity
    
    def reset(self):
        self.velocity = None

class RMSPropOptimizer(Optimizer):
    """RMSProp optimizer"""
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.sq_grad = None
    
    def update(self, weights, gradients):
        if self.sq_grad is None:
            self.sq_grad = np.zeros_like(weights)
        
        self.sq_grad = self.decay_rate * self.sq_grad + (1 - self.decay_rate) * np.square(gradients)
        return weights - self.learning_rate * gradients / (np.sqrt(self.sq_grad) + self.epsilon)
    
    def reset(self):
        self.sq_grad = None

class AdamOptimizer(Optimizer):
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Dictionary to store momentum terms for each parameter
        self.v = {}  # Dictionary to store velocity terms for each parameter
        self.t = 0
    
    def update(self, weights, gradients):
        # Create unique key for this parameter based on its shape
        param_key = f"{weights.shape}"
        
        # Initialize momentum terms for this parameter if not exists
        if param_key not in self.m:
            self.m[param_key] = np.zeros_like(weights)
            self.v[param_key] = np.zeros_like(weights)
        
        self.t += 1
        
        # Ensure gradients have same shape as weights
        if gradients.shape != weights.shape:
            raise ValueError(f"Gradient shape {gradients.shape} does not match weight shape {weights.shape}")
        
        # Update momentum terms
        self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * gradients
        self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * np.square(gradients)
        
        # Compute bias-corrected momentum terms
        m_hat = self.m[param_key] / (1 - np.power(self.beta1, self.t))
        v_hat = self.v[param_key] / (1 - np.power(self.beta2, self.t))
        
        # Update weights
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0

class OptimizerFactory:
    """Factory for creating optimizers"""
    @staticmethod
    def create_optimizer(optimizer_type: OptimizerType, **kwargs):
        if optimizer_type == OptimizerType.SGD:
            return SGDOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.MOMENTUM:
            return MomentumOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.RMSPROP:
            return RMSPropOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.ADAM:
            return AdamOptimizer(**kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

class Layer:
    """Base class for network layers"""
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """Forward pass computation"""
        raise NotImplementedError
    
    def backward(self, output_error, learning_rate):
        """Backward pass computation"""
        raise NotImplementedError
    
    def get_parameters(self):
        """Return layer parameters"""
        return {}
    
    def set_parameters(self, parameters):
        """Set layer parameters"""
        pass

class DenseLayer(Layer):
    """Fully connected dense layer"""
    def __init__(self, input_size, output_size, 
                 activation_type=ActivationType.SIGMOID,
                 init_type=InitializationType.XAVIER,
                 reg_type=RegularizationType.NONE,
                 lambda_param=0.01):
        super().__init__()
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self.output_size = output_size
        
        # Set activation function
        self.activation_type = activation_type
        self.activation, self.activation_derivative = Activation.get_activation_function(activation_type)
        
        # Set initializer
        initializer = Initializer.get_initializer(init_type)
        self.weights = initializer(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        
        # Set regularizer
        self.reg_type = reg_type
        self.regularizer = Regularizer.get_regularizer(reg_type)
        self.lambda_param = lambda_param
        
        # Store activation pre-output for backward pass
        self.z = None
    
    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, output_error, optimizer):
        # Gradient of loss with respect to activation
        if self.activation_type == ActivationType.SOFTMAX:
            # For softmax, we typically handle the derivative as part of the loss function
            activation_error = output_error
        else:
            activation_error = output_error * self.activation_derivative(self.z)
        
        # Compute gradients with respect to weights, bias
        weights_gradient = np.dot(self.input.T, activation_error)
        bias_gradient = np.sum(activation_error, axis=0, keepdims=True)
        
        # Add regularization gradient
        reg_loss, reg_grad = self.regularizer(self.weights, self.lambda_param)
        weights_gradient += reg_grad
        
        # Update weights and bias using the optimizer
        self.weights = optimizer.update(self.weights, weights_gradient)
        self.bias = optimizer.update(self.bias, bias_gradient)
        
        # Return error for next layer
        input_error = np.dot(activation_error, self.weights.T)
        
        return input_error
    
    def get_parameters(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation_type': self.activation_type,
            'reg_type': self.reg_type,
            'lambda_param': self.lambda_param
        }
    
    def set_parameters(self, parameters):
        self.weights = parameters['weights']
        self.bias = parameters['bias']
        self.input_size = parameters['input_size']
        self.output_size = parameters['output_size']
        self.activation_type = parameters['activation_type']
        self.activation, self.activation_derivative = Activation.get_activation_function(self.activation_type)
        self.reg_type = parameters['reg_type']
        self.regularizer = Regularizer.get_regularizer(self.reg_type)
        self.lambda_param = parameters['lambda_param']

class DropoutLayer(Layer):
    """Dropout layer for regularization"""
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training_mode = True
    
    def forward(self, input_data):
        self.input = input_data
        
        if self.training_mode:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape) / (1 - self.dropout_rate)
            self.output = input_data * self.mask
        else:
            self.output = input_data
            
        return self.output
    
    def backward(self, output_error, optimizer):
        if self.training_mode:
            return output_error * self.mask
        else:
            return output_error
    
    def set_training_mode(self, training_mode):
        self.training_mode = training_mode
    
    def get_parameters(self):
        return {
            'dropout_rate': self.dropout_rate
        }
    
    def set_parameters(self, parameters):
        self.dropout_rate = parameters['dropout_rate']

class BatchNormLayer(Layer):
    """Batch Normalization Layer"""
    def __init__(self, input_size, epsilon=1e-8, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.input_size = input_size
        
        # Parameters to be learned
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
        # Variables needed for backward pass
        self.x_norm = None
        self.x_centered = None
        self.std = None
        self.var = None
        self.batch_size = None
        
        self.training_mode = True
    
    def forward(self, input_data):
        self.input = input_data
        self.batch_size = input_data.shape[0]
        
        if self.training_mode:
            mean = np.mean(input_data, axis=0, keepdims=True)
            var = np.var(input_data, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalize
            self.x_centered = input_data - mean
            self.std = np.sqrt(var + self.epsilon)
            self.x_norm = self.x_centered / self.std
            
            # Scale and shift
            self.output = self.gamma * self.x_norm + self.beta
        else:
            # Use running statistics for inference
            x_norm = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * x_norm + self.beta
        
        return self.output
    
    def backward(self, output_error, optimizer):
        # Compute gradients for gamma and beta
        dgamma = np.sum(output_error * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(output_error, axis=0, keepdims=True)
        
        # Update gamma and beta
        self.gamma = optimizer.update(self.gamma, dgamma)
        self.beta = optimizer.update(self.beta, dbeta)
        
        # Compute gradient with respect to input
        dx_norm = output_error * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * (-0.5) * np.power(self.std, -3), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * (-1/self.std), axis=0, keepdims=True) + dvar * np.mean(-2 * self.x_centered, axis=0, keepdims=True)
        
        input_error = dx_norm / self.std + dvar * 2 * self.x_centered / self.batch_size + dmean / self.batch_size
        
        return input_error
    
    def set_training_mode(self, training_mode):
        self.training_mode = training_mode
    
    def get_parameters(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'input_size': self.input_size
        }
    
    def set_parameters(self, parameters):
        self.gamma = parameters['gamma']
        self.beta = parameters['beta']
        self.running_mean = parameters['running_mean']
        self.running_var = parameters['running_var']
        self.epsilon = parameters['epsilon']
        self.momentum = parameters['momentum']
        self.input_size = parameters['input_size']

class NetworkConfig:
    """Configuration class for neural network architecture and training parameters"""
    def __init__(self):
        # Network architecture
        self.layers = []
        
        # Training parameters
        self.loss_type = LossType.MSE
        self.optimizer_type = OptimizerType.SGD
        self.optimizer_params = {'learning_rate': 0.01}
        
        # Stop criteria
        self.stop_criteria = StopCriteriaType.MAX_EPOCHS
        self.max_epochs = 1000
        self.min_loss = 1e-4
        self.patience = 10  # For early stopping
        self.min_delta = 1e-4  # For convergence checking
        
        # Batch parameters
        self.batch_size = 32
        self.shuffle = True
        
        # Validation
        self.validation_split = 0.2
        
        # Parallel training
        self.use_parallel = False
        self.num_threads = 4
    
    def add_dense_layer(self, input_size, output_size, activation=ActivationType.SIGMOID, 
                         init_type=InitializationType.XAVIER, reg_type=RegularizationType.NONE, 
                         lambda_param=0.01):
        self.layers.append({
            'type': 'dense',
            'input_size': input_size,
            'output_size': output_size,
            'activation_type': activation,
            'init_type': init_type,
            'reg_type': reg_type,
            'lambda_param': lambda_param
        })
        return self
    
    def add_dropout_layer(self, dropout_rate=0.5):
        self.layers.append({
            'type': 'dropout',
            'dropout_rate': dropout_rate
        })
        return self
    
    def add_batch_norm_layer(self, input_size, epsilon=1e-8, momentum=0.9):
        self.layers.append({
            'type': 'batch_norm',
            'input_size': input_size,
            'epsilon': epsilon,
            'momentum': momentum
        })
        return self
    
    def set_loss(self, loss_type):
        self.loss_type = loss_type
        return self
    
    def set_optimizer(self, optimizer_type, **kwargs):
        self.optimizer_type = optimizer_type
        self.optimizer_params = kwargs
        return self
    
    def set_stop_criteria(self, criteria_type, **kwargs):
        self.stop_criteria = criteria_type
        
        if 'max_epochs' in kwargs:
            self.max_epochs = kwargs['max_epochs']
        if 'min_loss' in kwargs:
            self.min_loss = kwargs['min_loss']
        if 'patience' in kwargs:
            self.patience = kwargs['patience']
        if 'min_delta' in kwargs:
            self.min_delta = kwargs['min_delta']
            
        return self
    
    def set_batch_params(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        return self
    
    def set_validation_split(self, validation_split):
        self.validation_split = validation_split
        return self
    
    def enable_parallel(self, num_threads=4):
        self.use_parallel = True
        self.num_threads = num_threads
        return self
    
    def get_config(self):
        return {
            'layers': self.layers,
            'loss_type': self.loss_type,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'stop_criteria': self.stop_criteria,
            'max_epochs': self.max_epochs,
            'min_loss': self.min_loss,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'validation_split': self.validation_split,
            'use_parallel': self.use_parallel,
            'num_threads': self.num_threads
        }
    
    def load_config(self, config_dict):
        self.layers = config_dict['layers']
        self.loss_type = config_dict['loss_type']
        self.optimizer_type = config_dict['optimizer_type']
        self.optimizer_params = config_dict['optimizer_params']
        self.stop_criteria = config_dict['stop_criteria']
        self.max_epochs = config_dict['max_epochs']
        self.min_loss = config_dict['min_loss']
        self.patience = config_dict['patience']
        self.min_delta = config_dict['min_delta']
        self.batch_size = config_dict['batch_size']
        self.shuffle = config_dict['shuffle']
        self.validation_split = config_dict['validation_split']
        self.use_parallel = config_dict['use_parallel']
        self.num_threads = config_dict['num_threads']
        return self

class NeuralNetwork:
    """Main Neural Network class"""
    def __init__(self, config=None):
        self.layers = []
        self.loss_function = None
        self.loss_derivative = None
        self.loss_history = {'train': [], 'val': []}
        
        if config:
            self._build_from_config(config)
    
    def _build_from_config(self, config):
        # Create layers based on configuration
        for layer_config in config.layers:
            if layer_config['type'] == 'dense':
                self.add_layer(DenseLayer(
                    layer_config['input_size'],
                    layer_config['output_size'],
                    layer_config['activation_type'],
                    layer_config['init_type'],
                    layer_config['reg_type'],
                    layer_config['lambda_param']
                ))
            elif layer_config['type'] == 'dropout':
                self.add_layer(DropoutLayer(layer_config['dropout_rate']))
            elif layer_config['type'] == 'batch_norm':
                self.add_layer(BatchNormLayer(
                    layer_config['input_size'],
                    layer_config['epsilon'],
                    layer_config['momentum']
                ))
        
        # Set loss function
        self.loss_function, self.loss_derivative = Loss.get_loss_function(config.loss_type)
    
    def add_layer(self, layer):
        self.layers.append(layer)
        return self
    
    def set_loss(self, loss_type):
        self.loss_function, self.loss_derivative = Loss.get_loss_function(loss_type)
        return self
    
    def _forward_pass(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def _backward_pass(self, y_true, y_pred, optimizer):
        # Compute initial gradient from loss function
        error = self.loss_derivative(y_true, y_pred)
        
        # Propagate error through the network backward
        for layer in reversed(self.layers):
            error = layer.backward(error, optimizer)
    
    def fit(self, X, y, config, verbose=True):
        """Train the neural network"""
        # Setup optimizer
        optimizer = OptimizerFactory.create_optimizer(config.optimizer_type, **config.optimizer_params)
        
        # Setup loss function if not already set
        if self.loss_function is None:
            self.loss_function, self.loss_derivative = Loss.get_loss_function(config.loss_type)
        
        # Split data into training and validation sets if needed
        if config.validation_split > 0:
            val_size = int(X.shape[0] * config.validation_split)
            train_size = X.shape[0] - val_size
            
            # Shuffle data before splitting
            indices = np.random.permutation(X.shape[0])
            X_train = X[indices[:train_size]]
            y_train = y[indices[:train_size]]
            X_val = X[indices[train_size:]]
            y_val = y[indices[train_size:]]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Initialize variables for early stopping and convergence checking
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Set all layers to training mode
        self._set_training_mode(True)
        
        # Training loop
        for epoch in range(config.max_epochs):
            # Shuffle training data for each epoch if enabled
            if config.shuffle:
                indices = np.random.permutation(X_train.shape[0])
                X_train = X_train[indices]
                y_train = y_train[indices]
            
            # Process mini-batches
            if config.use_parallel and config.num_threads > 1:
                self._train_parallel(X_train, y_train, optimizer, config.batch_size, config.num_threads)
            else:
                self._train_batch(X_train, y_train, optimizer, config.batch_size)
            
            # Calculate training loss
            train_pred = self.predict(X_train)
            train_loss = self.loss_function(y_train, train_pred)
            self.loss_history['train'].append(train_loss)
            
            # Calculate validation loss if validation data is available
            if X_val is not None and y_val is not None:
                # Set to evaluation mode for validation
                self._set_training_mode(False)
                val_pred = self.predict(X_val)
                val_loss = self.loss_function(y_val, val_pred)
                self.loss_history['val'].append(val_loss)
                # Set back to training mode
                self._set_training_mode(True)
            else:
                val_loss = None
            
            # Print progress if verbose
            if verbose and (epoch % 10 == 0 or epoch == config.max_epochs - 1):
                if val_loss is not None:
                    print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}: train_loss={train_loss:.6f}")
            
            # Check stopping criteria
            if config.stop_criteria == StopCriteriaType.MIN_LOSS and train_loss <= config.min_loss:
                if verbose:
                    print(f"Stopping: Reached target loss {config.min_loss}")
                break
            
            if config.stop_criteria == StopCriteriaType.EARLY_STOPPING and val_loss is not None:
                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            
            if config.stop_criteria == StopCriteriaType.CONVERGENCE and epoch > 0:
                if abs(self.loss_history['train'][-2] - train_loss) < config.min_delta:
                    if verbose:
                        print(f"Stopping: Convergence achieved at epoch {epoch}")
                    break
        
        # Set to evaluation mode after training
        self._set_training_mode(False)
        
        return self.loss_history
    
    def _train_batch(self, X, y, optimizer, batch_size):
        """Train on mini-batches sequentially"""
        # Process each mini-batch
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            # Forward pass
            y_pred = self._forward_pass(X_batch)
            
            # Backward pass
            self._backward_pass(y_batch, y_pred, optimizer)
    
    def _train_parallel(self, X, y, optimizer, batch_size, num_threads):
        """Train on mini-batches in parallel"""
        # Split data into chunks for each thread
        chunk_size = X.shape[0] // num_threads
        if chunk_size < batch_size:
            # Fall back to sequential if chunks are too small
            return self._train_batch(X, y, optimizer, batch_size)
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            # Submit tasks for each chunk
            for i in range(0, X.shape[0], chunk_size):
                end_idx = min(i + chunk_size, X.shape[0])
                futures.append(
                    executor.submit(
                        self._process_chunk,
                        X[i:end_idx],
                        y[i:end_idx],
                        optimizer,
                        batch_size
                    )
                )
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    def _process_chunk(self, X_chunk, y_chunk, optimizer, batch_size):
        """Process a chunk of data in a worker thread"""
        # Create a copy of network parameters for this thread
        thread_params = self._get_parameters()
        thread_network = NeuralNetwork()
        for layer_params in thread_params:
            if layer_params['type'] == 'dense':
                layer = DenseLayer(
                    layer_params['input_size'],
                    layer_params['output_size'],
                    layer_params['activation_type'],
                    InitializationType.ZERO,  # Doesn't matter as we'll set weights
                    layer_params['reg_type'],
                    layer_params['lambda_param']
                )
                layer.weights = layer_params['weights'].copy()
                layer.bias = layer_params['bias'].copy()
                thread_network.add_layer(layer)
            elif layer_params['type'] == 'dropout':
                layer = DropoutLayer(layer_params['dropout_rate'])
                thread_network.add_layer(layer)
            elif layer_params['type'] == 'batch_norm':
                layer = BatchNormLayer(
                    layer_params['input_size'],
                    layer_params['epsilon'],
                    layer_params['momentum']
                )
                layer.gamma = layer_params['gamma'].copy()
                layer.beta = layer_params['beta'].copy()
                layer.running_mean = layer_params['running_mean'].copy()
                layer.running_var = layer_params['running_var'].copy()
                thread_network.add_layer(layer)
        
        # Set the same loss function
        thread_network.loss_function = self.loss_function
        thread_network.loss_derivative = self.loss_derivative
        
        # Train on this chunk
        thread_network._train_batch(X_chunk, y_chunk, optimizer, batch_size)
        
        # Return gradients to be applied to main network
        return thread_network._get_parameters()
    
    def _set_training_mode(self, training_mode):
        """Set all layers to training or evaluation mode"""
        for layer in self.layers:
            if hasattr(layer, 'set_training_mode'):
                layer.set_training_mode(training_mode)
    
    def _get_parameters(self):
        """Get all network parameters"""
        params = []
        for layer in self.layers:
            layer_params = layer.get_parameters()
            # Add layer type to parameters
            if isinstance(layer, DenseLayer):
                layer_params['type'] = 'dense'
            elif isinstance(layer, DropoutLayer):
                layer_params['type'] = 'dropout'
            elif isinstance(layer, BatchNormLayer):
                layer_params['type'] = 'batch_norm'
            params.append(layer_params)
        return params
    
    def _set_parameters(self, params):
        """Set all network parameters"""
        if len(params) != len(self.layers):
            raise ValueError("Parameters do not match network architecture")
        
        for i, layer_params in enumerate(params):
            self.layers[i].set_parameters(layer_params)
    
    def predict(self, X):
        """Make predictions with the network"""
        return self._forward_pass(X)
    
    def evaluate(self, X, y):
        """Evaluate the network on test data"""
        # Set to evaluation mode
        self._set_training_mode(False)
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate loss
        loss = self.loss_function(y, y_pred)
        
        return loss, y_pred
    
    def save(self, filepath):
        """Save the network to a file"""
        # Collect all network parameters
        params = {
            'layers': self._get_parameters(),
            'loss_history': self.loss_history
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, filepath):
        """Load the network from a file"""
        # Load from file
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        # Set network parameters
        self.layers = []
        for layer_params in params['layers']:
            if layer_params['type'] == 'dense':
                layer = DenseLayer(
                    layer_params['input_size'],
                    layer_params['output_size'],
                    layer_params['activation_type'],
                    InitializationType.ZERO,  # Doesn't matter as we'll set weights
                    layer_params['reg_type'],
                    layer_params['lambda_param']
                )
                layer.weights = layer_params['weights']
                layer.bias = layer_params['bias']
                self.add_layer(layer)
            elif layer_params['type'] == 'dropout':
                layer = DropoutLayer(layer_params['dropout_rate'])
                self.add_layer(layer)
            elif layer_params['type'] == 'batch_norm':
                layer = BatchNormLayer(
                    layer_params['input_size'],
                    layer_params['epsilon'],
                    layer_params['momentum']
                )
                layer.gamma = layer_params['gamma']
                layer.beta = layer_params['beta']
                layer.running_mean = layer_params['running_mean']
                layer.running_var = layer_params['running_var']
                self.add_layer(layer)
        
        self.loss_history = params['loss_history']
        
        # Set to evaluation mode
        self._set_training_mode(False)

class ConfusionMatrix:
    """Confusion matrix and classification metrics"""
    def __init__(self, y_true, y_pred, threshold=0.5):
        """
        Initialize a confusion matrix
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels/probabilities
            threshold: Threshold for converting probabilities to binary predictions (for binary classification)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.threshold = threshold
        self.num_classes = None
        self.matrix = None
        self.metrics = {}
        
        self._compute_matrix()
        self._compute_metrics()
    
    def _compute_matrix(self):
        """Compute the confusion matrix"""
        # Handle different input formats
        if len(self.y_true.shape) > 1 and self.y_true.shape[1] > 1:
            # Multi-class one-hot encoded
            self.num_classes = self.y_true.shape[1]
            y_true_class = np.argmax(self.y_true, axis=1)
            y_pred_class = np.argmax(self.y_pred, axis=1)
        elif len(self.y_pred.shape) > 1 and self.y_pred.shape[1] > 1:
            # Predicted probabilities for multi-class
            self.num_classes = self.y_pred.shape[1]
            y_true_class = self.y_true.astype(int)
            y_pred_class = np.argmax(self.y_pred, axis=1)
        else:
            # Binary classification
            self.num_classes = 2
            y_true_class = self.y_true.flatten().astype(int)
            y_pred_class = (self.y_pred.flatten() >= self.threshold).astype(int)
        
        # Create confusion matrix
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(len(y_true_class)):
            self.matrix[y_true_class[i], y_pred_class[i]] += 1
    
    def _compute_metrics(self):
        """Compute classification metrics from the confusion matrix"""
        # Overall accuracy
        self.metrics['accuracy'] = np.trace(self.matrix) / np.sum(self.matrix)
        
        # Per-class metrics
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(self.num_classes):
            # True positives for this class
            tp = self.matrix[i, i]
            
            # False positives for this class (sum of column i excluding tp)
            fp = np.sum(self.matrix[:, i]) - tp
            
            # False negatives for this class (sum of row i excluding tp)
            fn = np.sum(self.matrix[i, :]) - tp
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)
            
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            
            # F1 score = 2 * (Precision * Recall) / (Precision + Recall)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        self.metrics['precision'] = np.array(precisions)
        self.metrics['recall'] = np.array(recalls)
        self.metrics['f1_score'] = np.array(f1_scores)
        
        # Macro-averaged metrics
        self.metrics['macro_precision'] = np.mean(precisions)
        self.metrics['macro_recall'] = np.mean(recalls)
        self.metrics['macro_f1'] = np.mean(f1_scores)
    
    def display(self):
        """Display the confusion matrix and metrics"""
        print("Confusion Matrix:")
        print(self.matrix)
        print("\nMetrics:")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Macro Precision: {self.metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {self.metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {self.metrics['macro_f1']:.4f}")
        
        print("\nPer-class Metrics:")
        for i in range(self.num_classes):
            print(f"Class {i} - Precision: {self.metrics['precision'][i]:.4f}, Recall: {self.metrics['recall'][i]:.4f}, F1: {self.metrics['f1_score'][i]:.4f}")
    
    def plot(self, class_names=None):
        """Plot the confusion matrix as a heatmap"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = self.matrix.max() / 2
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, str(self.matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if self.matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


class MLPClassifier:
    """Convenience class for classification tasks using MLP"""
    def __init__(self, hidden_layer_sizes=(100,), activation=ActivationType.RELU, 
                 init_type=InitializationType.HE, solver=OptimizerType.ADAM,
                 alpha=0.0001, batch_size=32, learning_rate=0.001,
                 max_iter=200, validation_split=0.1, early_stopping=True,
                 n_iter_no_change=10, reg_type=RegularizationType.L2,
                 random_state=None):
        """
        A multi-layer perceptron classifier.
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(100,)
            The ith element represents the number of neurons in the ith hidden layer.
        activation : ActivationType, default=ActivationType.RELU
            Activation function for the hidden layers.
        init_type : InitializationType, default=InitializationType.HE
            Weight initialization strategy.
        solver : OptimizerType, default=OptimizerType.ADAM
            The solver for weight optimization.
        alpha : float, default=0.0001
            L2 regularization parameter.
        batch_size : int, default=32
            Size of minibatches for stochastic optimizers.
        learning_rate : float, default=0.001
            Learning rate for weight updates.
        max_iter : int, default=200
            Maximum number of iterations.
        validation_split : float, default=0.1
            Proportion of training data to set aside as validation set.
        early_stopping : bool, default=True
            Whether to use early stopping to terminate training when validation score is not improving.
        n_iter_no_change : int, default=10
            Maximum number of epochs to not meet improvement for early stopping.
        reg_type : RegularizationType, default=RegularizationType.L2
            Type of regularization.
        random_state : int, default=None
            Seed for random number generator.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.init_type = init_type
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.reg_type = reg_type
        
        self.network = None
        self.config = None
        self.classes_ = None
        self._is_fitted = False
        self.loss_history_ = None
    
    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        
        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
            
        Returns:
        --------
        self : returns a fitted MLPClassifier instance.
        """
        # Convert inputs to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        # Get number of features and classes
        n_samples, n_features = X.shape
        
        # Handle different target formats
        if len(y.shape) == 1:
            # Get unique classes
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            
            # Convert to one-hot encoding if more than 2 classes
            if n_classes > 2:
                y_encoded = np.zeros((n_samples, n_classes))
                for i, cls in enumerate(self.classes_):
                    y_encoded[:, i] = (y == cls).astype(int)
                y = y_encoded
            else:
                # Binary classification
                y = y.reshape(-1, 1)
        else:
            # Already in one-hot format
            n_classes = y.shape[1]
            self.classes_ = np.arange(n_classes)
        
        # Build network configuration
        self.config = NetworkConfig()
        
        # Add input layer to first hidden layer
        self.config.add_dense_layer(
            n_features, self.hidden_layer_sizes[0],
            self.activation, self.init_type, self.reg_type, self.alpha
        )
        
        # Add hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1):
            self.config.add_dense_layer(
                self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1],
                self.activation, self.init_type, self.reg_type, self.alpha
            )
        
        # Add output layer
        output_activation = ActivationType.SIGMOID if n_classes == 1 else ActivationType.SOFTMAX
        self.config.add_dense_layer(
            self.hidden_layer_sizes[-1], n_classes if n_classes > 1 else 1,
            output_activation, self.init_type, RegularizationType.NONE, 0.0
        )
        
        # Set optimizer
        self.config.set_optimizer(self.solver, learning_rate=self.learning_rate)
        
        # Set loss function
        if n_classes == 1:
            self.config.set_loss(LossType.BINARY_CROSS_ENTROPY)
        else:
            self.config.set_loss(LossType.CROSS_ENTROPY)
        
        # Configure early stopping if enabled
        if self.early_stopping:
            self.config.set_stop_criteria(
                StopCriteriaType.EARLY_STOPPING,
                max_epochs=self.max_iter,
                patience=self.n_iter_no_change,
                min_delta=1e-4
            )
        else:
            self.config.set_stop_criteria(
                StopCriteriaType.MAX_EPOCHS,
                max_epochs=self.max_iter
            )
        
        # Set validation split
        self.config.set_validation_split(self.validation_split)
        
        # Set batch parameters
        self.config.set_batch_params(self.batch_size, shuffle=True)
        
        # Create and train the network
        self.network = NeuralNetwork(self.config)
        self.loss_history_ = self.network.fit(X, y, self.config)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before predicting")
        
        # Convert input to numpy array
        X = np.array(X, dtype=np.float64)
        
        # Get predictions
        y_proba = self.predict_proba(X)
        
        # Convert to class labels
        if y_proba.shape[1] == 1:
            # Binary classification
            y_pred = (y_proba >= 0.5).astype(int).flatten()
        else:
            # Multi-class
            y_pred = np.argmax(y_proba, axis=1)
        
        # Map indices back to original class labels if needed
        return self.classes_[y_pred]
    
    def predict_proba(self, X):
        """
        Probability estimates for samples in X.
        
        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
            
        Returns:
        --------
        y_proba : ndarray of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before predicting probabilities")
        
        # Convert input to numpy array
        X = np.array(X, dtype=np.float64)
        
        # Get predictions
        y_proba = self.network.predict(X)
        
        # For binary classification with a single output
        if y_proba.shape[1] == 1:
            # Return probabilities for both classes
            return np.hstack([1 - y_proba, y_proba])
        
        return y_proba
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
            
        Returns:
        --------
        score : float
            Accuracy of the classifier.
        """
        y_pred = self.predict(X)
        
        # Convert y to 1D if needed
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
            y = self.classes_[y]
        
        # Calculate accuracy
        return np.mean(y_pred == y)


class MLPRegressor:
    """Convenience class for regression tasks using MLP"""
    def __init__(self, hidden_layer_sizes=(100,), activation=ActivationType.RELU, 
                 init_type=InitializationType.HE, solver=OptimizerType.ADAM,
                 alpha=0.0001, batch_size=32, learning_rate=0.001,
                 max_iter=200, validation_split=0.1, early_stopping=True,
                 n_iter_no_change=10, reg_type=RegularizationType.L2,
                 random_state=None):
        """
        A multi-layer perceptron regressor.
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(100,)
            The ith element represents the number of neurons in the ith hidden layer.
        activation : ActivationType, default=ActivationType.RELU
            Activation function for the hidden layers.
        init_type : InitializationType, default=InitializationType.HE
            Weight initialization strategy.
        solver : OptimizerType, default=OptimizerType.ADAM
            The solver for weight optimization.
        alpha : float, default=0.0001
            L2 regularization parameter.
        batch_size : int, default=32
            Size of minibatches for stochastic optimizers.
        learning_rate : float, default=0.001
            Learning rate for weight updates.
        max_iter : int, default=200
            Maximum number of iterations.
        validation_split : float, default=0.1
            Proportion of training data to set aside as validation set.
        early_stopping : bool, default=True
            Whether to use early stopping to terminate training when validation score is not improving.
        n_iter_no_change : int, default=10
            Maximum number of epochs to not meet improvement for early stopping.
        reg_type : RegularizationType, default=RegularizationType.L2
            Type of regularization.
        random_state : int, default=None
            Seed for random number generator.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.init_type = init_type
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.reg_type = reg_type
        
        self.network = None
        self.config = None
        self._is_fitted = False
        self.loss_history_ = None
        self.n_outputs_ = None
    
    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        
        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
            
        Returns:
        --------
        self : returns a fitted MLPRegressor instance.
        """
        # Input validation for X
        try:
            X = np.array(X, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert X to numpy array: {e}. "
                            f"Ensure X contains only numeric values and has consistent dimensions.")
        
        # Input validation for y
        try:
            # Handle different input types for y
            if isinstance(y, (list, tuple)):
                # Check if all elements are numeric or can be converted to numeric
                if all(isinstance(item, (int, float, bool, np.number)) or 
                      (isinstance(item, (list, tuple)) and 
                       all(isinstance(subitem, (int, float, bool, np.number)) for subitem in item))
                      for item in y):
                    y = np.array(y, dtype=np.float64)
                else:
                    raise ValueError("Target values must be numeric")
            elif isinstance(y, np.ndarray):
                y = y.astype(np.float64)
            else:
                raise ValueError(f"Unsupported type for y: {type(y)}. Expected list, tuple, or numpy array.")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert y to numpy array: {e}. "
                            f"Ensure y contains only numeric values and has consistent dimensions.")
        
        # Handle target shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Check that X and y have compatible shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes. "
                            f"X has {X.shape[0]} samples, but y has {y.shape[0]} samples.")
        
        # Get number of features and outputs
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        self.n_outputs_ = n_outputs
        
        # Build network configuration
        self.config = NetworkConfig()
        
        # Add input layer to first hidden layer
        self.config.add_dense_layer(
            n_features, self.hidden_layer_sizes[0],
            self.activation, self.init_type, self.reg_type, self.alpha
        )
        
        # Add hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1):
            self.config.add_dense_layer(
                self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1],
                self.activation, self.init_type, self.reg_type, self.alpha
            )
        
        # Add output layer with linear activation for regression
        self.config.add_dense_layer(
            self.hidden_layer_sizes[-1], n_outputs,
            ActivationType.LINEAR, self.init_type, RegularizationType.NONE, 0.0
        )
        
        # Set optimizer
        self.config.set_optimizer(self.solver, learning_rate=self.learning_rate)
        
        # Set loss function (MSE for regression)
        self.config.set_loss(LossType.MSE)
        
        # Configure early stopping if enabled
        if self.early_stopping:
            self.config.set_stop_criteria(
                StopCriteriaType.EARLY_STOPPING,
                max_epochs=self.max_iter,
                patience=self.n_iter_no_change,
                min_delta=1e-4
            )
        else:
            self.config.set_stop_criteria(
                StopCriteriaType.MAX_EPOCHS,
                max_epochs=self.max_iter
            )
        
        # Set validation split
        self.config.set_validation_split(self.validation_split)
        
        # Set batch parameters
        self.config.set_batch_params(self.batch_size, shuffle=True)
        
        # Create and train the network
        self.network = NeuralNetwork(self.config)
        self.loss_history_ = self.network.fit(X, y, self.config)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict values for samples in X.
        
        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before predicting")
        
        # Input validation for X
        try:
            X = np.array(X, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert X to numpy array: {e}. "
                            f"Ensure X contains only numeric values and has consistent dimensions.")
        
        # Check input shape
        if len(X.shape) == 1:
            # Single sample, reshape to 2D
            X = X.reshape(1, -1)
        
        # Verify that X has the correct number of features
        if X.shape[1] != self.network.layers[0].input_size:
            raise ValueError(f"X has {X.shape[1]} features, but the model was trained with "
                           f"{self.network.layers[0].input_size} features.")
        
        # Get predictions
        y_pred = self.network.predict(X)
        
        # Return predictions (reshape if needed)
        if self.n_outputs_ == 1:
            return y_pred.flatten()
        return y_pred
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
            
        Returns:
        --------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before scoring")
        
        # Input validation for X and y
        try:
            X = np.array(X, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert X to numpy array: {e}. "
                            f"Ensure X contains only numeric values and has consistent dimensions.")
        
        try:
            y = np.array(y, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert y to numpy array: {e}. "
                            f"Ensure y contains only numeric values and has consistent dimensions.")
        
        # Handle target shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Check that X and y have compatible shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes. "
                            f"X has {X.shape[0]} samples, but y has {y.shape[0]} samples.")
        
        # Get predictions
        y_pred = self.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Check that predictions and targets have compatible shapes
        if y.shape[1] != y_pred.shape[1]:
            raise ValueError(f"Predictions and targets have incompatible shapes. "
                            f"Predictions have {y_pred.shape[1]} outputs, but targets have {y.shape[1]} outputs.")
        
        # Calculate R^2 score
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean(axis=0)) ** 2).sum()
        
        # Handle edge case to avoid division by zero
        if v == 0:
            return 0.0
        
        return 1 - u / v
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : string
            Path to the file.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before saving")
        
        # Save network
        self.network.save(filepath)
    
    def load(self, filepath):
        """
        Load the model from a file.
        
        Parameters:
        -----------
        filepath : string
            Path to the file.
        """
        # Create network if not exists
        if self.network is None:
            self.network = NeuralNetwork()
        
        # Load network
        self.network.load(filepath)
        self._is_fitted = True
        
        # Determine number of outputs from the network
        output_layer = self.network.layers[-1]
        if isinstance(output_layer, DenseLayer):
            self.n_outputs_ = output_layer.output_size
        
        return self


if __name__ == '__main__':
    # Load and prepare Boston Housing dataset
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Read the data
    data = pd.read_csv('boston.csv')
    
    # Split features and target
    X = data.drop('MEDV', axis=1).values  # All columns except price
    y = data['MEDV'].values.reshape(-1, 1)  # Price is the target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation=ActivationType.RELU,
        solver=OptimizerType.ADAM,
        learning_rate=0.001,
        max_iter=200,
        batch_size=32,
        validation_split=0.2
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")