import numpy as np
from ..layers.base import Layer

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