import numpy as np
from enum import Enum

class OptimizerType(Enum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    RMSPROP = "rmsprop"
    ADAM = "adam"

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
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients)
        
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))
        
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        self.m = None
        self.v = None
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