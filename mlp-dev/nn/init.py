import numpy as np
from enum import Enum

class InitializationType(Enum):
    ZERO = "zero"
    RANDOM = "random"
    XAVIER = "xavier"
    HE = "he"

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