import numpy as np
from enum import Enum

class LossType(Enum):
    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"

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