import numpy as np
from enum import Enum

class RegularizationType(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"

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