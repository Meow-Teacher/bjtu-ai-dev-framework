import numpy as np
from ..layers.base import Layer

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