import numpy as np
from ..layers.base import Layer
from ..functional.activations import Activation, ActivationType
from ..nn.init import Initializer, InitializationType
from ..nn.regularization import Regularizer, RegularizationType

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