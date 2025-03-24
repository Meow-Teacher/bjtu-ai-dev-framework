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