import numpy as np
import pickle
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from ..nn.loss import Loss, LossType
from ..optim.optimizer import OptimizerFactory, OptimizerType

class StopCriteriaType(Enum):
    MAX_EPOCHS = "max_epochs"
    MIN_LOSS = "min_loss"
    EARLY_STOPPING = "early_stopping"
    CONVERGENCE = "convergence"

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
            self.add_layer(layer_config)
        
        # Set loss function
        self.loss_function, self.loss_derivative = Loss.get_loss_function(config.loss_type)
    
    def add_layer(self, layer):
        self.layers.append(layer)
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
            if self._check_stopping_criteria(config, train_loss, val_loss, epoch):
                break
        
        # Set to evaluation mode after training
        self._set_training_mode(False)
        
        return self.loss_history
    
    def _check_stopping_criteria(self, config, train_loss, val_loss, epoch):
        """Check if training should stop based on stopping criteria"""
        if config.stop_criteria == StopCriteriaType.MIN_LOSS and train_loss <= config.min_loss:
            return True
        
        if config.stop_criteria == StopCriteriaType.EARLY_STOPPING and val_loss is not None:
            if len(self.loss_history['val']) > config.patience:
                recent_losses = self.loss_history['val'][-config.patience:]
                if all(recent_losses[i] - recent_losses[i-1] > -config.min_delta for i in range(1, len(recent_losses))):
                    return True
        
        if config.stop_criteria == StopCriteriaType.CONVERGENCE and epoch > 0:
            if abs(self.loss_history['train'][-2] - train_loss) < config.min_delta:
                return True
        
        return False
    
    def _train_batch(self, X, y, optimizer, batch_size):
        """Train on mini-batches sequentially"""
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            # Forward pass
            y_pred = self._forward_pass(X_batch)
            
            # Backward pass
            self._backward_pass(y_batch, y_pred, optimizer)
    
    def _train_parallel(self, X, y, optimizer, batch_size, num_threads):
        """Train on mini-batches in parallel"""
        chunk_size = X.shape[0] // num_threads
        if chunk_size < batch_size:
            return self._train_batch(X, y, optimizer, batch_size)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
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
            
            for future in futures:
                future.result()
    
    def _process_chunk(self, X_chunk, y_chunk, optimizer, batch_size):
        """Process a chunk of data in a worker thread"""
        return self._train_batch(X_chunk, y_chunk, optimizer, batch_size)
    
    def _set_training_mode(self, training_mode):
        """Set all layers to training or evaluation mode"""
        for layer in self.layers:
            if hasattr(layer, 'set_training_mode'):
                layer.set_training_mode(training_mode)
    
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
        params = {
            'layers': [layer.get_parameters() for layer in self.layers],
            'loss_history': self.loss_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, filepath):
        """Load the network from a file"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        for layer_params in params['layers']:
            layer = self._create_layer_from_params(layer_params)
            self.add_layer(layer)
        
        self.loss_history = params['loss_history']
        self._set_training_mode(False)
    
    def _create_layer_from_params(self, params):
        """Create a layer instance from parameters"""
        raise NotImplementedError("Subclasses must implement _create_layer_from_params") 