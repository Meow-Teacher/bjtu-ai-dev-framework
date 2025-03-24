import numpy as np
from .network import NeuralNetwork, NetworkConfig
from ..functional.activations import ActivationType
from ..nn.init import InitializationType
from ..nn.regularization import RegularizationType
from ..nn.loss import LossType
from ..optim.optimizer import OptimizerType
from ..layers.dense import DenseLayer
from ..layers.dropout import DropoutLayer
from ..layers.batch_norm import BatchNormLayer

class MLPClassifier(NeuralNetwork):
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
        super().__init__()
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
        
        self.classes_ = None
        self._is_fitted = False
        self.loss_history_ = None
    
    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
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
        config = NetworkConfig()
        
        # Add input layer to first hidden layer
        config.layers.append(DenseLayer(
            n_features, self.hidden_layer_sizes[0],
            self.activation, self.init_type, self.reg_type, self.alpha
        ))
        
        # Add hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1):
            config.layers.append(DenseLayer(
                self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1],
                self.activation, self.init_type, self.reg_type, self.alpha
            ))
        
        # Add output layer
        output_activation = ActivationType.SIGMOID if n_classes == 1 else ActivationType.SOFTMAX
        config.layers.append(DenseLayer(
            self.hidden_layer_sizes[-1], n_classes if n_classes > 1 else 1,
            output_activation, self.init_type, RegularizationType.NONE, 0.0
        ))
        
        # Set optimizer
        config.optimizer_type = self.solver
        config.optimizer_params = {'learning_rate': self.learning_rate}
        
        # Set loss function
        config.loss_type = LossType.BINARY_CROSS_ENTROPY if n_classes == 1 else LossType.CROSS_ENTROPY
        
        # Configure early stopping if enabled
        if self.early_stopping:
            config.stop_criteria = StopCriteriaType.EARLY_STOPPING
            config.max_epochs = self.max_iter
            config.patience = self.n_iter_no_change
            config.min_delta = 1e-4
        else:
            config.stop_criteria = StopCriteriaType.MAX_EPOCHS
            config.max_epochs = self.max_iter
        
        # Set validation split
        config.validation_split = self.validation_split
        
        # Set batch parameters
        config.batch_size = self.batch_size
        
        # Train the network
        self.loss_history_ = super().fit(X, y, config)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
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
        X : array-like of shape (n_samples, n_features)
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
        y_proba = super().predict(X)
        
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

class MLPRegressor(NeuralNetwork):
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
        super().__init__()
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
        
        self._is_fitted = False
        self.loss_history_ = None
        self.n_outputs_ = None
    
    def fit(self, X, y):
        """
        Fit the model to data matrix X and target y.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
            
        Returns:
        --------
        self : returns a fitted MLPRegressor instance.
        """
        # Convert inputs to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        # Handle target shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Get number of features and outputs
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        self.n_outputs_ = n_outputs
        
        # Build network configuration
        config = NetworkConfig()
        
        # Add input layer to first hidden layer
        config.layers.append(DenseLayer(
            n_features, self.hidden_layer_sizes[0],
            self.activation, self.init_type, self.reg_type, self.alpha
        ))
        
        # Add hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1):
            config.layers.append(DenseLayer(
                self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1],
                self.activation, self.init_type, self.reg_type, self.alpha
            ))
        
        # Add output layer with linear activation for regression
        config.layers.append(DenseLayer(
            self.hidden_layer_sizes[-1], n_outputs,
            ActivationType.LINEAR, self.init_type, RegularizationType.NONE, 0.0
        ))
        
        # Set optimizer
        config.optimizer_type = self.solver
        config.optimizer_params = {'learning_rate': self.learning_rate}
        
        # Set loss function (MSE for regression)
        config.loss_type = LossType.MSE
        
        # Configure early stopping if enabled
        if self.early_stopping:
            config.stop_criteria = StopCriteriaType.EARLY_STOPPING
            config.max_epochs = self.max_iter
            config.patience = self.n_iter_no_change
            config.min_delta = 1e-4
        else:
            config.stop_criteria = StopCriteriaType.MAX_EPOCHS
            config.max_epochs = self.max_iter
        
        # Set validation split
        config.validation_split = self.validation_split
        
        # Set batch parameters
        config.batch_size = self.batch_size
        
        # Train the network
        self.loss_history_ = super().fit(X, y, config)
        
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict values for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before predicting")
        
        # Convert input to numpy array
        X = np.array(X, dtype=np.float64)
        
        # Get predictions
        y_pred = super().predict(X)
        
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
        
        # Convert inputs to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        # Handle target shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Get predictions
        y_pred = self.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Calculate R^2 score
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean(axis=0)) ** 2).sum()
        
        # Handle edge case to avoid division by zero
        if v == 0:
            return 0.0
        
        return 1 - u / v 