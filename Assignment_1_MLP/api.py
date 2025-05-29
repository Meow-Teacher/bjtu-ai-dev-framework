from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import numpy as np
import uuid
import datetime
import os
import json
import pickle
from mlp import MLPClassifier, MLPRegressor, ActivationType, OptimizerType, RegularizationType, InitializationType

app = FastAPI(title="MLP Neural Network API", description="API for training and inference with MLP neural networks")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models storage
models = {}
training_tasks = {}

# Models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Data models
class ModelConfig(BaseModel):
    hidden_layer_sizes: List[int]
    activation: str
    init_type: str
    solver: str
    alpha: float
    batch_size: int
    learning_rate: float
    max_iter: int
    validation_split: float
    early_stopping: bool
    n_iter_no_change: int
    reg_type: str
    use_parallel: bool
    num_threads: int

class CreateModelRequest(BaseModel):
    name: str
    type: str  # "classifier" or "regressor"
    config: ModelConfig

class TrainingData(BaseModel):
    X: List[List[float]]
    y: List[Union[float, int, List[float]]]

class TrainingRequest(BaseModel):
    data: TrainingData
    training_params: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    data: List[List[float]]

# Helper functions
def get_activation_type(activation_str):
    activation_map = {
        "relu": ActivationType.RELU,
        "sigmoid": ActivationType.SIGMOID,
        "tanh": ActivationType.TANH,
        "leaky_relu": ActivationType.LEAKY_RELU,
        "softmax": ActivationType.SOFTMAX,
        "linear": ActivationType.LINEAR
    }
    return activation_map.get(activation_str, ActivationType.RELU)

def get_optimizer_type(optimizer_str):
    optimizer_map = {
        "adam": OptimizerType.ADAM,
        "sgd": OptimizerType.SGD,
        "momentum": OptimizerType.MOMENTUM,
        "rmsprop": OptimizerType.RMSPROP
    }
    return optimizer_map.get(optimizer_str, OptimizerType.ADAM)

def get_regularization_type(reg_str):
    reg_map = {
        "none": RegularizationType.NONE,
        "l1": RegularizationType.L1,
        "l2": RegularizationType.L2,
        "elastic_net": RegularizationType.ELASTIC_NET
    }
    return reg_map.get(reg_str, RegularizationType.NONE)

def get_initialization_type(init_str):
    init_map = {
        "zero": InitializationType.ZERO,
        "random": InitializationType.RANDOM,
        "xavier": InitializationType.XAVIER,
        "he": InitializationType.HE
    }
    return init_map.get(init_str, InitializationType.HE)

def save_model(model_id, model_obj):
    """Save model to disk"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_obj, f)

def load_model(model_id):
    """Load model from disk"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def train_model_task(model_id, X, y, training_params=None):
    """Background task for model training"""
    try:
        # Update task status
        training_tasks[model_id]["status"] = "training"
        
        # Get model
        model_info = models[model_id]
        model_obj = model_info["model_obj"]
        
        # Set training parameters if provided
        if training_params:
            if "batch_size" in training_params:
                model_obj.batch_size = training_params["batch_size"]
            if "learning_rate" in training_params:
                model_obj.learning_rate = training_params["learning_rate"]
            if "max_iter" in training_params:
                model_obj.max_iter = training_params["max_iter"]
        
        # Train the model
        model_obj.fit(X, y)
        
        # Update model status
        model_info["status"] = "trained"
        model_info["is_fitted"] = True
        model_info["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save loss history
        training_tasks[model_id]["loss_history"] = {
            "train": model_obj.loss_history_["train"] if "train" in model_obj.loss_history_ else [],
            "val": model_obj.loss_history_["val"] if "val" in model_obj.loss_history_ else []
        }
        
        # Update task status
        training_tasks[model_id]["status"] = "completed"
        
        # Save model to disk
        save_model(model_id, model_obj)
        
    except Exception as e:
        # Update task status on error
        training_tasks[model_id]["status"] = "failed"
        training_tasks[model_id]["error"] = str(e)
        print(f"Training error: {str(e)}")

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "MLP Neural Network API", "version": "1.0.0"}

@app.get("/api/models")
def get_models():
    """Get list of all models"""
    result = []
    for model_id, model_info in models.items():
        # Exclude model object from response
        model_data = {k: v for k, v in model_info.items() if k != "model_obj"}
        result.append(model_data)
    return result

@app.post("/api/models")
def create_model(request: CreateModelRequest):
    """Create a new model"""
    model_id = str(uuid.uuid4())
    
    # Convert config parameters
    hidden_layer_sizes = tuple(request.config.hidden_layer_sizes)
    activation = get_activation_type(request.config.activation)
    init_type = get_initialization_type(request.config.init_type)
    solver = get_optimizer_type(request.config.solver)
    reg_type = get_regularization_type(request.config.reg_type)
    
    # Create model object based on type
    if request.type == "classifier":
        model_obj = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            init_type=init_type,
            solver=solver,
            alpha=request.config.alpha,
            batch_size=request.config.batch_size,
            learning_rate=request.config.learning_rate,
            max_iter=request.config.max_iter,
            validation_split=request.config.validation_split,
            early_stopping=request.config.early_stopping,
            n_iter_no_change=request.config.n_iter_no_change,
            reg_type=reg_type
        )
    elif request.type == "regressor":
        model_obj = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            init_type=init_type,
            solver=solver,
            alpha=request.config.alpha,
            batch_size=request.config.batch_size,
            learning_rate=request.config.learning_rate,
            max_iter=request.config.max_iter,
            validation_split=request.config.validation_split,
            early_stopping=request.config.early_stopping,
            n_iter_no_change=request.config.n_iter_no_change,
            reg_type=reg_type
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Must be 'classifier' or 'regressor'")
    
    # Store model info
    created_at = datetime.datetime.now().isoformat()
    models[model_id] = {
        "model_id": model_id,
        "name": request.name,
        "type": request.type,
        "config": request.config.dict(),
        "status": "created",
        "is_fitted": False,
        "created_at": created_at,
        "updated_at": created_at,
        "model_obj": model_obj
    }
    
    # Save model to disk
    save_model(model_id, model_obj)
    
    # Return model info (excluding model object)
    return {k: v for k, v in models[model_id].items() if k != "model_obj"}

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    """Get model details"""
    if model_id not in models:
        # Try to load from disk
        model_obj = load_model(model_id)
        if model_obj is None:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        # Create model info
        models[model_id] = {
            "model_id": model_id,
            "name": f"Loaded model {model_id}",
            "type": "classifier" if isinstance(model_obj, MLPClassifier) else "regressor",
            "config": {
                "hidden_layer_sizes": model_obj.hidden_layer_sizes,
                "activation": model_obj.activation.value,
                "init_type": model_obj.init_type.value,
                "solver": model_obj.solver.value,
                "alpha": model_obj.alpha,
                "batch_size": model_obj.batch_size,
                "learning_rate": model_obj.learning_rate,
                "max_iter": model_obj.max_iter,
                "validation_split": model_obj.validation_split,
                "early_stopping": model_obj.early_stopping,
                "n_iter_no_change": model_obj.n_iter_no_change,
                "reg_type": model_obj.reg_type.value,
                "use_parallel": False,
                "num_threads": 1
            },
            "status": "trained" if model_obj._is_fitted else "created",
            "is_fitted": model_obj._is_fitted,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "model_obj": model_obj
        }
    
    # Return model info (excluding model object)
    return {k: v for k, v in models[model_id].items() if k != "model_obj"}

@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    """Delete a model"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Remove from memory
    del models[model_id]
    
    # Remove from disk if exists
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Remove training task if exists
    if model_id in training_tasks:
        del training_tasks[model_id]
    
    return {"message": f"Model {model_id} deleted successfully"}

@app.post("/api/models/{model_id}/train")
def train_model_endpoint(model_id: str, request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a model"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
        # Convert data to numpy arrays
    try:
        X = np.array(request.data.X, dtype=np.float64)
    except (ValueError, TypeError):
        # If conversion fails, keep original data
        X = request.data.X
    y = np.array(request.data.y)
    
    # Create training task
    training_tasks[model_id] = {
        "status": "queued",
        "started_at": datetime.datetime.now().isoformat(),
        "loss_history": {"train": [], "val": []}
    }
    
    # Start training in background
    background_tasks.add_task(
        train_model_task, 
        model_id, 
        X, 
        y, 
        request.training_params
    )
    
    return {"message": f"Training started for model {model_id}", "task_id": model_id}


@app.get("/api/models/{model_id}/train/status")
def get_training_status(model_id: str):
    """Get training status"""
    if model_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"No training task found for model {model_id}")
    
    return training_tasks[model_id]

@app.post("/api/models/{model_id}/predict")
def predict(model_id: str, request: PredictionRequest):
    """Make predictions with a model"""
    if model_id not in models:
        # Try to load from disk
        model_obj = load_model(model_id)
        if model_obj is None:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        # Create model info
        models[model_id] = {
            "model_id": model_id,
            "name": f"Loaded model {model_id}",
            "type": "classifier" if isinstance(model_obj, MLPClassifier) else "regressor",
            "config": {},  # Simplified for brevity
            "status": "trained" if model_obj._is_fitted else "created",
            "is_fitted": model_obj._is_fitted,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "model_obj": model_obj
        }
    
    model_info = models[model_id]
    model_obj = model_info["model_obj"]
    
    if not model_info["is_fitted"]:
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not trained yet")
    
    try:
        # Validate input data
        if not request.data or len(request.data) == 0:
            raise ValueError("Empty input data")
        
        # Check if all rows have the same number of features
        num_features = len(request.data[0])
        if any(len(row) != num_features for row in request.data):
            raise ValueError("Inconsistent number of features across input samples")
        
        # Check if all values are numeric
        for row in request.data:
            for val in row:
                if not isinstance(val, (int, float)) and val is not None:
                    raise ValueError(f"Non-numeric value found in input: {val}")
        
        # Convert data to numpy array
        X = np.array(request.data, dtype=np.float64)
        
        # Verify that X has the correct number of features
        expected_features = model_obj.network.layers[0].input_size
        if X.shape[1] != expected_features:
            raise ValueError(f"Input has {X.shape[1]} features, but model expects {expected_features} features")
        
        # Make predictions
        if model_info["type"] == "classifier":
            # For classifiers, return both class predictions and probabilities
            predictions = model_obj.predict(X)
            probabilities = model_obj.predict_proba(X)
            return {
                "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                "probabilities": probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
            }
        else:
            # For regressors, return only predictions
            predictions = model_obj.predict(X)
            return {
                "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Load existing models on startup
@app.on_event("startup")
def load_existing_models():
    """Load existing models from disk on startup"""
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".pkl"):
                model_id = filename.split(".")[0]
                try:
                    model_obj = load_model(model_id)
                    if model_obj:
                        # Create model info
                        models[model_id] = {
                            "model_id": model_id,
                            "name": f"Loaded model {model_id}",
                            "type": "classifier" if isinstance(model_obj, MLPClassifier) else "regressor",
                            "config": {
                                "hidden_layer_sizes": model_obj.hidden_layer_sizes,
                                "activation": model_obj.activation.value if hasattr(model_obj.activation, "value") else "relu",
                                "solver": model_obj.solver.value if hasattr(model_obj.solver, "value") else "adam",
                                "alpha": model_obj.alpha,
                                "batch_size": model_obj.batch_size,
                                "learning_rate": model_obj.learning_rate,
                                "max_iter": model_obj.max_iter,
                                "validation_split": model_obj.validation_split,
                                "early_stopping": model_obj.early_stopping,
                                "n_iter_no_change": model_obj.n_iter_no_change,
                                "reg_type": model_obj.reg_type.value if hasattr(model_obj.reg_type, "value") else "none",
                                "use_parallel": False,
                                "num_threads": 1
                            },
                            "status": "trained" if model_obj._is_fitted else "created",
                            "is_fitted": model_obj._is_fitted,
                            "created_at": datetime.datetime.now().isoformat(),
                            "updated_at": datetime.datetime.now().isoformat(),
                            "model_obj": model_obj
                        }
                        print(f"Loaded model {model_id} from disk")
                except Exception as e:
                    print(f"Error loading model {model_id}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 