import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    """Confusion matrix and classification metrics"""
    def __init__(self, y_true, y_pred, threshold=0.5):
        """
        Initialize a confusion matrix
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels/probabilities
            threshold: Threshold for converting probabilities to binary predictions (for binary classification)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.threshold = threshold
        self.num_classes = None
        self.matrix = None
        self.metrics = {}
        
        self._compute_matrix()
        self._compute_metrics()
    
    def _compute_matrix(self):
        """Compute the confusion matrix"""
        # Handle different input formats
        if len(self.y_true.shape) > 1 and self.y_true.shape[1] > 1:
            # Multi-class one-hot encoded
            self.num_classes = self.y_true.shape[1]
            y_true_class = np.argmax(self.y_true, axis=1)
            y_pred_class = np.argmax(self.y_pred, axis=1)
        elif len(self.y_pred.shape) > 1 and self.y_pred.shape[1] > 1:
            # Predicted probabilities for multi-class
            self.num_classes = self.y_pred.shape[1]
            y_true_class = self.y_true.astype(int)
            y_pred_class = np.argmax(self.y_pred, axis=1)
        else:
            # Binary classification
            self.num_classes = 2
            y_true_class = self.y_true.flatten().astype(int)
            y_pred_class = (self.y_pred.flatten() >= self.threshold).astype(int)
        
        # Create confusion matrix
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(len(y_true_class)):
            self.matrix[y_true_class[i], y_pred_class[i]] += 1
    
    def _compute_metrics(self):
        """Compute classification metrics from the confusion matrix"""
        # Overall accuracy
        self.metrics['accuracy'] = np.trace(self.matrix) / np.sum(self.matrix)
        
        # Per-class metrics
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(self.num_classes):
            # True positives for this class
            tp = self.matrix[i, i]
            
            # False positives for this class (sum of column i excluding tp)
            fp = np.sum(self.matrix[:, i]) - tp
            
            # False negatives for this class (sum of row i excluding tp)
            fn = np.sum(self.matrix[i, :]) - tp
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)
            
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            
            # F1 score = 2 * (Precision * Recall) / (Precision + Recall)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        self.metrics['precision'] = np.array(precisions)
        self.metrics['recall'] = np.array(recalls)
        self.metrics['f1_score'] = np.array(f1_scores)
        
        # Macro-averaged metrics
        self.metrics['macro_precision'] = np.mean(precisions)
        self.metrics['macro_recall'] = np.mean(recalls)
        self.metrics['macro_f1'] = np.mean(f1_scores)
    
    def display(self):
        """Display the confusion matrix and metrics"""
        print("Confusion Matrix:")
        print(self.matrix)
        print("\nMetrics:")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Macro Precision: {self.metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {self.metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {self.metrics['macro_f1']:.4f}")
        
        print("\nPer-class Metrics:")
        for i in range(self.num_classes):
            print(f"Class {i} - Precision: {self.metrics['precision'][i]:.4f}, Recall: {self.metrics['recall'][i]:.4f}, F1: {self.metrics['f1_score'][i]:.4f}")
    
    def plot(self, class_names=None):
        """Plot the confusion matrix as a heatmap"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = self.matrix.max() / 2
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, str(self.matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if self.matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show() 