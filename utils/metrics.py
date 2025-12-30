"""
Evaluation metrics for ECG classification
Based on Hannun et al. (2019) evaluation methodology
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import torch

class ECGMetrics:
    """
    Metrics calculator for ECG classification
    
    Hannun et al. evaluation metrics:
    - F1 score: "harmonic mean of positive predictive value and sensitivity"
    - AUC: "area under receiver operating characteristic curve"
    - Sensitivity (Recall)
    - Specificity
    - Precision (PPV)
    
    Sahu et al. report mean F1-score as primary metric
    """
    
    def __init__(self, n_classes=4, class_names=None):
        """
        Args:
            n_classes: Number of classes (4 for PhysioNet 2017)
            class_names: List of class names ['Normal', 'AF', 'Other', 'Noisy']
        """
        self.n_classes = n_classes
        
        if class_names is None:
            # Default names from PhysioNet 2017 (Clifford et al., 2017)
            self.class_names = ['Normal', 'AF', 'Other', 'Noisy']
        else:
            self.class_names = class_names
    
    def compute_f1_scores(self, y_true, y_pred, average='macro'):
        """
        Compute F1 scores
        
        Hannun et al. Table 1b: "F1 score... harmonic mean of positive 
        predictive value (precision) and sensitivity (recall)"
        
        Sahu et al. Table II: Report class-wise F1 scores
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'macro' for unweighted mean, 'weighted' for class frequency weights
        
        Returns:
            Dictionary with overall and per-class F1 scores
        """
        # Overall F1 score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class F1 scores
        # Hannun et al. and Sahu et al. both report per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        results = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            results['f1_per_class'][class_name] = f1_per_class[i]
        
        return results
    
    def compute_precision_recall(self, y_true, y_pred):
        """
        Compute precision and recall (sensitivity)
        
        Hannun et al.: Report sensitivity and precision (PPV) in Tables
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with precision and recall metrics
        """
        # Overall metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        results = {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_per_class': {},
            'recall_per_class': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            results['precision_per_class'][class_name] = precision_per_class[i]
            results['recall_per_class'][class_name] = recall_per_class[i]
        
        return results
    
    def compute_specificity(self, y_true, y_pred):
        """
        Compute specificity for each class
        
        Hannun et al. Table 2: Report specificity fixed at cardiologist level
        
        Specificity = TN / (TN + FP)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with per-class specificity
        """
        specificity_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Convert to binary problem (one-vs-rest)
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Compute confusion matrix for this class
            tn, fp, fn, tp = confusion_matrix(
                y_true_binary, 
                y_pred_binary, 
                labels=[0, 1]
            ).ravel()
            
            # Calculate specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_per_class[class_name] = specificity
        
        return {'specificity_per_class': specificity_per_class}
    
    def compute_auc(self, y_true, y_prob):
        """
        Compute Area Under ROC Curve
        
        Hannun et al. Table 1a: "area under receiver operating characteristic curve"
        Report both macro and per-class AUC
        
        Args:
            y_true: True labels (integers)
            y_prob: Predicted probabilities (n_samples, n_classes)
        
        Returns:
            Dictionary with AUC scores
        """
        # Convert y_true to one-hot for multi-class AUC
        y_true_onehot = np.eye(self.n_classes)[y_true]
        
        # Macro AUC (average of per-class AUC)
        # Hannun et al.: "class-weighted average AUC"
        try:
            auc_macro = roc_auc_score(
                y_true_onehot, 
                y_prob, 
                average='macro',
                multi_class='ovr'  # One-vs-Rest as in Hannun et al.
            )
        except ValueError:
            auc_macro = 0.0
        
        # Per-class AUC
        # Hannun et al. report AUC for each rhythm class
        auc_per_class = {}
        for i, class_name in enumerate(self.class_names):
            try:
                auc = roc_auc_score(y_true_onehot[:, i], y_prob[:, i])
                auc_per_class[class_name] = auc
            except ValueError:
                # Handle case where class not present in y_true
                auc_per_class[class_name] = 0.0
        
        return {
            'auc_macro': auc_macro,
            'auc_per_class': auc_per_class
        }
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix
        
        Hannun et al. Figure 2: Show confusion matrices for DNN and cardiologists
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Confusion matrix and normalized version
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalized confusion matrix (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized
        }
    
    def compute_all_metrics(self, y_true, y_pred, y_prob=None):
        """
        Compute all evaluation metrics
        
        Comprehensive evaluation as in Hannun et al. and Sahu et al.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_prob: Predicted probabilities (optional, needed for AUC)
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # F1 scores (primary metric in Sahu et al.)
        # Sahu et al. Table III: "Mean Test F1-score"
        metrics.update(self.compute_f1_scores(y_true, y_pred))
        
        # Precision and Recall
        metrics.update(self.compute_precision_recall(y_true, y_pred))
        
        # Specificity
        metrics.update(self.compute_specificity(y_true, y_pred))
        
        # Confusion Matrix
        metrics.update(self.compute_confusion_matrix(y_true, y_pred))
        
        # AUC (if probabilities provided)
        if y_prob is not None:
            metrics.update(self.compute_auc(y_true, y_prob))
        
        # Overall accuracy (not primary metric but useful)
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print metrics in readable format
        
        Format similar to Sahu et al. Table II and III
        """
        print("=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
        print(f"  Precision:       {metrics['precision_macro']:.4f}")
        print(f"  Recall:          {metrics['recall_macro']:.4f}")
        
        if 'auc_macro' in metrics:
            print(f"  AUC (macro):     {metrics['auc_macro']:.4f}")
        
        # Per-class metrics (as in Sahu et al. Table II)
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'F1':>8} {'Precision':>12} {'Recall':>10} {'Specificity':>12}", end='')
        if 'auc_per_class' in metrics:
            print(f" {'AUC':>8}")
        else:
            print()
        
        print("-" * 60)
        
        for class_name in self.class_names:
            f1 = metrics['f1_per_class'][class_name]
            prec = metrics['precision_per_class'][class_name]
            rec = metrics['recall_per_class'][class_name]
            spec = metrics['specificity_per_class'][class_name]
            
            print(f"{class_name:<15} {f1:>8.4f} {prec:>12.4f} {rec:>10.4f} {spec:>12.4f}", end='')
            
            if 'auc_per_class' in metrics:
                auc = metrics['auc_per_class'][class_name]
                print(f" {auc:>8.4f}")
            else:
                print()
        
        print("=" * 60)

def evaluate_model(model, dataloader, device, n_classes=4):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        n_classes: Number of classes
    
    Returns:
        Dictionary with predictions and metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(signals)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Compute metrics
    metrics_calculator = ECGMetrics(n_classes=n_classes)
    metrics = metrics_calculator.compute_all_metrics(y_true, y_pred, y_prob)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'metrics': metrics
    }