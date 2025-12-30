"""
Evaluation script for ECG classification model
Reproduces evaluation methodology from Hannun et al. (2019) and Sahu et al. (2022)
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data.dataset import PhysioNet2017Dataset
from models.resnet1d import ResNet1d
from utils.metrics import ECGMetrics, evaluate_model

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Hannun et al. Figure 2: Display confusion matrices for visualization
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """
    Plot ROC curves for each class
    
    Hannun et al. Figure 1 and Extended Data Figure 2: ROC curves for each rhythm class
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curve for each class
    # Hannun et al.: "one vs. other strategy"
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{class_name} (AUC = {roc_auc:.3f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()

def main(args):
    """
    Main evaluation function
    
    Evaluates trained model and computes metrics as in:
    - Hannun et al. (2019): AUC, F1, sensitivity, specificity
    - Sahu et al. (2022): Mean F1-score comparison
    """
    
    # Configuration
    config = Config()
    device = config.device
    
    print("=" * 60)
    print("ECG CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    
    # For PhysioNet 2017, use official test set if available
    # Otherwise, use a held-out portion from training
    test_dataset = PhysioNet2017Dataset(
        data_dir=args.data_dir if args.data_dir else config.data_dir,
        target_length=config.input_length
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = ResNet1d(
        in_channels=1,
        base_filters=config.base_filters,
        kernel_size=config.kernel_size,
        stride=2,
        n_classes=config.n_classes,
        dropout_rate=config.dropout_rate
    )
    
    # Load trained weights
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print("Model loaded successfully!")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        n_classes=config.n_classes
    )
    
    # Extract results
    y_true = results['y_true']
    y_pred = results['y_pred']
    y_prob = results['y_prob']
    metrics = results['metrics']
    
    # Print metrics
    print("\n")
    metrics_calculator = ECGMetrics(n_classes=config.n_classes)
    metrics_calculator.print_metrics(metrics)
    
    # Compare with benchmarks
    print("\n" + "=" * 60)
    print("COMPARISON WITH BENCHMARKS")
    print("=" * 60)
    
    # Hannun et al. results on PhysioNet 2017 (Supplementary Table 7)
    # Reported in their validation on this specific dataset
    hannun_f1_scores = {
        'Normal': 0.909,  # Normal sinus rhythm
        'AF': 0.827,      # Atrial Fibrillation
        'Other': 0.772,   # Other rhythms
        'Noisy': 0.506    # Noisy recordings
    }
    hannun_mean_f1 = 0.836  # Mean F1-score
    
    print("\nHannun et al. (2019) - Supplementary Table 7:")
    print(f"  Mean F1-score: {hannun_mean_f1:.3f}")
    for class_name, f1 in hannun_f1_scores.items():
        print(f"  {class_name}: {f1:.3f}")
    
    print(f"\nCurrent model:")
    print(f"  Mean F1-score: {metrics['f1_macro']:.3f}")
    for class_name in ['Normal', 'AF', 'Other', 'Noisy']:
        our_f1 = metrics['f1_per_class'][class_name]
        bench_f1 = hannun_f1_scores[class_name]
        diff = our_f1 - bench_f1
        print(f"  {class_name}: {our_f1:.3f} (Δ = {diff:+.3f})")
    
    # Sahu et al. benchmark
    # Table II: Show class-wise F1 scores after compression
    print("\n" + "-" * 60)
    print("Sahu et al. (2022) - Table II (Uncompressed benchmark):")
    sahu_f1_scores = {
        'Normal': 0.909,
        'AF': 0.827,
        'Other': 0.772,
        'Noisy': 0.506
    }
    sahu_mean_f1 = 0.836
    
    print(f"  Mean F1-score: {sahu_mean_f1:.3f}")
    print("\nOur model is the baseline for applying LTH-ECG compression")
    print("Target: Match Sahu et al. performance before compression")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    our_mean_f1 = metrics['f1_macro']
    f1_diff = our_mean_f1 - hannun_mean_f1
    
    print(f"\nMean F1-score: {our_mean_f1:.4f}")
    print(f"Hannun et al. benchmark: {hannun_mean_f1:.4f}")
    print(f"Difference: {f1_diff:+.4f} ({f1_diff/hannun_mean_f1*100:+.2f}%)")
    
    if abs(f1_diff) < 0.01:
        print("\n✓ Performance matches Hannun et al. benchmark!")
        print("  Ready for model compression (Sahu et al. LTH-ECG)")
    elif f1_diff > 0:
        print("\n✓ Performance exceeds Hannun et al. benchmark!")
    else:
        print("\n⚠ Performance below Hannun et al. benchmark")
        print("  Consider: more training epochs, hyperparameter tuning")
    
    # Visualization
    if args.plot:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Plot confusion matrix
        # Hannun et al. Figure 2
        print("\nPlotting confusion matrix...")
        cm_normalized = metrics['confusion_matrix_normalized']
        plot_confusion_matrix(
            cm_normalized,
            class_names=['Normal', 'AF', 'Other', 'Noisy'],
            save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        # Plot ROC curves
        # Hannun et al. Extended Data Figure 2
        print("Plotting ROC curves...")
        plot_roc_curves(
            y_true,
            y_prob,
            class_names=['Normal', 'AF', 'Other', 'Noisy'],
            save_path=os.path.join(args.output_dir, 'roc_curves.png')
        )
    
    # Save results
    if args.save_results:
        results_path = os.path.join(args.output_dir, 'evaluation_results.npz')
        np.savez(
            results_path,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, np.ndarray))}
        )
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained ECG classification model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to test data directory (default: use config)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save evaluation results to file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save plots and results'
    )
    
    args = parser.parse_args()
    main(args)