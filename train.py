"""
Main training script for ECG classification
Reproduces Hannun et al. (2019) methodology on PhysioNet 2017 dataset
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import argparse

from config import Config
from data.dataset import PhysioNet2017Dataset
from models.resnet1d import ResNet1d
from utils.training import Trainer, initialize_weights

def set_seed(seed):
    """
    Set random seeds for reproducibility
    
    Important for reproducing results as in scientific papers
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    """
    Main training function
    
    Implements complete training pipeline from Hannun et al.:
    1. Load and preprocess data
    2. Create train/validation split
    3. Initialize model with He initialization
    4. Train with Adam optimizer and LR scheduling
    5. Validate and save best model
    """
    
    # Configuration
    config = Config()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    print("=" * 60)
    print("ECG CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Reproducing: Hannun et al. (2019) Nature Medicine")
    print(f"Dataset: PhysioNet/CinC Challenge 2017")
    print(f"Model: 34-layer ResNet (ResNet1d)")
    print("=" * 60)
    
    # Device
    device = config.device
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print("\nLoading dataset...")
    # Hannun et al.: "training dataset comprised of 91,232 ECG records from 53,549 patients"
    # PhysioNet 2017: 8,528 records total
    
    full_dataset = PhysioNet2017Dataset(
        data_dir=config.data_dir,
        target_length=config.input_length
    )
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Display class distribution
    # Hannun et al.: "rare rhythms... were intentionally oversampled"
    class_counts = full_dataset.get_class_distribution()
    print(f"\nClass distribution:")
    class_names = ['Normal', 'AF', 'Other', 'Noisy']
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {name}: {count} ({count/len(full_dataset)*100:.1f}%)")
    
    # Train/validation split
    # Hannun et al.: "held-out records from random 10% of training dataset patients 
    # for use as development dataset"
    val_size = int(config.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    # Hannun et al.: "minibatch size of 128"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    # Hannun et al. Extended Data Figure 1: 34-layer architecture
    # - 33 convolutional layers in 16 residual blocks
    # - 1 final linear layer
    model = ResNet1d(
        in_channels=1,  # Single-lead ECG
        base_filters=config.base_filters,  # 32
        kernel_size=config.kernel_size,  # 16
        stride=2,
        n_classes=config.n_classes,  # 4 for PhysioNet 2017
        dropout_rate=config.dropout_rate  # 0.2
    )
    
    # Initialize weights with He initialization
    # Hannun et al.: "network was trained de-novo with random initialization 
    # of weights as described by He et al."
    model = initialize_weights(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Hannun et al. report ~10.5M parameters for their model
    # Sahu et al. mention 115 MB model size (~10.5M params * 4 bytes/param * overhead)
    print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB (32-bit floats)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            start_epoch = trainer.load_checkpoint(args.resume)
        else:
            print(f"\nCheckpoint not found: {args.resume}")
            print("Starting from scratch...")
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)
    
    # Hannun et al. training procedure:
    # - Adam optimizer with β1=0.9, β2=0.999
    # - Initial LR: 1e-3
    # - Reduce LR by 10x when dev loss plateaus for 2 epochs
    # - Save model with best dev set performance
    trainer.train(num_epochs=config.max_epochs)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {os.path.join(config.model_save_path, 'best_model.pth')}")
    
    # Note about next steps
    print("\nNext steps:")
    print("1. Evaluate model on test set using evaluate.py")
    print("2. Compare results with Hannun et al. benchmarks")
    print("3. Apply model compression (Sahu et al. LTH-ECG) if needed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ECG classification model (Hannun et al. 2019 reproduction)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    main(args)