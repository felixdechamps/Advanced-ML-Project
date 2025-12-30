"""
Training utilities for ECG classification
Based on Hannun et al. (2019) training methodology
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm

class Trainer:
    """
    Trainer class implementing Hannun et al. training procedure
    
    Key training details from paper:
    - Adam optimizer with β1=0.9, β2=0.999
    - Initial learning rate: 1e-3
    - Learning rate reduction by factor of 10 when dev loss plateaus for 2 epochs
    - Random weight initialization (He et al., 2015)
    - Minibatch size: 128
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Args:
            model: PyTorch model (ResNet1d)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to train on (cuda/cpu)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        # Cross-entropy for multi-class classification
        # Hannun et al.: "final fully-connected softmax layer"
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        # Hannun et al.: "Adam optimizer with default parameters (β1=0.9 and β2=0.999)"
        # "initialized learning rate to 1e-3"
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Learning rate scheduler
        # Hannun et al.: "reduced [learning rate] by factor of ten when 
        # development-set loss stopped improving for two consecutive epochs"
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_factor,  # 0.1
            patience=config.lr_patience,  # 2 epochs
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
    
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training')
        
        for signals, labels in pbar:
            # Move to device
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # Hannun et al.: "network takes as input raw ECG data"
            outputs = self.model(signals)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            # Hannun et al. use Adam optimizer with standard update rules
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self):
        """
        Validate model on validation set
        
        Returns:
            Tuple of (avg_loss, f1_score)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(signals)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                n_batches += 1
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        
        # Compute F1 score
        # Sahu et al. use mean F1-score as primary evaluation metric
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, f1
    
    def save_checkpoint(self, epoch, filepath):
        """
        Save model checkpoint
        
        Hannun et al.: "chose model that achieved lowest error on development dataset"
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_f1_scores = checkpoint['val_f1_scores']
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch']
    
    def train(self, num_epochs):
        """
        Main training loop
        
        Implements training procedure from Hannun et al.:
        - Train until development loss stops improving
        - Save best model based on development set performance
        - Early stopping if no improvement
        
        Args:
            num_epochs: Maximum number of epochs to train
        """
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Initial learning rate: {self.config.learning_rate}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val F1:     {val_f1:.4f}")
            
            # Learning rate scheduling
            # Hannun et al.: "reduced [LR] by factor of ten when development-set 
            # loss stopped improving for two consecutive epochs"
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            # Hannun et al.: "chose model that achieved lowest error on development dataset"
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(
                    self.config.model_save_path,
                    'best_model.pth'
                )
                os.makedirs(self.config.model_save_path, exist_ok=True)
                self.save_checkpoint(epoch, checkpoint_path)
                
                print(f"✓ New best model! F1: {val_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation F1: {self.best_val_f1:.4f}")
                break
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)

def initialize_weights(model):
    """
    Initialize model weights
    
    Hannun et al.: "network was trained de-novo with random initialization 
    of weights as described by He et al."
    
    This refers to He initialization (He et al., 2015 - Delving Deep into Rectifiers)
    which is the default initialization in PyTorch for Conv and Linear layers
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # He initialization for Conv layers
            # PyTorch default, but explicitly setting for clarity
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            # BatchNorm initialization
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # He initialization for Linear layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    return model