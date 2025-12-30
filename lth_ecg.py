"""
LTH-ECG: Lottery Ticket Hypothesis-based model compression
Implementation of Sahu et al. (2022) compression method

Goal: Reduce model from 10.5M parameters to ~74K (142x reduction)
while maintaining F1-score within 1% of original

Reference: Sahu et al. (2022) "LTH-ECG: Lottery Ticket Hypothesis-based 
Deep Learning Model Compression for Atrial Fibrillation Detection"
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm

class LTHECGPruner:
    """
    Lottery Ticket Hypothesis-based pruner for ECG models
    
    Sahu et al. Algorithm: "Goal-driven winning lottery ticket discovery"
    
    Key idea:
    1. Start with initial pruning quantity p_init (e.g., 30%)
    2. Iteratively prune least-magnitude weights
    3. Decrease pruning rate exponentially (p_init / α each round)
    4. Stop when target parameter reduction θ_target is reached
    5. Reset remaining weights to initial values
    
    This avoids over-pruning and discovers optimal sparse sub-networks
    """
    
    def __init__(self, model, target_reduction_factor=142, 
                 initial_prune_rate=0.30, alpha=1.1):
        """
        Args:
            model: PyTorch model to prune
            target_reduction_factor: θ_target from Sahu et al. (default: 142)
            initial_prune_rate: p_init - initial percentage to prune (default: 30%)
            alpha: Factor to reduce pruning rate each iteration (default: 1.1)
        
        Sahu et al. report:
        - θ_target = 142 (142x parameter reduction)
        - p_init = 30
        - α = 1.1
        - Final F1-score: 0.8360 vs 0.8360 (benchmark) - no degradation!
        """
        self.model = model
        self.target_reduction_factor = target_reduction_factor
        self.initial_prune_rate = initial_prune_rate
        self.alpha = alpha
        
        # Store initial weights for reset
        # Frankle & Carbin (2019): "reset surviving weights to initial values"
        self.initial_weights = self._get_weights()
        self.initial_param_count = self._count_parameters()
        
        # Target parameter count
        # Sahu et al.: "discover that sparse model with parameter reduction factor θ_target"
        self.target_param_count = self.initial_param_count // target_reduction_factor
        
        # Pruning mask (1 = keep, 0 = prune)
        self.mask = {name: torch.ones_like(param) 
                     for name, param in model.named_parameters() 
                     if 'weight' in name and param.requires_grad}
        
        print(f"LTH-ECG Pruner initialized")
        print(f"Initial parameters: {self.initial_param_count:,}")
        print(f"Target parameters: {self.target_param_count:,}")
        print(f"Target reduction: {target_reduction_factor}x")
    
    def _get_weights(self):
        """
        Get copy of current model weights
        
        Frankle & Carbin (2019): "At end of each round, surviving weights 
        are reset to their initial values"
        """
        weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights[name] = param.data.clone()
        return weights
    
    def _count_parameters(self, include_pruned=False):
        """
        Count number of parameters in model
        
        Args:
            include_pruned: If False, only count non-zero parameters
        
        Returns:
            Number of parameters
        """
        if include_pruned:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            count = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # Count non-zero parameters (not pruned)
                    count += (self.mask[name] > 0).sum().item()
            return count
    
    def _apply_mask(self):
        """
        Apply pruning mask to model weights
        
        Sets pruned weights to 0
        """
        for name, param in self.model.named_parameters():
            if name in self.mask:
                param.data *= self.mask[name]
    
    def _global_magnitude_prune(self, prune_rate):
        """
        Prune weights globally based on L1 magnitude
        
        Sahu et al.: "Prune p_init% of weights which are of least magnitude"
        
        Global pruning (used in fine-tuned global pruning, Sahu et al. Table III):
        - Consider all weights across all layers together
        - Prune the smallest magnitude weights globally
        
        Args:
            prune_rate: Percentage of remaining weights to prune (0-1)
        
        Returns:
            Number of parameters pruned
        """
        # Collect all weights with current mask
        all_weights = []
        for name, param in self.model.named_parameters():
            if name in self.mask:
                # Only consider currently active (non-pruned) weights
                active_weights = param.data[self.mask[name] > 0]
                all_weights.append(active_weights.abs().flatten())
        
        if len(all_weights) == 0:
            return 0
        
        # Concatenate all active weights
        all_weights = torch.cat(all_weights)
        
        # Calculate number of weights to prune
        n_active = len(all_weights)
        n_prune = int(n_active * prune_rate)
        
        if n_prune == 0:
            return 0
        
        # Find threshold: prune weights below this value
        # Sahu et al.: "weights which are of least magnitude"
        threshold = torch.sort(all_weights)[0][n_prune]
        
        # Update masks
        pruned_count = 0
        for name, param in self.model.named_parameters():
            if name in self.mask:
                # Prune weights below threshold
                prune_mask = (param.data.abs() <= threshold) & (self.mask[name] > 0)
                self.mask[name][prune_mask] = 0
                pruned_count += prune_mask.sum().item()
        
        return pruned_count
    
    def _reset_weights(self):
        """
        Reset remaining (non-pruned) weights to initial values
        
        Frankle & Carbin (2019) Lottery Ticket Hypothesis:
        "At the end of each round, the surviving weights are reset to 
        their initial values"
        
        Sahu et al. Step 7: "Reset the remaining parameters to their values as in Step 2"
        """
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                # Reset to initial values, but keep pruned weights at 0
                param.data = self.initial_weights[name] * self.mask[name]
    
    def prune_iteratively(self, train_fn, max_iterations=20):
        """
        Iterative magnitude pruning with goal-driven stopping
        
        Sahu et al. Algorithm - LTH-ECG procedure:
        
        Step 1: Set initial pruning quantity p_init
        Step 2: Randomly initialize network
        Step 3: Train the network
        Step 4: Prune p_init% of least magnitude weights
        Step 5: if η/η' ≥ θ_target, STOP and output pruned model
        Step 6: else, p_init = p_init / α
        Step 7: Reset remaining parameters to initial values
        Step 8: Go to Step 3
        
        Args:
            train_fn: Function to train model, signature: train_fn(model) -> val_f1
            max_iterations: Maximum pruning iterations (safety limit)
        
        Returns:
            Dictionary with pruning results
        """
        print("\n" + "=" * 60)
        print("LTH-ECG: Goal-Driven Winning Lottery Ticket Discovery")
        print("=" * 60)
        print(f"Target reduction: {self.target_reduction_factor}x")
        print(f"Initial pruning rate: {self.initial_prune_rate * 100:.1f}%")
        print(f"Pruning rate decay factor: {self.alpha}")
        
        current_prune_rate = self.initial_prune_rate
        iteration = 0
        
        pruning_history = []
        
        while iteration < max_iterations:
            iteration += 1
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            print(f"{'='*60}")
            
            # Step 3: Train the network
            # Sahu et al.: "Train the DL network with the given data"
            print(f"Training model...")
            val_f1 = train_fn(self.model)
            
            current_params = self._count_parameters(include_pruned=False)
            current_reduction = self.initial_param_count / current_params
            
            print(f"Validation F1: {val_f1:.4f}")
            print(f"Current parameters: {current_params:,}")
            print(f"Current reduction: {current_reduction:.1f}x")
            
            # Step 5: Check if target reached
            # Sahu et al.: "if η/η' ≥ θ_target, STOP"
            if current_reduction >= self.target_reduction_factor:
                print(f"\n✓ Target reduction {self.target_reduction_factor}x reached!")
                print(f"Final parameters: {current_params:,}")
                print(f"Final reduction: {current_reduction:.1f}x")
                print(f"Final F1-score: {val_f1:.4f}")
                break
            
            # Step 4: Prune p_init% of weights
            # Sahu et al.: "Prune p_init% of weights which are of least magnitude"
            print(f"\nPruning {current_prune_rate*100:.2f}% of remaining weights...")
            pruned = self._global_magnitude_prune(current_prune_rate)
            
            # Apply mask
            self._apply_mask()
            
            remaining_params = self._count_parameters(include_pruned=False)
            reduction_factor = self.initial_param_count / remaining_params
            
            print(f"Pruned {pruned:,} weights")
            print(f"Remaining parameters: {remaining_params:,}")
            print(f"Reduction factor: {reduction_factor:.1f}x")
            
            # Save iteration results
            pruning_history.append({
                'iteration': iteration,
                'prune_rate': current_prune_rate,
                'parameters': remaining_params,
                'reduction_factor': reduction_factor,
                'val_f1': val_f1
            })
            
            # Step 6: Reduce pruning rate
            # Sahu et al.: "p_init = p_init / α, typically α > 1. Here, α = 1.1"
            current_prune_rate = current_prune_rate / self.alpha
            
            # Step 7: Reset weights
            # Frankle & Carbin (2019): "reset surviving weights to initial values"
            print(f"Resetting weights to initial values...")
            self._reset_weights()
            
            # Step 8: Continue to next iteration (loop)
        
        # Final statistics
        print("\n" + "=" * 60)
        print("LTH-ECG PRUNING COMPLETE")
        print("=" * 60)
        
        final_params = self._count_parameters(include_pruned=False)
        final_reduction = self.initial_param_count / final_params
        sparsity = 1.0 - (final_params / self.initial_param_count)
        
        print(f"\nInitial parameters: {self.initial_param_count:,}")
        print(f"Final parameters: {final_params:,}")
        print(f"Reduction factor: {final_reduction:.1f}x")
        print(f"Sparsity: {sparsity*100:.2f}%")
        print(f"Model size reduction: {final_reduction:.1f}x")
        
        # Sahu et al. Table III comparison
        print("\n" + "-" * 60)
        print("Comparison with Sahu et al. (2022) Table III:")
        print(f"  Target reduction: {self.target_reduction_factor}x")
        print(f"  Achieved reduction: {final_reduction:.1f}x")
        
        if final_reduction >= self.target_reduction_factor:
            print(f"  ✓ Target achieved!")
        else:
            print(f"  ⚠ Target not reached (may need more iterations)")
        
        return {
            'initial_params': self.initial_param_count,
            'final_params': final_params,
            'reduction_factor': final_reduction,
            'sparsity': sparsity,
            'iterations': iteration,
            'pruning_history': pruning_history,
            'mask': self.mask
        }
    
    def save_pruned_model(self, filepath):
        """
        Save pruned model with mask
        
        Saves:
        - Model state dict (with 0s for pruned weights)
        - Pruning mask
        - Pruning statistics
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'mask': self.mask,
            'initial_param_count': self.initial_param_count,
            'final_param_count': self._count_parameters(include_pruned=False),
            'reduction_factor': self.initial_param_count / self._count_parameters(include_pruned=False),
            'target_reduction_factor': self.target_reduction_factor
        }
        
        torch.save(checkpoint, filepath)
        print(f"\nPruned model saved to {filepath}")

def example_train_function(model):
    """
    Example training function for LTH-ECG pruning
    
    In practice, this should:
    1. Train model for several epochs
    2. Validate on development set
    3. Return validation F1-score
    
    Args:
        model: PyTorch model to train
    
    Returns:
        Validation F1-score
    """
    # Placeholder - implement actual training logic
    # See Trainer class in utils/training.py
    
    print("  Training for N epochs...")
    # ... training code ...
    
    val_f1 = 0.83  # Example validation F1
    return val_f1

# Usage example
if __name__ == '__main__':
    """
    Example usage of LTH-ECG pruner
    
    Steps:
    1. Load trained baseline model (from Hannun et al. reproduction)
    2. Create LTH-ECG pruner
    3. Define training function
    4. Run iterative pruning
    5. Save compressed model
    
    Expected result (Sahu et al. Table III):
    - Parameter reduction: 142x
    - F1-score: 0.8360 (vs 0.8360 baseline)
    - Model size: ~0.8 MB (vs 115 MB)
    """
    
    print("LTH-ECG Compression Example")
    print("=" * 60)
    
    # This is a placeholder - replace with actual model and training
    print("\nNote: This is an example. For actual usage:")
    print("1. Train baseline model using train.py")
    print("2. Load trained model checkpoint")
    print("3. Implement proper train_function with your data loaders")
    print("4. Run pruning with: pruner.prune_iteratively(train_function)")
    
    # Example (pseudocode):
    """
    from models.resnet1d import ResNet1d
    from config import Config
    
    # Load baseline model
    config = Config()
    model = ResNet1d(...)
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create pruner
    pruner = LTHECGPruner(
        model=model,
        target_reduction_factor=142,  # Sahu et al. result
        initial_prune_rate=0.30,
        alpha=1.1
    )
    
    # Define training function
    def train_fn(model):
        # Train for a few epochs
        trainer = Trainer(model, train_loader, val_loader, config, device)
        # ... training ...
        return val_f1_score
    
    # Run pruning
    results = pruner.prune_iteratively(train_fn, max_iterations=20)
    
    # Save compressed model
    pruner.save_pruned_model('checkpoints/lth_ecg_compressed.pth')
    
    # Expected: 142x reduction, F1 ≈ 0.836 (within 1% of baseline)
    """