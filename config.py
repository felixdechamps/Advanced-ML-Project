"""
Configuration file for ECG classification
Based on Hannun et al. (2019) Nature Medicine paper specifications
"""

import torch


class Config:
    # Device configuration
    # Hannun et al. used GPU training for their 34-layer network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data parameters
    # PhysioNet 2017 dataset uses 300Hz sampling rate (Clifford et al., 2017)
    # but Hannun et al. used 200Hz for Zio patch data
    # For PhysioNet 2017: keep original 300Hz
    sampling_rate = 300  # Hz
    
    # Hannun et al. used 30-second ECG segments
    # PhysioNet 2017 has variable length recordings (9-60s, mean ~30s)
    input_length = 9000  # 30s * 300Hz = 9000 samples
    
    # Model architecture parameters
    # Hannun et al.: "34 layers... 16 residual blocks with 2 convolutional layers per block"
    n_classes = 4  # PhysioNet 2017: Normal, AF, Other, Noisy
    
    # Network architecture from Hannun et al. (Extended Data Figure 1)
    # "convolutional layers have a filter width of 16"
    kernel_size = 16
    
    # "32*2^k filters, where k starts at zero and incremented by one every fourth residual block"
    base_filters = 32
    
    # Dropout rate from Hannun et al.: "applied Dropout with probability of 0.2"
    dropout_rate = 0.2
    
    # Training parameters
    # Hannun et al.: "Adam optimizer with default parameters (β1=0.9 and β2=0.999)"
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    
    # Hannun et al.: "minibatch size of 128"
    batch_size = 128
    
    # Hannun et al.: "reduced [learning rate] by factor of ten when development-set loss 
    # stopped improving for two consecutive epochs"
    lr_patience = 2
    lr_factor = 0.1
    
    # Maximum training epochs (not explicitly stated in paper, typically 50-100)
    max_epochs = 100
    
    # Early stopping patience
    early_stopping_patience = 10
    
    # Data split
    # Hannun et al.: "held-out records from random 10% of training dataset patients 
    # for use as development dataset"
    val_split = 0.1
    
    # Number of workers for data loading
    num_workers = 4
    
    # Random seed for reproducibility
    seed = 42
    
    # Paths
    data_dir = './data/physionet2017'
    model_save_path = './checkpoints'
    log_dir = './logs'